"""GPT-Live WebRTC signaling; keys are used for one request and never returned.

Official protocol: https://developers.openai.com/api/docs/guides/voice-webrtc?api=live
Events/history: https://developers.openai.com/api/docs/guides/live-conversations
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Literal
import urllib.error
import urllib.request

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, SecretStr

router = APIRouter()
MODEL = 'gpt-live-1'
API_URL = 'https://api.openai.com/v1/live/sessions'
INSTRUCTIONS = '''شما آوا، دستیار مکالمه‌ای فارسی هستید. فارسی طبیعی و روشن صحبت کنید.
پاسخ‌ها را کوتاه نگه دارید. جزئیات قبلی مکالمه و اصلاحات کاربر را به خاطر بسپارید.
اگر صدای کاربر نامفهوم است، پرسش کوتاهی برای روشن شدن آن بپرسید.
Backchannel policy: تأییدهای کوتاه و طبیعی داشته باشید و با گفتار کاربر رقابت نکنید.
Interruption policy: وقتی کاربر صحبت شما را قطع کرد، پاسخ را متوقف کنید و به او گوش دهید.
Delegation policy:
Backend tools: یک مدل متنی برای پاسخ به پرسش‌ها و استدلال؛ ابزار خارجی وجود ندارد.
Delegate to the backend when: پاسخ نیاز به استدلال یا توضیح دقیق دارد.
Do not delegate to the backend when: احوال‌پرسی، پرسش روشن‌کننده یا تکرار پاسخ قبلی کافی است.
نتیجه‌ای را که هنوز از مدل متنی دریافت نکرده‌اید حدس نزنید.'''


class HistoryMessage(BaseModel):
    role: Literal['user', 'assistant']
    text: str = Field(min_length=1, max_length=4096)


class Offer(BaseModel):
    api_key: SecretStr | None = Field(default=None, max_length=512, repr=False)
    sdp: str = Field(min_length=1, max_length=50000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=64)


def session_payload(offer):
    if sum(len(m.text.encode('utf-8')) for m in offer.history) > 4096:
        raise HTTPException(400, 'Conversation history is too large')
    return {
        'session': {
            'model': MODEL,
            'instructions': INSTRUCTIONS,
            'audio': {'output': {'voice': 'marin'}},
            'input': [
                {'type': 'message', 'role': m.role,
                 'content': [{'type': 'input_text' if m.role == 'user' else 'output_text', 'text': m.text}]}
                for m in offer.history
            ],
            'delegation': {
                'type': 'responses',
                'responses': {
                    'model': os.environ.get('OPENAI_LIVE_BACKEND_MODEL', 'gpt-6-luna'),
                    'instructions': 'به پرسش کاربر با توجه به تاریخچه مکالمه، کوتاه و دقیق و فقط به فارسی پاسخ بده. ابزار خارجی در دسترس نیست.',
                },
            },
        },
        'transport': {'type': 'webrtc', 'sdp': offer.sdp},
    }


def exchange_offer(payload, key):
    request = urllib.request.Request(
        API_URL, data=json.dumps(payload).encode('utf-8'), method='POST',
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    try:
        # No automatic retry: creating another session can incur another charge.
        with urllib.request.urlopen(request, timeout=45) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        # Neither raw upstream errors nor request headers reach the frontend/logs.
        messages = {401: 'OpenAI API key was rejected', 403: 'OpenAI access was denied',
                    404: 'GPT-Live is not available to this project',
                    429: 'OpenAI quota or rate limit reached'}
        raise HTTPException(exc.code if exc.code in messages else 502,
                            messages.get(exc.code, 'OpenAI Live session creation failed')) from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        raise HTTPException(502, 'Could not connect to OpenAI Live') from None
    try:
        session_id = result['session']['id']
        sdp = result['transport']['sdp']
        if not isinstance(session_id, str) or not isinstance(sdp, str) or not sdp.strip():
            raise ValueError()
    except (KeyError, TypeError, ValueError):
        raise HTTPException(502, 'Invalid OpenAI Live session response') from None
    return {'session': {'id': session_id}, 'transport': {'type': 'webrtc', 'sdp': sdp}}


@router.get('/openai-live/config')
async def config():
    return {'model': MODEL, 'configured': bool(os.environ.get('OPENAI_API_KEY'))}


@router.get('/openai_live.js')
async def javascript():
    return FileResponse(Path(__file__).resolve().parent / 'openai_live.js', media_type='application/javascript')


@router.post('/openai-live/session')
async def create_session(request: Request):
    if request.headers.get('origin') != f'{request.url.scheme}://{request.url.netloc}':
        raise HTTPException(403, 'Unexpected request origin')
    contents = bytearray()
    async for chunk in request.stream():
        contents.extend(chunk)
        if len(contents) > 65536:
            raise HTTPException(413, 'Session request is too large')
    try:
        offer = Offer.model_validate_json(contents)
    except ValueError:
        raise HTTPException(400, 'A valid SDP offer and conversation history are required') from None
    if not offer.sdp.strip().startswith('v=0'):
        raise HTTPException(400, 'A valid SDP offer is required')
    supplied = offer.api_key.get_secret_value().strip() if offer.api_key else ''
    if supplied and any(not 33 <= ord(c) <= 126 for c in supplied):
        raise HTTPException(400, 'Invalid API key format')
    if supplied and request.url.scheme != 'https':
        raise HTTPException(400, 'Use HTTPS to submit an API key')
    key = supplied or os.environ.get('OPENAI_API_KEY')
    if not key:
        raise HTTPException(503, 'Enter your OpenAI API key in the frontend or set OPENAI_API_KEY on the server')
    result = await asyncio.to_thread(exchange_offer, session_payload(offer), key)
    return JSONResponse(result, status_code=201, headers={'Cache-Control': 'no-store'})
