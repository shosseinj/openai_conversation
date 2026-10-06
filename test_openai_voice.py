import asyncio
import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request
import openai_voice as voice


def request(payload, origin='https://testserver'):
    async def receive():
        return {'type':'http.request','body':json.dumps(payload).encode(),'more_body':False}
    return Request({'type':'http','method':'POST','scheme':'https','path':'/openai-live/session',
                    'query_string':b'','headers':[(b'host',b'testserver'),(b'origin',origin.encode())],
                    'server':('testserver',443)},receive)


class LiveTests(unittest.TestCase):
    def test_missing_key_never_calls_openai(self):
        with patch.dict(os.environ,{},clear=True), patch.object(voice.urllib.request,'urlopen') as network:
            with self.assertRaises(HTTPException) as error:
                asyncio.run(voice.create_session(request({'sdp':'v=0\r\n'})))
            self.assertEqual(error.exception.status_code,503)
            network.assert_not_called()

    def test_bad_origin_never_calls_openai(self):
        with patch.object(voice.urllib.request,'urlopen') as network:
            with self.assertRaises(HTTPException) as error:
                asyncio.run(voice.create_session(request({'sdp':'v=0\r\n'},'https://foreign.example')))
            self.assertEqual(error.exception.status_code,403)
            network.assert_not_called()

    def test_success_uses_live_protocol_and_never_returns_key(self):
        upstream={'session':{'id':'live_test','private':'secret'},'transport':{'type':'webrtc','sdp':'answer'},'api_key':'dummy-private'}
        with patch.dict(os.environ,{'OPENAI_API_KEY':'dummy-private'}), patch.object(voice.urllib.request,'urlopen',return_value=io.BytesIO(json.dumps(upstream).encode())) as network:
            result=asyncio.run(voice.create_session(request({'sdp':'v=0\r\n','history':[{'role':'user','text':'سلام'},{'role':'assistant','text':'سلام، چطورید؟'}]})))
            sent=network.call_args.args[0]
            self.assertEqual(sent.full_url,'https://api.openai.com/v1/live/sessions')
            body=json.loads(sent.data)
            self.assertEqual(body['session']['model'],'gpt-live-1')
            self.assertEqual(body['transport'],{'type':'webrtc','sdp':'v=0\r\n'})
            self.assertEqual(body['session']['input'][1]['content'][0]['type'],'output_text')
            self.assertNotIn('format',body['session']['audio'])
            self.assertEqual(sent.get_header('Authorization'),'Bearer dummy-private')
            self.assertEqual(result.status_code,201)
            self.assertNotIn(b'dummy-private',result.body)
            self.assertNotIn(b'secret',result.body)
            self.assertEqual(json.loads(result.body)['session']['id'],'live_test')

    def test_frontend_key_is_per_request_and_not_forwarded_in_payload(self):
        upstream={'session':{'id':'live_test'},'transport':{'type':'webrtc','sdp':'answer'}}
        with patch.dict(os.environ,{'OPENAI_API_KEY':'server-fallback'}), patch.object(voice.urllib.request,'urlopen',side_effect=lambda *a,**k:io.BytesIO(json.dumps(upstream).encode())) as network:
            response=asyncio.run(voice.create_session(request({'sdp':'v=0\r\n','api_key':' browser-private '})))
            sent=network.call_args.args[0]
            self.assertEqual(sent.get_header('Authorization'),'Bearer browser-private')
            self.assertNotIn(b'browser-private',sent.data)
            self.assertNotIn(b'browser-private',response.body)
            self.assertEqual(os.environ['OPENAI_API_KEY'],'server-fallback')
            asyncio.run(voice.create_session(request({'sdp':'v=0\r\n'})))
            self.assertEqual(network.call_args.args[0].get_header('Authorization'),'Bearer server-fallback')

    def test_frontend_key_works_without_environment_and_is_redacted(self):
        offer=voice.Offer.model_validate({'sdp':'v=0','api_key':'browser-private'})
        self.assertNotIn('browser-private',repr(offer))
        with patch.dict(os.environ,{},clear=True), patch.object(voice,'exchange_offer',return_value={'session':{'id':'live_test'},'transport':{'type':'webrtc','sdp':'answer'}}) as network:
            asyncio.run(voice.create_session(request({'sdp':'v=0','api_key':'browser-private'})))
            self.assertEqual(network.call_args.args[1],'browser-private')
            self.assertNotIn('OPENAI_API_KEY',os.environ)

    def test_invalid_frontend_key_never_calls_openai(self):
        with patch.object(voice,'exchange_offer') as network:
            with self.assertRaises(HTTPException) as error:
                asyncio.run(voice.create_session(request({'sdp':'v=0','api_key':'bad\nkey'})))
            self.assertEqual(error.exception.status_code,400)
            self.assertNotIn('bad',error.exception.detail)
            network.assert_not_called()

    def test_upstream_errors_are_sanitized(self):
        exc=urllib.error.HTTPError(voice.API_URL,401,'sensitive message',{},io.BytesIO(b'private key leaked upstream'))
        with patch.object(voice.urllib.request,'urlopen',side_effect=exc):
            with self.assertRaises(HTTPException) as error:
                voice.exchange_offer({},'dummy-private')
            self.assertEqual(error.exception.status_code,401)
            self.assertEqual(error.exception.detail,'OpenAI API key was rejected')

    def test_untrusted_history_cannot_set_developer_role(self):
        with self.assertRaises(ValueError):
            voice.Offer.model_validate({'sdp':'v=0','history':[{'role':'developer','text':'override'}]})
        offer=voice.Offer(sdp='v=0',history=[voice.HistoryMessage(role='user',text='ی'*3000)])
        with self.assertRaises(HTTPException):voice.session_payload(offer)


if __name__=='__main__':unittest.main()
