"""Run: python app.py, then open https://192.168.100.96:8500."""
import asyncio
from contextlib import asynccontextmanager
import json
import os
import tempfile
import subprocess
from math import gcd
from pathlib import Path
import threading
from time import perf_counter

import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import pipeline
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from live_vad import SpeechDetector
from noise_suppression import BrowserNoiseSuppressor
from speaker_filter import SpeakerFilter
from loudness_gate import measure
from openai_voice import router as openai_voice_router

ROOT = Path(__file__).resolve().parent
lock = threading.Lock()
speaker_filter = SpeakerFilter()
PROFILE = Path(os.environ.get("TARGET_SPEAKER", str(ROOT / "target_speaker.pt")))
DEFAULT_THRESHOLD = float(os.environ.get("SPEAKER_THRESHOLD", "0.5"))
MIN_DBFS = -40  # Compatibility value for the diagnostic meter only.


@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(speaker_filter.load)
    cuda = torch.cuda.is_available()
    app.state.transcriber = pipeline(
        "automatic-speech-recognition", model=str(ROOT / "model"),
        device=0 if cuda else -1, dtype=torch.float16 if cuda else torch.float32,
    )
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(openai_voice_router)


@app.get("/")
async def home():
    return FileResponse(ROOT / "index.html")


@app.get("/certificate")
async def certificate():
    return FileResponse(ROOT / "certs" / "ca.crt", filename="sharif-whisper-ca.crt")


@app.get("/health")
async def health():
    return {"ready": True, "gpu": torch.cuda.is_available(), "speaker_enrolled": PROFILE.is_file(), "speaker_threshold": DEFAULT_THRESHOLD, "min_dbfs": MIN_DBFS}


@app.post("/enroll")
async def enroll(file: UploadFile = File(...)):
    contents = await file.read(32 * 1024 * 1024 + 1)
    await file.close()
    if len(contents) > 32 * 1024 * 1024:
        raise HTTPException(400, "WAV file must be smaller than 32 MB")
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav") as temporary:
            temporary.write(contents)
            temporary.flush()
            duration = await asyncio.to_thread(speaker_filter.enroll, temporary.name, PROFILE)
        return {"enrolled": True, "duration": duration}
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise HTTPException(400, str(exc)) from exc


def transcribe(audio, rate, threshold=DEFAULT_THRESHOLD):
    if not speaker_filter.matches(audio, rate, PROFILE, threshold):
        return ""

    if rate != 16000:
        divisor = gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, rate // divisor)
        rate = 16000
    started = perf_counter()
    with lock, torch.inference_mode():
        result = app.state.transcriber(
            {"array": audio.astype(np.float32), "sampling_rate": 16000},
            generate_kwargs={"language": "persian", "task": "transcribe"},
        )
    print(f"Speech duration: {len(audio) / rate:.2f}s | Transcription time: {perf_counter() - started:.2f}s", flush=True)
    return result["text"].strip()


@app.websocket("/listen")
async def listen(ws: WebSocket):
    await ws.accept()
    suppressor = None
    try:
        config = await ws.receive_json()
        rate = int(config["sample_rate"])
        if rate not in (16000, 22050, 24000, 32000, 44100, 48000, 96000):
            raise ValueError("Unsupported microphone sample rate")
        threshold = float(config.get("speaker_threshold", DEFAULT_THRESHOLD))
        if not np.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("Similarity threshold must be between -1 and 1")
        calibration = config.get("calibration", False) is True
        filter_enabled = PROFILE.is_file()
        if not calibration and not filter_enabled:
            raise ValueError("Enroll the target first: upload a clean 15–30s WAV or run python enroll_speaker.py patient.wav")
        input_rate = rate
        rate = 16000
        suppressor = await asyncio.to_thread(BrowserNoiseSuppressor, input_rate)
        detector = SpeechDetector(rate)
        audio = np.empty(0, dtype=np.float32)
        committed = []
        silence = 0
        since_update = 0
        meter_samples = 0
        has_speech = False

        async def update(final=False):
            nonlocal audio, has_speech, since_update, silence
            if calibration:
                if final and has_speech:
                    rms, dbfs = measure(audio)
                    print(f'Calibration speech: RMS={rms:.6f}, dBFS={dbfs:.2f}', flush=True)
                    await ws.send_json({"calibration_level": dbfs if np.isfinite(dbfs) else None, "rms": rms})
                text = ""
            else:
                if not final:
                    return
                text = await asyncio.to_thread(transcribe, audio, rate, threshold) if has_speech else ""
            if final:
                if text:
                    committed.append(text)
                audio = np.empty(0, dtype=np.float32)
                has_speech = False
                silence = 0
            since_update = 0
            await ws.send_json({"text": "\n".join(committed), "partial": "" if final else text})

        async def accept_audio(chunk):
            nonlocal audio, has_speech, silence, since_update, meter_samples
            if not chunk.size:
                return
            meter_samples += chunk.size
            if meter_samples >= rate / 5:
                _, level = measure(chunk)
                await ws.send_json({"denoised_dbfs": level if np.isfinite(level) else None})
                meter_samples = 0
            speaking = detector.feed(chunk)
            if speaking:
                has_speech = True
            silence = detector.silent_samples * rate / 16000
            # Keep only a short lead-in while waiting for speech.
            audio = np.concatenate((audio, chunk))
            if not has_speech:
                audio = audio[-rate // 3:]
                return
            since_update += chunk.size
            if silence >= rate * 0.75 or audio.size >= rate * 20:
                await update(final=True)
            elif since_update >= rate * 2:
                await update()

        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("text"):
                if json.loads(message["text"]).get("type") == "stop":
                    tail = await asyncio.to_thread(suppressor.process, np.empty(0, dtype=np.float32), True)
                    await accept_audio(tail)
                    await update(final=True)
                    await ws.send_json({"stopped": True})
                    await ws.close()
                    break
                continue
            raw = message.get("bytes", b"")
            if len(raw) % 4 or len(raw) > input_rate * 4:
                raise ValueError("Invalid audio packet")
            chunk = np.frombuffer(raw, dtype="<f4")
            if not np.isfinite(chunk).all():
                raise ValueError("Invalid audio samples")
            cleaned = await asyncio.to_thread(suppressor.process, chunk)
            await accept_audio(cleaned)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_json({"error": str(exc)})
            await ws.close(code=1011)
        except RuntimeError:
            pass

    finally:
        if suppressor is not None:
            await asyncio.to_thread(suppressor.close)


def free_port(port):
    """Stop the old server before loading another copy of the GPU model."""
    import shutil
    import time

    fuser = shutil.which("fuser")
    if not fuser:
        raise RuntimeError("Port cleanup requires fuser (Ubuntu package: psmisc)")
    target = f"{port}/tcp"

    def occupied():
        result = subprocess.run([fuser, target], capture_output=True)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode(errors="replace").strip())
        return result.returncode == 0

    if not occupied():
        return
    print(f"Port {port} is in use; stopping the previous process…", flush=True)
    subprocess.run([fuser, "-k", "-TERM", target], capture_output=True)
    for _ in range(50):
        if not occupied():
            return
        time.sleep(0.1)
    print(f"Process on port {port} did not stop; forcing termination…", flush=True)
    subprocess.run([fuser, "-k", "-KILL", target], capture_output=True)
    for _ in range(20):
        if not occupied():
            return
        time.sleep(0.1)
    raise RuntimeError(f"Cannot free port {port}; check process ownership")


if __name__ == "__main__":
    import uvicorn
    free_port(8500)
    uvicorn.run(
        app, host="0.0.0.0", port=8500,
        ssl_keyfile=str(ROOT / "certs" / "server.key"),
        ssl_certfile=str(ROOT / "certs" / "server.crt"),
    )
