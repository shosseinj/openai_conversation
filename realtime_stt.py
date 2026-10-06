"""Continuous microphone -> DeepFilterNet -> existing Silero -> existing Whisper. Ctrl+C to stop."""
import argparse
from collections import deque
import queue
import shutil
import subprocess
import sys
import threading
from time import perf_counter

import numpy as np
from live_vad import SpeechDetector
from transcribe import load_transcriber
from noise_suppression import NoiseSuppressor, denoised_frames
from loudness_gate import measure, print_calibration
from speaker_filter import SpeakerFilter, ROOT
from pathlib import Path

RATE = 16000
FRAME = 512


def utterances(frames, detector, pause=0.5, on_start=None):
    lead = deque(maxlen=6)  # Preserve about 190 ms before speech is detected.
    active = None
    size = 0
    for frame in frames:
        speaking = detector.feed(frame)
        if active is None:
            if not speaking:
                lead.append(frame)
                continue
            active = list(lead)
            lead.clear()
            size = sum(len(chunk) for chunk in active)
            if on_start:
                on_start()
        active.append(frame)
        size += len(frame)
        ended = detector.silent_samples >= RATE * pause
        if ended or size >= RATE * 25:
            audio = np.concatenate(active)
            # Retain 64 ms of padding, rather than sending the entire silent pause.
            trim = max(0, detector.silent_samples - 1024)
            if trim:
                audio = audio[:-trim]
            if len(audio) >= RATE * 0.25:
                yield audio
            lead.extend(active[-6:])
            active = None
            size = 0
    if active:
        audio = np.concatenate(active)
        trim = max(0, detector.silent_samples - 1024)
        if trim:
            audio = audio[:-trim]
        if len(audio) >= RATE * 0.25:
            yield audio


def microphone_frames(recorder, frame_size=FRAME):
    while True:
        raw = recorder.stdout.read(frame_size * 2)
        if not raw:
            raise RuntimeError("Microphone capture ended; check arecord output and input device")
        audio = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
        if len(audio) < frame_size:
            audio = np.pad(audio, (0, frame_size - len(audio)))
        yield audio


def transcribe_utterance(model, audio, verifier, profile, threshold=0.5):
    if not verifier.matches(audio, RATE, profile, threshold):
        return ""
    started = perf_counter()
    result = model(
        {"array": audio, "sampling_rate": RATE},
        generate_kwargs={"language": "persian", "task": "transcribe"},
    )
    elapsed = perf_counter() - started
    print(result["text"].strip(), flush=True)
    print(f"Speech duration: {len(audio) / RATE:.2f}s | Transcription time: {elapsed:.2f}s", flush=True)
    return result["text"].strip()


def calibrate(device, pause, samples):
    levels = []
    print('Keep microphone gain fixed. Calibration measures denoised speech with no normalization.', flush=True)
    for label in ('Close voice', 'Distant speech'):
        input(f'{label}: prepare the speaker position, then press Enter. Speak {samples} sentences with pauses. ')
        suppressor = NoiseSuppressor()
        detector = SpeechDetector(RATE)
        recorder = subprocess.Popen(
            ['arecord', '-q', '-D', device, '-t', 'raw', '-f', 'S16_LE',
             '-r', '48000', '-c', '1'], stdout=subprocess.PIPE)
        values = []
        try:
            stream = denoised_frames(microphone_frames(recorder, 480), suppressor)
            for audio in utterances(stream, detector, pause):
                rms, dbfs = measure(audio)
                values.append(dbfs)
                print(f'{label} {len(values)}/{samples}: RMS={rms:.6f}, dBFS={dbfs:.2f}', flush=True)
                if len(values) == samples:
                    break
        finally:
            recorder.terminate()
            try:
                recorder.wait(timeout=3)
            except subprocess.TimeoutExpired:
                recorder.kill()
                recorder.wait()
            recorder.stdout.close()
            suppressor.close()
        levels.append(values)
    print_calibration(*levels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='default', help='ALSA capture device (see arecord -L)')
    parser.add_argument('--pause', type=float, default=0.5, help='Seconds of silence ending an utterance')
    parser.add_argument('--speaker', type=Path, default=ROOT / 'target_speaker.pt', help='Enrolled speaker embedding')
    parser.add_argument('--speaker-threshold', type=float, default=0.5, help='Minimum cosine similarity for the target speaker')
    parser.add_argument('--calibrate', action='store_true', help='Measure close and distant voice levels without running Whisper')
    parser.add_argument('--samples', type=int, default=5, help='Utterances per calibration phase')
    args = parser.parse_args()
    if not np.isfinite(args.speaker_threshold) or not -1 <= args.speaker_threshold <= 1:
        parser.error('Speaker threshold must be between -1 and 1')
    if not args.calibrate and not args.speaker.is_file():
        parser.error('Enroll first: python enroll_speaker.py patient.wav')
    if not 1 <= args.samples <= 30:
        parser.error('--samples must be between 1 and 30')
    if not 0.1 <= args.pause <= 3:
        parser.error('--pause must be between 0.1 and 3 seconds')
    if not shutil.which('arecord'):
        parser.error('arecord is required (Ubuntu package: alsa-utils)')
    if args.calibrate:
        calibrate(args.device, args.pause, args.samples)
        return 0
    print('Loading ECAPA speaker model once…', flush=True)
    verifier = SpeakerFilter().load()
    print('Loading stateful DeepFilterNet once (48 kHz CPU streaming)…', flush=True)
    suppressor = NoiseSuppressor()
    print('Loading Whisper once…', flush=True)
    model = load_transcriber()
    detector = SpeechDetector(RATE)
    jobs = queue.Queue(maxsize=8)
    failures = queue.Queue()

    def worker():
        while True:
            audio = jobs.get()
            try:
                if audio is None:
                    return
                transcribe_utterance(model, audio, verifier, args.speaker, args.speaker_threshold)
            except Exception as exc:
                failures.put(exc)
            finally:
                jobs.task_done()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    recorder = subprocess.Popen(
        ['arecord', '-q', '-D', args.device, '-t', 'raw', '-f', 'S16_LE',
         '-r', str(suppressor.sample_rate), '-c', '1'], stdout=subprocess.PIPE,
    )
    print('Listening: microphone 48 kHz mono -> DeepFilterNet -> 16 kHz Silero/Whisper. Press Ctrl+C to stop.', flush=True)
    try:
        for audio in utterances(denoised_frames(microphone_frames(recorder, suppressor.frame_size), suppressor), detector, args.pause,
                                lambda: print('Speech started', flush=True)):
            if not failures.empty():
                raise failures.get()
            print(f'Speech ended ({len(audio) / RATE:.2f}s); verifying speaker…', flush=True)
            try:
                jobs.put_nowait(audio)
            except queue.Full:
                raise RuntimeError('Transcription cannot keep up; utterance queue is full')
    except KeyboardInterrupt:
        print('\nStopping microphone…', flush=True)
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr, flush=True)
        return_code = 1
    else:
        return_code = 0
    finally:
        recorder.terminate()
        try:
            recorder.wait(timeout=3)
        except subprocess.TimeoutExpired:
            recorder.kill()
            recorder.wait()
        recorder.stdout.close()
        suppressor.close()
        jobs.put(None)
        thread.join()
    if not failures.empty():
        raise RuntimeError('Transcription failed') from failures.get()
    return locals().get('return_code', 0)


if __name__ == '__main__':
    raise SystemExit(main())
