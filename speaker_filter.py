"""ECAPA enrollment and cosine verification; no changes to Whisper or Silero."""
import argparse
from math import gcd
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from silero_vad import load_silero_vad, get_speech_timestamps

ROOT = Path(__file__).resolve().parent
MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"


def decode_audio(path):
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
         "-t", "31", "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
        capture_output=True, check=True,
    )
    return np.frombuffer(result.stdout, dtype="<f4").copy()


class SpeakerFilter:
    def __init__(self):
        self.encoder = None
        self.lock = threading.RLock()
        self.target_cache = None

    def load(self):
        with self.lock:
            if self.encoder is None:
                from speechbrain.inference.speaker import EncoderClassifier
                self.encoder = EncoderClassifier.from_hparams(
                    source=MODEL_ID, savedir=str(ROOT / "speaker_model"),
                    run_opts={"device": "cuda:0" if torch.cuda.is_available() else "cpu"},
                )
        return self

    def embedding(self, audio, rate=16000):
        audio = np.asarray(audio, dtype=np.float32)
        if rate != 16000:
            divisor = gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, rate // divisor)
        if audio.size < 6400 or not np.isfinite(audio).all():
            raise ValueError("Speaker audio must contain at least 0.4 seconds of finite samples")
        with self.lock, torch.inference_mode():
            self.load()
            wave = torch.from_numpy(audio.copy()).unsqueeze(0)
            embedding = self.encoder.encode_batch(wave).flatten().float().cpu()
            if embedding.numel() != 192 or not torch.isfinite(embedding).all():
                raise ValueError("ECAPA returned an invalid embedding")
            return F.normalize(embedding, dim=0)

    def enroll(self, path, destination):
        with open(path, "rb") as wav:
            header = wav.read(12)
        if header[:4] not in (b"RIFF", b"RF64") or header[8:12] != b"WAVE":
            raise ValueError("Enrollment must be a WAV recording")
        audio = decode_audio(path)
        seconds = audio.size / 16000
        if not 15 <= seconds <= 30:
            raise ValueError(f"Enrollment must be 15–30 seconds; received {seconds:.1f}s")
        timestamps = get_speech_timestamps(torch.from_numpy(audio), load_silero_vad(), sampling_rate=16000)
        if not timestamps:
            raise ValueError("No speech detected in enrollment")
        speech = np.concatenate([audio[s['start']:s['end']] for s in timestamps])
        if speech.size < 5 * 16000:
            raise ValueError("Enrollment must contain at least 5 seconds of detected speech")
        embedding = self.embedding(speech)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            temporary = destination.with_name(destination.name + ".tmp")
            torch.save({"model": MODEL_ID, "embedding": embedding}, temporary)
            temporary.replace(destination)
        return seconds

    def matches(self, audio, rate, profile, threshold=0.5):
        if not np.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("Similarity threshold must be between -1 and 1")
        if len(audio) / rate < 0.4:
            print("[Speaker score: unavailable] TOO SHORT → ignored", file=sys.stderr, flush=True)
            return False
        profile = Path(profile).resolve()
        stamp = profile.stat().st_mtime_ns
        with self.lock:
            if self.target_cache is None or self.target_cache[:2] != (profile, stamp):
                self.target_cache = (profile, stamp, torch.load(profile, map_location="cpu", weights_only=True))
            target = self.target_cache[2]
        if target.get("model") != MODEL_ID:
            raise ValueError("Enrollment uses a different speaker model")
        embedding = target["embedding"].flatten()
        if embedding.numel() != 192 or not torch.isfinite(embedding).all() or embedding.norm() == 0:
            raise ValueError("Invalid enrolled embedding")
        score = float(F.cosine_similarity(self.embedding(audio, rate), embedding, dim=0))
        accepted = score >= threshold
        print(f"[Speaker score: {score:.4f}] {'TARGET → transcribing...' if accepted else 'OTHER → ignored'} (threshold {threshold:.2f})", flush=True)
        return accepted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a target speaker from a clean 15–30s WAV")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "target_speaker.pt")
    args = parser.parse_args()
    duration = SpeakerFilter().enroll(args.audio, args.output)
    print(f"Enrolled {duration:.1f}s recording -> {args.output}")
