"""Per-recording Silero state for browser PCM audio."""
from math import gcd

import numpy as np
import torch
from scipy.signal import resample_poly
from silero_vad import load_silero_vad


class SpeechDetector:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.model = load_silero_vad()
        self.pending = np.empty(0, dtype=np.float32)
        self.silent_samples = 0
        self.in_speech = False

    def feed(self, chunk):
        if self.sample_rate != 16000:
            divisor = gcd(self.sample_rate, 16000)
            chunk = resample_poly(chunk, 16000 // divisor, self.sample_rate // divisor)
        self.pending = np.concatenate((self.pending, chunk.astype(np.float32)))
        detected = False
        with torch.inference_mode():
            while self.pending.size >= 512:
                frame = self.pending[:512].copy()
                self.pending = self.pending[512:]
                probability = float(self.model(torch.from_numpy(frame), 16000))
                # Hysteresis keeps quiet word endings inside a speech segment.
                self.in_speech = probability >= (0.35 if self.in_speech else 0.5)
                if self.in_speech:
                    detected = True
                    self.silent_samples = 0
                else:
                    self.silent_samples += 512
        return detected
