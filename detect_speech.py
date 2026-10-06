"""Print speech start/end times: python detect_speech.py test.wav"""
import argparse
from pathlib import Path
import subprocess

import numpy as np
import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


def main():
    parser = argparse.ArgumentParser(description="Detect speech segments with Silero VAD.")
    parser.add_argument("audio", nargs="?", type=Path, default=Path("test.wav"))
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")
    torch.set_num_threads(1)
    model = load_silero_vad()
    try:
        audio = read_audio(str(args.audio), sampling_rate=16000)
    except (RuntimeError, ValueError, ImportError):
        # Existing FFmpeg handles MP3 and files with unknown-length headers too.
        try:
            result = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-i", str(args.audio),
                 "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
                check=True, capture_output=True,
            )
        except FileNotFoundError:
            parser.error("ffmpeg is required for audio decoding")
        except subprocess.CalledProcessError as exc:
            parser.error(exc.stderr.decode(errors="replace").strip())
        audio = torch.from_numpy(np.frombuffer(result.stdout, dtype="<f4").copy())
    if not audio.numel():
        parser.error("Audio file is empty")
    speech_timestamps = get_speech_timestamps(
        audio, model, sampling_rate=16000, return_seconds=True,
    )
    print("Detected speech:")
    for segment in speech_timestamps:
        print(f"{segment['start']:.2f}s -> {segment['end']:.2f}s")
    if not speech_timestamps:
        print("No speech detected.")


if __name__ == "__main__":
    main()
