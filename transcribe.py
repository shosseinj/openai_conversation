import argparse
from pathlib import Path
import subprocess

import numpy as np
import torch
from transformers import pipeline


def load_transcriber():
    cuda = torch.cuda.is_available()
    return pipeline(
        "automatic-speech-recognition",
        model=str(Path(__file__).resolve().parent / "model"),
        device=0 if cuda else -1,
        dtype=torch.float16 if cuda else torch.float32,
    )


def main():
    parser = argparse.ArgumentParser(description="Transcribe local audio into Persian.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--speaker", type=Path, help="Saved target speaker embedding")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not -1 <= args.threshold <= 1:
        parser.error("Threshold must be between -1 and 1")
    if args.speaker and not args.speaker.is_file():
        parser.error(f"Enrollment not found: {args.speaker}")
    if not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")

    try:
        decoded = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(args.audio),
             "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
            check=True, capture_output=True,
        )
    except FileNotFoundError:
        parser.error("ffmpeg is required but was not found in PATH")
    except subprocess.CalledProcessError as exc:
        parser.error(exc.stderr.decode(errors="replace").strip())
    audio = np.frombuffer(decoded.stdout, dtype="<f4").copy()
    if not audio.size:
        parser.error("Audio file is empty")
    transcriber = load_transcriber()
    if args.speaker:
        from silero_vad import load_silero_vad, get_speech_timestamps
        from speaker_filter import SpeakerFilter
        verifier = SpeakerFilter()
        segments = get_speech_timestamps(torch.from_numpy(audio), load_silero_vad(), sampling_rate=16000)
        for segment in segments:
            speech = audio[segment["start"]:segment["end"]]
            if verifier.matches(speech, 16000, args.speaker, args.threshold):
                result = transcriber(
                    {"array": speech, "sampling_rate": 16000},
                    generate_kwargs={"language": "persian", "task": "transcribe"},
                )
                print(result["text"].strip())
    else:
        result = transcriber(
            {"array": np.asarray(audio, dtype=np.float32), "sampling_rate": 16000},
            return_timestamps=True,
            generate_kwargs={"language": "persian", "task": "transcribe"},
        )
        print(result["text"].strip())



if __name__ == "__main__":
    main()
