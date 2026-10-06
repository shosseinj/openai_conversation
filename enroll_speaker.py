"""Enroll the target: python enroll_speaker.py patient.wav"""
import argparse
from pathlib import Path
import subprocess
from speaker_filter import SpeakerFilter, ROOT


def main():
    parser = argparse.ArgumentParser(description='Enroll a speaker from 15–30 seconds of clean WAV speech')
    parser.add_argument('audio', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'target_speaker.pt')
    args = parser.parse_args()
    try:
        duration = SpeakerFilter().enroll(args.audio, args.output)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(f'Saved target speaker embedding: {args.output} ({duration:.1f}s recording)', flush=True)


if __name__ == '__main__':
    main()
