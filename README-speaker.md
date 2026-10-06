Target-speaker verification
===========================

Use the existing environment on Sharif:

```bash
cd ~/Music/whisper
source .venv/bin/activate
python enroll_speaker.py patient.wav
python app.py
```

Provide 15–30 seconds of clean speech in WAV format. The embedding is saved to
`target_speaker.pt`. You can also click the microphone enrollment button and speak for 20 seconds in the existing frontend at
https://192.168.100.96:8500/. Enrollment is required before transcription.

Set the speaker similarity threshold in the frontend enrollment section
(default 0.5). Stop and restart recording to apply changes. Higher values require
a closer match. Live dBFS meters remain diagnostic; loudness does not gate speech.

Terminal microphone mode:

```bash
python realtime_stt.py --speaker-threshold 0.5
```

ECAPA and Whisper remain loaded. Every completed Silero utterance passes through
speaker verification before Whisper. Other speakers are ignored.

Frontend default: `SPEAKER_THRESHOLD=0.6 python app.py`.
Alternate profile: `TARGET_SPEAKER=/path/profile.pt python app.py` or terminal
`--speaker /path/profile.pt`. Enrollment supports `--output /path/profile.pt`.

Tests: `python -m unittest discover -v`. Official SpeechBrain samples tested
a held-out target voice, another speaker, and attenuated other speech simulating
distance through the complete pipeline. Actual room/voice validation still needs
your enrollment and comparison recordings.
