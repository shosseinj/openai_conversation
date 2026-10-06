import subprocess
import unittest
import numpy as np
from live_vad import SpeechDetector
from realtime_stt import utterances


def frames(audio):
    for i in range(0, len(audio), 512):
        chunk = audio[i:i + 512]
        if len(chunk) < 512:
            chunk = np.pad(chunk, (0, 512 - len(chunk)))
        yield chunk


class RealTimeTests(unittest.TestCase):
    def test_two_sentences_separated_by_pause(self):
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', 'persian_sample.wav',
                                       '-f', 'f32le', '-ac', '1', '-ar', '16000', 'pipe:1'])
        speech = np.frombuffer(raw, dtype='<f4')
        audio = np.concatenate((np.zeros(16000), speech, np.zeros(16000), speech, np.zeros(16000))).astype(np.float32)
        segments = list(utterances(frames(audio), SpeechDetector(16000)))
        self.assertEqual(len(segments), 2)
        for segment in segments:
            self.assertGreater(len(segment), 32000)
            self.assertLess(len(segment), len(speech) + 8000)

    def test_silence_and_noise_do_not_trigger_transcription(self):
        rng = np.random.default_rng(42)
        audio = np.concatenate((np.zeros(16000), rng.normal(0, .04, 48000))).astype(np.float32)
        self.assertEqual(list(utterances(frames(audio), SpeechDetector(16000))), [])

    def test_last_utterance_is_flushed_at_end(self):
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', 'persian_sample.wav',
                                       '-f', 'f32le', '-ac', '1', '-ar', '16000', 'pipe:1'])
        self.assertEqual(len(list(utterances(frames(np.frombuffer(raw, dtype='<f4')), SpeechDetector(16000)))), 1)


if __name__ == '__main__':
    unittest.main()
