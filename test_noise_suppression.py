import unittest
import numpy as np
from noise_suppression import NoiseSuppressor, denoised_frames
from realtime_stt import utterances
from live_vad import SpeechDetector


class NoiseSuppressionTests(unittest.TestCase):
    def test_streaming_noise_reduction_and_no_false_utterances(self):
        rng = np.random.default_rng(42)
        noisy = rng.normal(0, .03, 96000).astype(np.float32)
        suppressor = NoiseSuppressor()
        try:
            result = np.concatenate(list(denoised_frames(
                (noisy[i:i + 480] for i in range(0, len(noisy), 480)), suppressor)))
            self.assertTrue(np.isfinite(result).all())
            self.assertLess(abs(len(result) - len(noisy) // 3), 512)
            original_rms = np.sqrt(np.mean(noisy[48000:] ** 2))
            reduced_rms = np.sqrt(np.mean(result[16000:32000] ** 2))
            self.assertGreater(20 * np.log10(original_rms / max(reduced_rms, 1e-10)), 6)
            segments = utterances((result[i:i + 512] for i in range(0, len(result), 512)), SpeechDetector(16000))
            self.assertEqual(list(segments), [])
        finally:
            suppressor.close()

    def test_invalid_input_is_rejected(self):
        suppressor = NoiseSuppressor()
        try:
            with self.assertRaises(ValueError):
                suppressor.process(np.zeros(512, dtype=np.float32))
            with self.assertRaises(ValueError):
                suppressor.process(np.full(480, np.nan, dtype=np.float32))
        finally:
            suppressor.close()


if __name__ == '__main__':
    unittest.main()
