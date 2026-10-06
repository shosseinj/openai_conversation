import unittest
from unittest.mock import Mock
import numpy as np
from loudness_gate import measure, passes, validate_threshold


class LoudnessTests(unittest.TestCase):
    def test_known_dbfs_and_unchanged_samples(self):
        audio = np.full(16000, .1, dtype=np.float32)
        original = audio.copy()
        rms, dbfs = measure(audio)
        self.assertAlmostEqual(rms, .1, places=6)
        self.assertAlmostEqual(dbfs, -20, places=5)
        self.assertTrue(passes(audio, -30))
        self.assertFalse(passes(audio, -10))
        np.testing.assert_array_equal(audio, original)

    def test_silence_and_invalid_threshold(self):
        self.assertEqual(measure(np.zeros(512)), (0, float('-inf')))
        self.assertFalse(passes(np.zeros(512), -120))
        for value in (float('nan'), float('inf'), 1, -121):
            with self.assertRaises(ValueError): validate_threshold(value)


if __name__ == '__main__':
    unittest.main()
