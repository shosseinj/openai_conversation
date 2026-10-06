import asyncio
from pathlib import Path
import unittest
from unittest.mock import patch, Mock
import numpy as np
import app
from diagnose_pipeline import decode, errors, frontend_segments, normalize, levels


class DiagnosticTests(unittest.TestCase):
    def test_persian_normalization_and_errors(self):
        self.assertEqual(normalize('  كِتاب، مي\u200cروم! '), 'کتاب می روم')
        self.assertEqual(errors('كِتاب', 'کتاب')['wer'], 0)
        self.assertEqual(errors('من خوبم', 'من')['wer'], .5)
        self.assertEqual(errors('من', '')['cer'], 1)
        with self.assertRaises(ValueError): errors('،', 'متن')

    def test_level_statistics(self):
        stats = levels(np.full(16000, .1, dtype=np.float32))
        self.assertAlmostEqual(stats['dbfs'], -20, places=5)
        self.assertEqual(stats['duration'], 1)
        self.assertEqual(stats['near_full_scale_samples'], 0)
        self.assertEqual(levels(np.zeros(10))['dbfs'], None)

    def test_frontend_segments_match_actual_app_handler(self):
        speech = decode(Path(__file__).parent / 'persian_sample.wav', 16000)
        audio = np.concatenate((np.zeros(16000), np.tile(speech, 7), np.zeros(16000), speech)).astype(np.float32)
        chunks = [audio[i:i + 1360] for i in range(0, len(audio), 1360)]
        tail = np.zeros(960, dtype=np.float32)
        expected = frontend_segments(chunks + [tail])
        captured = []
        class Suppressor:
            def __init__(self, rate): pass
            def process(self, chunk, final=False): return tail if final else chunk
            def close(self): pass
        class Socket:
            async def accept(self): pass
            async def receive_json(self): return {'sample_rate':16000}
            async def receive(self):
                if self.i < len(chunks):
                    data = chunks[self.i]; self.i += 1
                    return {'type':'websocket.receive','bytes':data.tobytes()}
                return {'type':'websocket.receive','text':'{"type":"stop"}'}
            async def send_json(self, data):
                if 'error' in data: raise AssertionError(data['error'])
            async def close(self, **kwargs): pass
            i = 0
        def transcribe(audio, rate, threshold):
            captured.append(audio.copy()); return 'متن'
        profile = Mock();profile.is_file.return_value = True
        with patch.object(app,'BrowserNoiseSuppressor',Suppressor), patch.object(app,'transcribe',transcribe), patch.object(app,'PROFILE',profile):
            asyncio.run(app.listen(Socket()))
        self.assertGreaterEqual(len(expected), 3)
        self.assertGreaterEqual(len(expected[0][2]), 20 * 16000)
        self.assertEqual(len(expected), len(captured))
        for segment, actual in zip(expected, captured):
            np.testing.assert_array_equal(segment[2], actual)


if __name__ == '__main__': unittest.main()
