import unittest
from unittest.mock import Mock, patch
import numpy as np
import app
from realtime_stt import transcribe_utterance

class RoutingTests(unittest.TestCase):
    def test_other_never_reaches_terminal_whisper(self):
        model = Mock()
        verifier = Mock()
        verifier.matches.return_value = False
        self.assertEqual(transcribe_utterance(model, np.ones(16000, dtype=np.float32), verifier, "profile", .6), "")
        model.assert_not_called()
        self.assertEqual(verifier.matches.call_args.args[-1], .6)

    def test_quiet_target_reaches_whisper(self):
        model = Mock(return_value={"text": "متن"})
        verifier = Mock()
        verifier.matches.return_value = True
        audio = np.full(16000, .0001, dtype=np.float32)
        self.assertEqual(transcribe_utterance(model, audio, verifier, "profile"), "متن")
        model.assert_called_once()
        np.testing.assert_array_equal(model.call_args.args[0]["array"], audio)

    def test_frontend_other_never_reaches_whisper(self):
        model = Mock()
        with patch.object(app.speaker_filter, "matches", return_value=False), patch.object(app.app.state, "transcriber", model, create=True):
            self.assertEqual(app.transcribe(np.ones(16000, dtype=np.float32), 16000), "")
            model.assert_not_called()

    def test_frontend_quiet_target_reaches_whisper(self):
        model = Mock(return_value={"text": "متن"})
        with patch.object(app.speaker_filter, "matches", return_value=True), patch.object(app.app.state, "transcriber", model, create=True):
            self.assertEqual(app.transcribe(np.full(16000, .0001, dtype=np.float32), 16000), "متن")
            model.assert_called_once()

if __name__ == "__main__":
    unittest.main()
