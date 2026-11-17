import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from jarvis_stt.jarvis_stt import JarvisSTT


class JarvisSTTTests(unittest.TestCase):
    @patch("jarvis_stt.jarvis_stt.WhisperModel")
    def test_initializes_whisper_for_cpu(self, whisper_model):
        stt = JarvisSTT()

        self.assertEqual(stt.device, "cpu")
        whisper_model.assert_called_once_with(
            "small",
            device="cpu",
            compute_type="int8",
        )

    def test_transcribes_recorded_audio(self):
        stt = JarvisSTT.__new__(JarvisSTT)
        stt._stop_event = threading.Event()
        stt.asr_model = Mock()
        stt.asr_model.transcribe.return_value = (
            iter([SimpleNamespace(text=" Ciao"), SimpleNamespace(text=" Jarvis")]),
            None,
        )
        audio = np.array([0.1, 0.2], dtype=np.float32)

        with patch.object(stt, "record_push_to_talk", return_value=audio):
            transcript = stt.transcribe_user_audio(push_to_talk=True)

        self.assertEqual(transcript, "Ciao Jarvis")
        stt.asr_model.transcribe.assert_called_once_with(audio, beam_size=1)


if __name__ == "__main__":
    unittest.main()
