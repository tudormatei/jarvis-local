import unittest
from unittest.mock import Mock, patch

from jarvis_tts.jarvis_tts import JarvisTTS


class JarvisTTSTests(unittest.TestCase):
    @patch("jarvis_tts.jarvis_tts.threading.Thread")
    @patch("jarvis_tts.jarvis_tts.TTSModel.load_model")
    def test_uses_public_italian_catalog_voice(self, load_model, thread):
        model = Mock(sample_rate=24000)
        load_model.return_value = model

        JarvisTTS()

        load_model.assert_called_once_with(language="italian")
        model.get_state_for_audio_prompt.assert_called_once_with("giovanni")
        thread.return_value.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
