import time
import torch
import sounddevice as sd
import numpy as np
import threading
import queue
import logging
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs

torch.serialization.add_safe_globals(
    [XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

logger = logging.getLogger(__name__)


class JarvisTTS:
    def __init__(self, model_path="jarvis_tts/models/jarvis_v2/", speaker_sample="jarvis_tts/models/jarvis_v2/reference.wav"):
        self.SAMPLE_RATE = 24000
        self.BLOCK_SIZE = 256
        self.audio_queue = queue.Queue()

        logger.info(f"Using CUDA: {torch.cuda.is_available()}")
        logger.info("Loading model...")

        self.config = XttsConfig()
        self.config.load_json(f"{model_path}/config.json")
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(
            self.config, checkpoint_dir=model_path, use_deepspeed=False)
        self.model.cuda()
        self.model = torch.compile(self.model)

        logger.info("Computing speaker embedding...")
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[speaker_sample])
        logger.info("Finished speaker embeddings.")

        threading.Thread(target=self._audio_player_thread, daemon=True).start()

    def speak(self, text):
        logger.info("TTS First chunk received.")
        chunks = self.model.inference_stream(
            text, "en", self.gpt_cond_latent, self.speaker_embedding)

        first_chunk = False
        for chunk in chunks:
            if not first_chunk:
                logger.info("TTS Finished inference.")
                first_chunk = True

            audio = chunk.cpu().numpy().flatten()
            audio /= np.max(np.abs(audio), initial=1.0)  # Normalize

            for start in range(0, len(audio), self.BLOCK_SIZE):
                block = audio[start:start + self.BLOCK_SIZE]
                if len(block) < self.BLOCK_SIZE:
                    fade_out = np.linspace(1, 0, self.BLOCK_SIZE - len(block))
                    block = np.concatenate([block, block[-1] * fade_out])
                self.audio_queue.put(block.astype(np.float32).reshape(-1, 1))

        # Signal end of speech
        self.audio_queue.put(None)

    def _audio_player_thread(self):
        with sd.OutputStream(samplerate=self.SAMPLE_RATE, channels=1, blocksize=self.BLOCK_SIZE) as stream:
            started_playing = False
            while True:
                block = self.audio_queue.get()
                if block is None:
                    started_playing = False
                    logger.info("TTS Playback finished.")
                else:
                    stream.write(block)
                    if not started_playing:
                        logger.info("TTS Playback started.")
                        started_playing = True

    def stop(self):
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()  # Drain remaining audio


if __name__ == "__main__":
    jarvis = JarvisTTS()

    jarvis.speak("Sir, the Tesseract is showing signs of activity.")
    jarvis.speak("I recommend we inform Director Fury immediately.")
    print("Sleeping...")
    time.sleep(10)
    jarvis.stop()
