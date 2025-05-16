import json
import time
import torch
import sounddevice as sd
import numpy as np
import threading
import asyncio
import websockets
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
    def __init__(self, model_path="./models/jarvis_v2/", speaker_sample="./models/jarvis_v2/reference.wav", should_stream=False):
        self.SAMPLE_RATE = 24000
        self.BLOCK_SIZE = 256
        self.audio_queue = queue.Queue()
        self.should_stream = should_stream

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
                    block = np.pad(
                        block, (0, self.BLOCK_SIZE - len(block)), 'constant')
                self.audio_queue.put(block.astype(np.float32))

        # Signal end of speech
        self.audio_queue.put(None)

    def start_websocket_server(self):
        async def handler(websocket):
            self.websocket_clients.add(websocket)
            try:
                while True:
                    await asyncio.sleep(1)  # Keep connection alive
            except Exception:
                pass
            finally:
                self.websocket_clients.remove(websocket)

        async def run_server():
            self.websocket_clients = set()
            self.websocket_server = await websockets.serve(handler, "localhost", 8765)
            await self.websocket_server.wait_closed()

        self.websocket_loop = asyncio.new_event_loop()
        threading.Thread(target=self.websocket_loop.run_until_complete, args=(
            run_server(),), daemon=True).start()

    def _audio_player_thread(self):
        started_playing = False
        if self.should_stream:
            if not hasattr(self, "websocket_clients"):
                self.start_websocket_server()

        with sd.OutputStream(samplerate=self.SAMPLE_RATE, channels=1, blocksize=self.BLOCK_SIZE) as stream:
            while True:
                block = self.audio_queue.get()
                if block is None:
                    started_playing = False
                    logger.info("TTS Playback finished.")
                else:
                    stream.write(block)
                    if self.should_stream:
                        for ws in list(getattr(self, "websocket_clients", [])):
                            asyncio.run_coroutine_threadsafe(
                                ws.send(block.astype(np.float32).tobytes()
                                        ), self.websocket_loop
                            )
                    if not started_playing:
                        logger.info("TTS Playback started.")
                        started_playing = True

    def stop(self):
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()  # Drain remaining audio


if __name__ == "__main__":
    jarvis = JarvisTTS(should_stream=True)
    print("Finished setup")
    for _ in range(10):
        print("\033[91mYourWordHere\033[0m")

    time.sleep(5)
    jarvis.speak("Sir, the Tesseract is showing signs of activity.")
    jarvis.speak("I recommend we inform Director Fury immediately.")
    print("Sleeping...")
    time.sleep(10)
    jarvis.stop()
