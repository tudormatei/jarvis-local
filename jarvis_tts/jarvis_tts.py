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
    [XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]
)

logger = logging.getLogger(__name__)


class JarvisTTS:
    def __init__(
        self,
        model_path="./models/jarvis_v2/",
        speaker_sample="./models/jarvis_v2/reference.wav",
        ui_enabled=False,
    ):
        self.SAMPLE_RATE = 24000
        self.BLOCK_SIZE = 256
        self.audio_queue = queue.Queue()
        self.ui_enabled = ui_enabled
        self.latest_block = None
        self._stop_sender = threading.Event()

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_capability = torch.cuda.get_device_capability(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            logger.info(
                f"Using CUDA: True | GPU: {gpu_name} | "
                f"Compute Capability: {gpu_capability} | "
                f"VRAM: {total_mem:.2f} GB"
            )
        else:
            logger.info("Using CUDA: False")
        logger.info("Loading model...")

        self.config = XttsConfig()
        self.config.load_json(f"{model_path}/config.json")
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(
            self.config, checkpoint_dir=model_path, use_deepspeed=False
        )
        self.model.cuda()
        self.model = torch.compile(self.model)

        logger.info("Computing speaker embedding...")
        self.gpt_cond_latent, self.speaker_embedding = (
            self.model.get_conditioning_latents(audio_path=[speaker_sample])
        )
        logger.info("Finished speaker embeddings.")

        threading.Thread(target=self._audio_player_thread, daemon=True).start()
        if self.ui_enabled:
            threading.Thread(target=self._websocket_sender_thread, daemon=True).start()

    def speak(self, text):
        logger.info("TTS First chunk received.")
        chunks = self.model.inference_stream(
            text, "en", self.gpt_cond_latent, self.speaker_embedding
        )

        first_chunk = False
        for chunk in chunks:
            if not first_chunk:
                logger.info("TTS Finished inference.")
                first_chunk = True

            audio = chunk.cpu().numpy().flatten()
            audio /= np.max(np.abs(audio), initial=1.0)

            for start in range(0, len(audio), self.BLOCK_SIZE):
                block = audio[start : start + self.BLOCK_SIZE]
                if len(block) < self.BLOCK_SIZE:
                    block = np.pad(block, (0, self.BLOCK_SIZE - len(block)), "constant")
                self.audio_queue.put(block.astype(np.float32))

        self.audio_queue.put(None)

    def start_websocket_server(self):
        async def handler(websocket):
            self.websocket_clients.add(websocket)
            try:
                while True:
                    await asyncio.sleep(1)
            except Exception:
                pass
            finally:
                self.websocket_clients.remove(websocket)

        async def run_server():
            self.websocket_clients = set()
            self.websocket_server = await websockets.serve(handler, "localhost", 8765)
            await self.websocket_server.wait_closed()

        self.websocket_loop = asyncio.new_event_loop()
        threading.Thread(
            target=self.websocket_loop.run_until_complete,
            args=(run_server(),),
            daemon=True,
        ).start()

    def _audio_player_thread(self):
        started_playing = False
        if self.ui_enabled:
            if not hasattr(self, "websocket_clients"):
                self.start_websocket_server()

        with sd.OutputStream(
            samplerate=self.SAMPLE_RATE, channels=1, blocksize=self.BLOCK_SIZE
        ) as stream:
            while True:
                block = self.audio_queue.get()
                if block is None:
                    started_playing = False
                    logger.info("TTS Playback finished.")
                    if self.ui_enabled:
                        self.latest_block = None
                        for ws in list(getattr(self, "websocket_clients", [])):
                            asyncio.run_coroutine_threadsafe(
                                ws.send(
                                    np.zeros(
                                        self.BLOCK_SIZE, dtype=np.float32
                                    ).tobytes()
                                ),
                                self.websocket_loop,
                            )
                else:
                    if self.ui_enabled:
                        self.latest_block = block.astype(np.float32)
                    stream.write(block)
                    if not started_playing:
                        logger.info("TTS Playback started.")
                        started_playing = True

    def _websocket_sender_thread(self):
        while not self._stop_sender.is_set():
            if (
                self.ui_enabled
                and hasattr(self, "websocket_clients")
                and self.latest_block is not None
            ):
                for ws in list(getattr(self, "websocket_clients", [])):
                    try:
                        asyncio.run_coroutine_threadsafe(
                            ws.send(self.latest_block.tobytes()), self.websocket_loop
                        )
                    except Exception as e:
                        logger.info(f"WebSocket send error: {e}")
            time.sleep(1 / 60)

    def stop(self):
        self._stop_sender.set()
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()


if __name__ == "__main__":
    jarvis = JarvisTTS(ui_enabled=True)
    jarvis.speak("Sir, the Tesseract is showing signs of activity.")
    jarvis.speak("I recommend we inform Director Fury immediately.")
    print("Sleeping...")
    time.sleep(10)
    jarvis.stop()
