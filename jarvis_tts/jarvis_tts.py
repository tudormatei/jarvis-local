import time
import sounddevice as sd
import numpy as np
import threading
import asyncio
import websockets
import queue
import logging

for name in [
    "pocket_tts",
    "pocket_tts.models.tts_model",
    "pocket_tts.utils.utils",
    "pocket_tts.conditioners.text",
    "huggingface_hub",
    "transformers",
    "httpx",  # optional: hides the "HTTP Request: POST ..." line
    "urllib3",
]:
    logging.getLogger(name).setLevel(logging.WARNING)

from pocket_tts import TTSModel

logger = logging.getLogger(__name__)


class JarvisTTS:
    """
    Pocket TTS drop-in replacement for your XTTSv2 module:
      - Loads model once
      - Extracts voice_state once from reference.wav (or .safetensors)
      - speak(text) streams audio, chunks into BLOCK_SIZE frames, pushes into audio_queue
      - Dedicated audio playback thread (sounddevice OutputStream)
      - Optional WebSocket UI streaming at ~60fps of the latest audio block
      - Uses None sentinel to mark end-of-utterance (same behavior as your old code)

    Requirements:
      pip install pocket-tts sounddevice numpy websockets

    Notes:
      - Voice cloning weights require accepting HF terms for kyutai/pocket-tts and logging in.
      - reference.wav must be PCM WAV (not IEEE float). If you see: wave.Error unknown format: 3
        convert:
          ffmpeg -y -i reference.wav -ac 1 -ar 24000 -c:a pcm_s16le reference_pcm.wav
    """

    def __init__(
        self,
        speaker_sample="./reference.wav",
        ui_enabled=False,
        sample_rate=24000,
        block_size=256,
        websocket_host="localhost",
        websocket_port=8765,
        normalize=True,
    ):
        self.SAMPLE_RATE = int(sample_rate)
        self.BLOCK_SIZE = int(block_size)
        self.audio_queue = queue.Queue()
        self.ui_enabled = ui_enabled
        self.latest_block = None
        self._stop_sender = threading.Event()

        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        self.normalize = normalize

        logger.info("TTS Loading Pocket TTS model...")
        self.model = TTSModel.load_model()

        self.SAMPLE_RATE = int(self.model.sample_rate)

        logger.info("TTS Computing voice embeddings")
        self.voice_state = self.model.get_state_for_audio_prompt(speaker_sample)
        logger.info("TTS Finished voice state.")

        threading.Thread(target=self._audio_player_thread, daemon=True).start()
        if self.ui_enabled:
            threading.Thread(target=self._websocket_sender_thread, daemon=True).start()

    def speak(self, text: str):
        logger.info("TTS submit text.")
        chunks = self.model.generate_audio_stream(self.voice_state, text)

        first_chunk_seen = False
        leftover = np.zeros((0,), dtype=np.float32)

        peak = 1.0

        for chunk in chunks:
            if not first_chunk_seen:
                logger.info("TTS first chunk received.")
                first_chunk_seen = True

            audio = (
                chunk.detach().cpu().numpy().astype(np.float32, copy=False).flatten()
            )
            if audio.size == 0:
                continue

            if self.normalize:
                peak = max(peak, float(np.max(np.abs(audio))) if audio.size else 1.0)
                if peak > 0:
                    audio = audio / peak

            if leftover.size:
                audio = np.concatenate([leftover, audio], axis=0)
                leftover = np.zeros((0,), dtype=np.float32)

            n_full = (audio.size // self.BLOCK_SIZE) * self.BLOCK_SIZE
            full = audio[:n_full]
            leftover = audio[n_full:]

            for start in range(0, full.size, self.BLOCK_SIZE):
                block = full[start : start + self.BLOCK_SIZE]
                self.audio_queue.put(block)

        if leftover.size:
            block = np.pad(
                leftover, (0, self.BLOCK_SIZE - leftover.size), mode="constant"
            )
            self.audio_queue.put(block.astype(np.float32))

        self.audio_queue.put(None)
        logger.info("TTS finished inference (all chunks queued).")

    def start_websocket_server(self):
        async def handler(websocket):
            self.websocket_clients.add(websocket)
            try:
                while True:
                    await asyncio.sleep(1)
            except Exception:
                pass
            finally:
                self.websocket_clients.discard(websocket)

        async def run_server():
            self.websocket_clients = set()
            self.websocket_server = await websockets.serve(
                handler, self.websocket_host, self.websocket_port
            )
            await self.websocket_server.wait_closed()

        self.websocket_loop = asyncio.new_event_loop()
        threading.Thread(
            target=self.websocket_loop.run_until_complete,
            args=(run_server(),),
            daemon=True,
        ).start()

    def _audio_player_thread(self):
        started_playing = False

        if self.ui_enabled and not hasattr(self, "websocket_clients"):
            self.start_websocket_server()

        with sd.OutputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            blocksize=self.BLOCK_SIZE,
            dtype="float32",
            latency="low",
        ) as stream:
            while True:
                block = self.audio_queue.get()

                if block is None:
                    started_playing = False
                    logger.info("TTS playback finished.")
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
                    continue

                if self.ui_enabled:
                    self.latest_block = block.astype(np.float32)

                stream.write(block)

                if not started_playing:
                    logger.info("TTS playback started.")
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
                            ws.send(self.latest_block.tobytes()),
                            self.websocket_loop,
                        )
                    except Exception as e:
                        logger.info(f"WebSocket send error: {e}")
            time.sleep(1 / 60)

    def stop(self):
        self._stop_sender.set()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Exception:
                break


if __name__ == "__main__":
    jarvis = JarvisTTS(
        speaker_sample="./reference.wav",
        ui_enabled=True,
        block_size=256,
        normalize=True,
    )

    jarvis.speak("Sir, the Tesseract is showing signs of activity.")
    jarvis.speak("I recommend we inform Director Fury immediately.")

    print("Sleeping...")
    time.sleep(10)
    jarvis.stop()
