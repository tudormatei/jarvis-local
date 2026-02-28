import time
import sounddevice as sd
import numpy as np
import threading
import asyncio
import websockets
import queue
import logging
from pocket_tts import TTSModel
import json


logger = logging.getLogger(__name__)


class JarvisTTS:
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
        self.is_playing = False
        self._stop_sender = threading.Event()

        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        self.normalize = normalize

        self.websocket_clients = set()
        self.websocket_loop = None

        logger.info("TTS Loading Pocket TTS model...")
        self.model = TTSModel.load_model()
        self.SAMPLE_RATE = int(self.model.sample_rate)
        self.block_duration = self.BLOCK_SIZE / self.SAMPLE_RATE

        logger.info(
            f"TTS model ready. sample_rate={self.SAMPLE_RATE} Hz, "
            f"block_size={self.BLOCK_SIZE} samples, "
            f"block_duration={self.block_duration * 1000:.2f} ms  "
            f"({self.SAMPLE_RATE // self.BLOCK_SIZE} blocks/sec)"
        )

        logger.info("TTS Computing voice embeddings")
        self.voice_state = self.model.get_state_for_audio_prompt(speaker_sample)
        logger.info("TTS Finished voice state.")

        threading.Thread(target=self._audio_player_thread, daemon=True).start()
        if self.ui_enabled:
            threading.Thread(target=self._websocket_sender_thread, daemon=True).start()

    def speak(self, text: str):
        logger.info("TTS received LLM response.")
        chunks = self.model.generate_audio_stream(self.voice_state, text)

        first_chunk_seen = False
        leftover = np.zeros((0,), dtype=np.float32)
        peak = 1.0

        for chunk in chunks:
            if not first_chunk_seen:
                logger.info("TTS first audio chunk received.")
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

    def stop(self):
        self._stop_sender.set()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Exception:
                break

    def _audio_player_thread(self):
        if self.ui_enabled and self.websocket_loop is None:
            self._start_websocket_server()

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
                    self.is_playing = False
                    self.latest_block = None
                    logger.info("TTS playback finished.")
                    continue

                self.latest_block = block.astype(np.float32)
                self.is_playing = True
                stream.write(block)  # blocks for ~block_duration seconds

    def _websocket_sender_thread(self):
        target_interval = self.block_duration
        next_send = time.perf_counter()

        while not self._stop_sender.is_set():
            now = time.perf_counter()

            if now >= next_send:
                block = self.latest_block
                if block is not None:
                    payload = block.tobytes()
                else:
                    payload = np.zeros(self.BLOCK_SIZE, dtype=np.float32).tobytes()

                if self.websocket_loop is not None and self.websocket_clients:
                    for ws in list(self.websocket_clients):
                        asyncio.run_coroutine_threadsafe(
                            self._safe_ws_send(ws, payload),
                            self.websocket_loop,
                        )

                next_send += target_interval

            sleep_for = max(0.0, next_send - time.perf_counter() - 0.001)
            time.sleep(sleep_for)

    async def _safe_ws_send(self, ws, payload: bytes):
        try:
            await ws.send(payload)
        except Exception as e:
            logger.debug(f"WebSocket send error: {e}")

    def _start_websocket_server(self):
        async def handler(websocket):
            logger.info("TTS WS client connected")
            self.websocket_clients.add(websocket)
            try:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "config",
                            "sample_rate": self.SAMPLE_RATE,
                            "block_size": self.BLOCK_SIZE,
                        }
                    )
                )
                await websocket.wait_closed()
            except Exception:
                pass
            finally:
                self.websocket_clients.discard(websocket)
                logger.info("TTS WS client disconnected.")

        async def run_server():
            server = await websockets.serve(
                handler, self.websocket_host, self.websocket_port
            )
            logger.info(
                f"TTS WS server on ws://{self.websocket_host}:{self.websocket_port}"
            )
            await server.wait_closed()

        self.websocket_loop = asyncio.new_event_loop()
        threading.Thread(
            target=self.websocket_loop.run_until_complete,
            args=(run_server(),),
            daemon=True,
        ).start()


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
