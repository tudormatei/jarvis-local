import logging
import os
import tempfile
from dataclasses import dataclass
import keyboard
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JarvisSTTConfig:
    # Audio
    sample_rate: int = 16000
    chunk_duration: float = 0.1  # seconds (silence detection chunk size)
    buffer_duration: float = 3.0  # max seconds to record per utterance (silence mode)
    silence_threshold: float = 0.01  # amplitude threshold for silence
    min_silence_duration: float = (
        0.6  # seconds of silence to stop recording (silence mode)
    )

    # UX
    push_to_talk_key: str = "space"

    # Model
    model_name: str = "nvidia/parakeet-tdt-0.6b-v2"


class JarvisSTT:
    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        chunk_duration: float = 0.1,
        buffer_duration: float = 3.0,
        silence_threshold: float = 0.01,
        min_silence_duration: float = 0.6,
        push_to_talk_key: str = "space",
        model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
    ):
        self.config = JarvisSTTConfig(
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            buffer_duration=buffer_duration,
            silence_threshold=silence_threshold,
            min_silence_duration=min_silence_duration,
            push_to_talk_key=push_to_talk_key,
            model_name=model_name,
        )

        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = resolved_device

        logger.info(
            "Initializing JarvisSTT | model=%s | device=%s",
            self.config.model_name,
            self.device,
        )

        import nemo.collections.asr as nemo_asr

        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.config.model_name
        )
        self.asr_model.eval()
        self.asr_model.to(self.device)

    def _write_temp_wav(self, audio_f32: np.ndarray) -> str:
        audio_f32 = np.asarray(audio_f32, dtype=np.float32).flatten()
        if audio_f32.size == 0:
            return ""

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="jarvis_stt_", text=False)
        os.close(fd)
        sf.write(path, audio_f32, self.config.sample_rate, subtype="PCM_16")
        return path

    def record_push_to_talk(self) -> np.ndarray:
        logger.info("STT can start PUSH-TO-TALK.")
        print(f"Hold '{self.config.push_to_talk_key}' and start talking...")

        buffer = []
        recording = [False]  # mutable flag

        def callback(indata, frames, time_info, status):
            if keyboard.is_pressed(self.config.push_to_talk_key):
                recording[0] = True
                audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
                buffer.append(audio.copy())
            elif recording[0]:
                raise sd.CallbackStop()

        with sd.InputStream(
            samplerate=self.config.sample_rate,
            blocksize=int(self.config.sample_rate * self.config.chunk_duration),
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            while not keyboard.is_pressed(self.config.push_to_talk_key):
                sd.sleep(10)

            logger.info("STT Started capturing audio.")
            while True:
                if (
                    not keyboard.is_pressed(self.config.push_to_talk_key)
                    and recording[0]
                ):
                    break
                sd.sleep(10)
            logger.info("STT Finished capturing audio.")

        return np.concatenate(buffer) if buffer else np.array([], dtype=np.float32)

    def record_until_silence(self) -> np.ndarray:
        buffer = []
        silence_chunks = 0

        max_chunks = int(self.config.buffer_duration / self.config.chunk_duration)
        min_silence_chunks = int(
            self.config.min_silence_duration / self.config.chunk_duration
        )

        def callback(indata, frames, time_info, status):
            nonlocal silence_chunks
            audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            buffer.append(audio.copy())

            if np.max(np.abs(audio)) < self.config.silence_threshold:
                silence_chunks += 1
            else:
                silence_chunks = 0

        with sd.InputStream(
            samplerate=self.config.sample_rate,
            blocksize=int(self.config.sample_rate * self.config.chunk_duration),
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            while True:
                sd.sleep(int(self.config.chunk_duration * 1000))
                if silence_chunks >= min_silence_chunks or len(buffer) >= max_chunks:
                    break

        return np.concatenate(buffer) if buffer else np.array([], dtype=np.float32)

    def transcribe_user_audio(self, *, push_to_talk: bool) -> str:
        if push_to_talk:
            audio = self.record_push_to_talk()
        else:
            logger.info("STT Started capturing audio.")
            audio = self.record_until_silence()
            logger.info("STT Finished capturing audio.")

        if audio.size == 0:
            return ""

        logger.info("STT Started transcribing.")
        tmp_path = self._write_temp_wav(audio)
        try:
            out = self.asr_model.transcribe([tmp_path], timestamps=False, verbose=False)
            text = out[0].text if hasattr(out[0], "text") else str(out[0])
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        logger.info("STT Finished transcribing.")
        return text.strip()


if __name__ == "__main__":
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    push_to_talk = True  # set False to use silence-stop mode

    stt = JarvisSTT(
        # override defaults here if needed:
        # device=None,  # auto cuda/cpu
        # device="cpu",
        # push_to_talk_key="space",
        # silence_threshold=0.01,
        # min_silence_duration=0.6,
    )

    try:
        print(
            f"Sequential transcription started (Model: {stt.config.model_name})... Press Ctrl+C to stop."
        )

        while True:
            record_start = time.time()

            if push_to_talk:
                audio = stt.record_push_to_talk()
            else:
                print(
                    time.strftime("%H-%M-%S-", time.localtime())
                    + f"{int((time.time() % 1) * 1000):03d}: STT Started capturing audio."
                )
                audio = stt.record_until_silence()
                print(
                    time.strftime("%H-%M-%S-", time.localtime())
                    + f"{int((time.time() % 1) * 1000):03d}: STT Finished capturing audio."
                )

            record_end = time.time()

            if audio.size == 0:
                continue

            record_duration_ms = (record_end - record_start) * 1000
            print(f"Total recording time: {record_duration_ms:.3f} ms")

            transcribe_start = time.time()
            print(
                time.strftime("%H-%M-%S-", time.localtime())
                + f"{int((time.time() % 1) * 1000):03d}: STT Started transcribing."
            )

            text = stt.transcribe_user_audio(push_to_talk=push_to_talk)

            transcribe_end = time.time()
            print(
                time.strftime("%H-%M-%S-", time.localtime())
                + f"{int((time.time() % 1) * 1000):03d}: STT Finished transcribing."
            )

            transcribe_duration_ms = (transcribe_end - transcribe_start) * 1000
            print(f"Total transcription time: {transcribe_duration_ms:.3f} ms")

            print(text.strip())

    except KeyboardInterrupt:
        print("Stopping STT...")
