import keyboard
import time
import numpy as np
import sounddevice as sd
import logging
import tempfile
import os
import soundfile as sf

import torch
import nemo.collections.asr as nemo_asr

logger = logging.getLogger(__name__)

logging.getLogger("nemo").setLevel(logging.ERROR)
logging.getLogger("nemo_logger").setLevel(logging.ERROR)
logging.getLogger("nemo_logging").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

nemo_logger = logging.getLogger("nemo_logging")
nemo_logger.handlers.clear()  # remove handlers NeMo attaches
nemo_logger.propagate = False  # stop bubbling to root
nemo_logger.disabled = True  # completely disable it

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1  # seconds, for silence detection
BUFFER_DURATION = 3.0  # max seconds to record per utterance
LANGUAGE = "en"  # Parakeet v2 is English; kept for interface consistency
SILENCE_THRESHOLD = 0.01  # adjust as needed
MIN_SILENCE_DURATION = 0.6  # seconds

PUSH_TO_TALK_KEY = "space"


MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"

_device = "cuda" if torch.cuda.is_available() else "cpu"
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
asr_model.eval()
asr_model.to(_device)


def _write_temp_wav(audio_f32: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """
    Write float32 mono audio to a temporary WAV file and return its path.
    NeMo's transcribe() API takes file paths.
    """
    # Ensure 1D float32
    audio_f32 = np.asarray(audio_f32, dtype=np.float32).flatten()

    # Avoid empty writes
    if audio_f32.size == 0:
        return ""

    fd, path = tempfile.mkstemp(suffix=".wav", prefix="jarvis_stt_", text=False)
    os.close(fd)  # soundfile will open it
    sf.write(path, audio_f32, sample_rate, subtype="PCM_16")
    return path


def record_push_to_talk():
    print(f"Hold '{PUSH_TO_TALK_KEY}' and start talking...")
    buffer = []
    recording = [False]  # mutable

    def callback(indata, frames, time_info, status):
        if keyboard.is_pressed(PUSH_TO_TALK_KEY):
            recording[0] = True
            audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            buffer.append(audio.copy())
        elif recording[0]:
            raise sd.CallbackStop()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        while not keyboard.is_pressed(PUSH_TO_TALK_KEY):
            sd.sleep(10)

        logger.info("SST Started capturing audio.")
        while True:
            if not keyboard.is_pressed(PUSH_TO_TALK_KEY) and recording[0]:
                break
            sd.sleep(10)
        logger.info("SST Finished capturing audio.")

    audio_data = np.concatenate(buffer) if buffer else np.array([], dtype=np.float32)
    return audio_data


def record_until_silence():
    buffer = []
    silence_chunks = 0
    max_chunks = int(BUFFER_DURATION / CHUNK_DURATION)
    min_silence_chunks = int(MIN_SILENCE_DURATION / CHUNK_DURATION)

    def callback(indata, frames, time_info, status):
        nonlocal silence_chunks
        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        buffer.append(audio.copy())
        if np.max(np.abs(audio)) < SILENCE_THRESHOLD:
            silence_chunks += 1
        else:
            silence_chunks = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        while True:
            sd.sleep(int(CHUNK_DURATION * 1000))
            if silence_chunks >= min_silence_chunks or len(buffer) >= max_chunks:
                break

    audio_data = np.concatenate(buffer) if buffer else np.array([], dtype=np.float32)
    return audio_data


def transcribe_user_audio(push_to_talk: bool) -> str:
    if push_to_talk:
        audio = record_push_to_talk()
    else:
        logger.info("SST Started capturing audio.")
        audio = record_until_silence()
        logger.info("SST Finished capturing audio.")

    if audio.size == 0:
        return ""

    # Optional: apply your VAD choice by controlling capture logic.
    # Parakeet here is run on the captured audio without extra VAD filtering.
    logger.info("SST Started transcribing.")

    tmp_path = _write_temp_wav(audio, SAMPLE_RATE)
    try:
        # NeMo transcribe returns a list of hypotheses; each has .text
        out = asr_model.transcribe([tmp_path], timestamps=False)
        # Depending on NeMo version, out[0] may be a Hypothesis with .text
        text = out[0].text if hasattr(out[0], "text") else str(out[0])
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    logger.info("SST Finished transcribing.")
    return text.strip()


if __name__ == "__main__":
    push_to_talk = True

    try:
        print(
            f"Sequential transcription started (Model: {MODEL_NAME})... Press Ctrl+C to stop."
        )
        while True:
            if push_to_talk:
                audio = record_push_to_talk()
            else:
                print(
                    time.strftime("%H-%M-%S-", time.localtime())
                    + f"{int((time.time() % 1) * 1000):03d}: SST Started capturing audio."
                )
                audio = record_until_silence()
                print(
                    time.strftime("%H-%M-%S-", time.localtime())
                    + f"{int((time.time() % 1) * 1000):03d}: SST Finished capturing audio."
                )

            if audio.size == 0:
                continue

            print(
                time.strftime("%H-%M-%S-", time.localtime())
                + f"{int((time.time() % 1) * 1000):03d}: SST Started transcribing."
            )
            tmp_path = _write_temp_wav(audio, SAMPLE_RATE)
            try:
                out = asr_model.transcribe([tmp_path], timestamps=False)
                text = out[0].text if hasattr(out[0], "text") else str(out[0])
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

            print(
                time.strftime("%H-%M-%S-", time.localtime())
                + f"{int((time.time() % 1) * 1000):03d}: SST Finished transcribing."
            )
            print(text.strip())

    except KeyboardInterrupt:
        print("Stopping STT...")
