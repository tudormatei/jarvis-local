import keyboard
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import logging

logger = logging.getLogger(__name__)

# Configuration
MODEL_SIZE = "base.en"
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1  # seconds, for silence detection
BUFFER_DURATION = 3.0  # max seconds to record per utterance
LANGUAGE = 'en'
USE_VAD = True
SILENCE_THRESHOLD = 0.01  # adjust as needed
MIN_SILENCE_DURATION = 0.6  # seconds

# Load Whisper model
model = WhisperModel(
    MODEL_SIZE,
    device="cuda",
    compute_type="int8",
    download_root="./jarvis_stt/models"
)

PUSH_TO_TALK_KEY = 'space'


def record_push_to_talk():
    print(f"Hold '{PUSH_TO_TALK_KEY}' and start talking...")
    buffer = []
    # Use a mutable object to allow modification in callback
    recording = [False]

    def callback(indata, frames, time_info, status):
        if keyboard.is_pressed(PUSH_TO_TALK_KEY):
            recording[0] = True
            audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            buffer.append(audio.copy())
        elif recording[0]:
            # If we were recording and the key is released, stop the stream
            raise sd.CallbackStop()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
        channels=1,
        dtype="float32",
        callback=callback
    ):
        # Wait for key press
        while not keyboard.is_pressed(PUSH_TO_TALK_KEY):
            sd.sleep(10)
        logger.info("SST Started capturing audio.")
        # Wait for callback to raise CallbackStop (when key is released)
        while True:
            if not keyboard.is_pressed(PUSH_TO_TALK_KEY) and recording[0]:
                break
            sd.sleep(10)
        logger.info("SST Finished capturing audio.")

    audio_data = np.concatenate(buffer) if buffer else np.array([])
    return audio_data


def record_until_silence():
    buffer = []
    silence_chunks = 0
    max_chunks = int(BUFFER_DURATION / CHUNK_DURATION)
    min_silence_chunks = int(MIN_SILENCE_DURATION / CHUNK_DURATION)

    def callback(indata, frames, time_info, status):
        nonlocal buffer, silence_chunks
        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        buffer.append(audio.copy())
        # Silence detection
        if np.max(np.abs(audio)) < SILENCE_THRESHOLD:
            silence_chunks += 1
        else:
            silence_chunks = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
        channels=1,
        dtype="float32",
        callback=callback
    ):
        while True:
            sd.sleep(int(CHUNK_DURATION * 1000))
            if silence_chunks >= min_silence_chunks or len(buffer) >= max_chunks:
                break

    audio_data = np.concatenate(buffer)
    return audio_data


def transcribe_user_audio(push_to_talk):
    if push_to_talk:
        audio = record_push_to_talk()
    else:
        logger.info("SST Started capturing audio.")
        audio = record_until_silence()
        logger.info("SST Finished capturing audio.")
    logger.info("SST Started transcribing.")
    segments, _ = model.transcribe(
        audio,
        language=LANGUAGE,
        vad_filter=(not push_to_talk and USE_VAD),
        vad_parameters={
            "min_silence_duration_ms": 200,
            "threshold": 0.5,
        }
    )
    logger.info("SST Finished transcribing.")
    res = ""
    for s in segments:
        res += s.text.strip()

    return res


if __name__ == "__main__":
    push_to_talk = True

    try:
        print(
            f"Sequential transcription started (Model: {MODEL_SIZE})... Press Ctrl+C to stop.")
        while True:
            if push_to_talk:
                audio = record_push_to_talk()
            else:
                print(
                    time.strftime("%H-%M-%S-", time.localtime()) + f"{int((time.time() % 1) * 1000):03d}: SST Started capturing audio.")
                audio = record_until_silence()
                print(
                    time.strftime("%H-%M-%S-", time.localtime()) + f"{int((time.time() % 1) * 1000):03d}: SST Finished capturing audio.")
            if len(audio) == 0:
                continue
            print(
                time.strftime("%H-%M-%S-", time.localtime()) + f"{int((time.time() % 1) * 1000):03d}: SST Started transcribing.")
            segments, _ = model.transcribe(
                audio,
                language=LANGUAGE,
                vad_filter=USE_VAD,
                vad_parameters={
                    "min_silence_duration_ms": 200,
                    "threshold": 0.5,
                }
            )
            print(
                time.strftime("%H-%M-%S-", time.localtime()) + f"{int((time.time() % 1) * 1000):03d}: SST Finished transcribing.")
            for s in segments:
                print(s.text.strip())

    except KeyboardInterrupt:
        print("Stopping STT...")
