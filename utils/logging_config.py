import logging
import time


# ---------------------------
# Formatter: adds (+Xms)
# ---------------------------
class DeltaFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self._last_time = None

    def format(self, record: logging.LogRecord) -> str:
        now = time.perf_counter()
        delta = 0.0 if self._last_time is None else (now - self._last_time) * 1000.0
        self._last_time = now
        record.delta_ms = f"+{int(delta)}ms"
        return super().format(record)


# ---------------------------
# Filters: drop specific spam
# ---------------------------
class DropNoise(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()

        # Pocket-TTS (or similar) sometimes logs via root with this exact line:
        if record.name == "root" and "TTS Model loaded successfully" in msg:
            return False

        # websockets lifecycle chatter
        if record.name.startswith("websockets.") and (
            "server listening" in msg
            or "connection open" in msg
            or "connection closed" in msg
        ):
            return False

        return True


class DropLhotseSpam(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Initializing Lhotse CutSet from a single NeMo manifest" in msg:
            return False
        if "The following configuration keys are ignored by Lhotse dataloader" in msg:
            return False
        if "requested tokenization during data sampling" in msg:
            return False
        return True


def _coerce_level(level) -> int:
    """
    Accepts: Enum (with .name), string ("INFO"), int (logging.INFO), etc.
    """
    if hasattr(level, "name"):  # Enum
        level = level.name
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def setup_logging(level="INFO") -> None:
    level_int = _coerce_level(level)

    handler = logging.StreamHandler()
    handler.setFormatter(
        DeltaFormatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s (%(delta_ms)s)"
        )
    )

    # Attach filters
    handler.addFilter(DropNoise())
    handler.addFilter(DropLhotseSpam())

    # Root handles everything by default
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level_int)
    root.addHandler(handler)

    # Silence noisy third-party trees (keep warnings/errors)
    noisy_to_warning = [
        "websockets",
        "websockets.server",
        "huggingface_hub",
        "transformers",
        "urllib3",
        "httpx",
        "pocket_tts",
    ]
    for name in noisy_to_warning:
        logging.getLogger(name).setLevel(logging.WARNING)

    # NeMo / Lightning / Lhotse tend to be insane: keep only errors
    nemo_to_error = [
        "nemo",
        "nemo.utils",
        "lightning",
        "pytorch_lightning",
        "lhotse",
        "nemo.collections.common.data.lhotse",
        "nemo.collections.common.data.lhotse.dataloader",
        "nemo.collections.asr.data.audio_to_text_lhotse",
    ]
    for name in nemo_to_error:
        logging.getLogger(name).setLevel(logging.ERROR)

    # NeMo has its own verbosity wrapper; enforce it too
    try:
        from nemo.utils import logging as nemo_logging

        nemo_logging.set_verbosity(logging.ERROR)
    except Exception:
        pass
