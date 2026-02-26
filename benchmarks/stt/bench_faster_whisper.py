import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


def load_audio_16k_mono(path: str, target_sr: int = 16000) -> tuple[np.ndarray, float]:
    """
    Loads audio via soundfile. Assumes file is already 16kHz mono wav for best benchmarking accuracy.
    Returns float32 mono samples in [-1, 1] and duration seconds.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]  # take first channel
    if sr != target_sr:
        raise ValueError(
            f"Expected {target_sr} Hz audio for clean benchmarking, got {sr}. "
            f"Please resample to {target_sr} Hz first (e.g., with ffmpeg)."
        )
    dur_s = float(len(audio)) / float(sr)
    return audio, dur_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=str, default="sample.wav")
    ap.add_argument("--model", type=str, default="small.en")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--compute_type",
        type=str,
        default="int8",
        help="Typical: int8 (fast), int8_float16, float16, float32",
    )
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--beam_size", type=int, default=1)
    ap.add_argument("--vad", action="store_true", help="Enable built-in VAD filter")
    args = ap.parse_args()

    audio_path = str(Path(args.audio).expanduser().resolve())
    audio, dur_s = load_audio_16k_mono(audio_path)

    print("== Config ==")
    print("model:", args.model)
    print("device:", args.device)
    print("compute_type:", args.compute_type)
    print("beam_size:", args.beam_size)
    print("vad_filter:", bool(args.vad))
    print("audio:", audio_path)
    print("audio seconds:", f"{dur_s:.2f}")

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        download_root=args.download_root,
    )

    def run_once():
        t0 = time.perf_counter()
        segments, info = model.transcribe(
            audio,
            language="en",
            beam_size=args.beam_size,
            vad_filter=bool(args.vad),
            vad_parameters=(
                {
                    "min_silence_duration_ms": 200,
                    "threshold": 0.5,
                }
                if args.vad
                else None
            ),
        )
        text = "".join([s.text for s in segments]).strip()
        t1 = time.perf_counter()
        return (t1 - t0), text, info

    print("\n== Warmup ==")
    for i in range(args.warmup):
        dt, _, _ = run_once()
        print(f"warmup {i+1}: {dt:.4f}s")

    print("\n== Timed runs ==")
    times = []
    last_text = ""
    for i in range(args.runs):
        dt, text, _ = run_once()
        times.append(dt)
        last_text = text
        rtf = dt / dur_s
        rtfx = dur_s / dt
        print(
            f"run {i+1}: {dt:.4f}s | audio {dur_s:.2f}s | RTF {rtf:.4f} | RTFx {rtfx:.1f}"
        )

    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    rtf_m = median / dur_s
    rtfx_m = dur_s / median

    print("\n== Summary (median) ==")
    print(f"median time: {median:.4f}s")
    print(f"RTF:  {rtf_m:.4f}")
    print(f"RTFx: {rtfx_m:.1f}")

    print("\n== Transcript (last run) ==")
    print(last_text)


if __name__ == "__main__":
    main()
