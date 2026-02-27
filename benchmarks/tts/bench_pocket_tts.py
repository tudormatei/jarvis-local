#!/usr/bin/env python3
"""
Pocket TTS voice-clone + streaming benchmark (Windows-friendly)

What it tests (per utterance):
  - submit→first_chunk: time from "we hand text to the generator" until PocketTTS yields audio
  - submit→first_audible (est): adds (1) first non-silence sample position + (2) output device latency estimate

It also demonstrates that you DO NOT need to re-extract the voice state for each text:
  - model is loaded once
  - voice_state is extracted once
  - multiple texts are generated back-to-back in the same audio OutputStream

Install (inside conda env):
  pip install pocket-tts sounddevice numpy

Run examples:
  python bench_pocket_tts_multi.py --ref reference_pcm.wav --text "Hello there" --text "Second sentence"
  python bench_pocket_tts_multi.py --ref reference_voice.safetensors --text "One" --text "Two"

Notes:
  - Your reference WAV must be PCM (not float). If you get wave.Error: unknown format: 3,
    convert with:
      ffmpeg -y -i reference.wav -ac 1 -ar 24000 -c:a pcm_s16le reference_pcm.wav
"""

import argparse
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd
from pocket_tts import TTSModel


def _device_latency_seconds() -> float:
    """Best-effort output latency estimate from PortAudio/device defaults."""
    try:
        dev_info = sd.query_devices(kind="output")
        return float(
            dev_info.get("default_low_output_latency", None)
            or dev_info.get("default_output_latency", 0.0)
            or 0.0
        )
    except Exception:
        return 0.0


def stream_one_utterance(
    out: sd.OutputStream,
    model: TTSModel,
    voice_state,
    text: str,
    out_latency_s: float,
    audible_threshold: float,
) -> Tuple[Optional[float], Optional[float], float, float]:
    """
    Returns:
      t_first_chunk_s, t_first_audible_est_s, audio_duration_s, wall_time_s
    """
    sr = int(model.sample_rate)
    t_submit = time.perf_counter()

    t_first_chunk_s: Optional[float] = None
    t_first_audible_est_s: Optional[float] = None
    saw_audible = False

    total_samples = 0

    # Exact "text submitted" moment is the creation/use of this iterator:
    stream_iter = model.generate_audio_stream(voice_state, text)

    for chunk in stream_iter:
        audio = chunk.detach().cpu().numpy().astype(np.float32, copy=False)
        if audio.size == 0:
            continue

        now = time.perf_counter()
        if t_first_chunk_s is None:
            t_first_chunk_s = now - t_submit

        # First *audible* sample detection (avoid counting initial silence)
        if not saw_audible:
            abs_audio = np.abs(audio)
            peak = float(abs_audio.max())
            if peak >= audible_threshold:
                idx = int(np.argmax(abs_audio >= audible_threshold))
                t_when_written_s = time.perf_counter() - t_submit
                t_first_audible_est_s = t_when_written_s + (idx / sr) + out_latency_s
                saw_audible = True

        # Play
        out.write(audio.reshape(-1, 1))
        total_samples += audio.size

    audio_duration_s = total_samples / sr
    wall_time_s = time.perf_counter() - t_submit
    return t_first_chunk_s, t_first_audible_est_s, audio_duration_s, wall_time_s


def main() -> int:
    p = argparse.ArgumentParser(
        description="Benchmark Pocket TTS voice cloning + streaming over multiple texts without re-extracting voice_state."
    )
    p.add_argument(
        "--ref",
        required=True,
        help="reference PCM wav OR a previously exported .safetensors voice state",
    )
    p.add_argument(
        "--text",
        action="append",
        required=True,
        help="Text to speak. Pass multiple times to test back-to-back utterances.",
    )
    p.add_argument(
        "--blocksize",
        type=int,
        default=0,
        help="sounddevice blocksize (0 = auto). Smaller can reduce latency but may underrun on weak systems.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.002,
        help="Amplitude threshold to define 'audible' (float32 PCM). Lower if it never triggers.",
    )
    p.add_argument(
        "--pause-ms",
        type=int,
        default=150,
        help="Silence between utterances in milliseconds (prevents them from running together).",
    )
    args = p.parse_args()

    texts: List[str] = args.text

    print("[info] loading model...", flush=True)
    model = TTSModel.load_model()

    print(f"[info] extracting voice state ONCE from: {args.ref}", flush=True)
    voice_state = model.get_state_for_audio_prompt(args.ref)

    sr = int(model.sample_rate)
    out_latency_s = _device_latency_seconds()
    print(f"[metric] output latency estimate: {out_latency_s*1000:.1f} ms", flush=True)

    # Keep ONE output stream open for all utterances (lower jitter, less overhead)
    with sd.OutputStream(
        samplerate=sr,
        channels=1,
        dtype="float32",
        blocksize=args.blocksize,
        latency="low",
    ) as out:
        print("[info] streaming playback (multiple utterances)...", flush=True)

        for i, text in enumerate(texts, start=1):
            print(f"\n=== UTTERANCE {i}/{len(texts)} ===", flush=True)
            print(f"[text] {text}", flush=True)

            t_first_chunk_s, t_first_audible_est_s, audio_s, wall_s = (
                stream_one_utterance(
                    out=out,
                    model=model,
                    voice_state=voice_state,
                    text=text,
                    out_latency_s=out_latency_s,
                    audible_threshold=args.threshold,
                )
            )

            if t_first_chunk_s is not None:
                print(
                    f"[metric] submit→first_chunk: {t_first_chunk_s*1000:.1f} ms",
                    flush=True,
                )
            else:
                print("[warn] no chunk produced (unexpected)", flush=True)

            if t_first_audible_est_s is not None:
                print(
                    f"[metric] submit→first_audible (est): {t_first_audible_est_s*1000:.1f} ms "
                    f"(threshold={args.threshold})",
                    flush=True,
                )
            else:
                print(
                    "[warn] never detected audio above threshold; try lowering --threshold",
                    flush=True,
                )

            rtf = wall_s / max(audio_s, 1e-9)
            print(
                f"[info] audio seconds: {audio_s:.2f}s | wall time: {wall_s:.2f}s | RTF: {rtf:.2f}",
                flush=True,
            )

            # Small pause between utterances
            if args.pause_ms > 0 and i != len(texts):
                n = int(sr * (args.pause_ms / 1000.0))
                out.write(np.zeros((n, 1), dtype=np.float32))

    print(
        "\n[done] model loaded once, voice state extracted once, multiple texts generated.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[info] stopped.", file=sys.stderr)
        raise SystemExit(130)
