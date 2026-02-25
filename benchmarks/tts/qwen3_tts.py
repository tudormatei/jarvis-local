import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def audio_seconds(wav: np.ndarray, sr: int) -> float:
    return float(wav.shape[0]) / float(sr)


def ms(x: float) -> float:
    return x * 1000.0


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Use Base for voice cloning. 0.6B fastest, 1.7B higher quality.",
    )
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument(
        "--attn",
        default="flash_attention_2",
        help="attn_implementation passed to from_pretrained (set empty '' to skip)",
    )

    ap.add_argument(
        "--ref_audio", required=True, help="Path to reference audio (~10s)."
    )
    ap.add_argument(
        "--ref_text", default="", help="Transcript of reference audio (recommended)."
    )
    ap.add_argument(
        "--xvec_only",
        action="store_true",
        help="Use x_vector_only_mode=True (no ref_text needed; may reduce quality).",
    )

    ap.add_argument(
        "--text",
        default="Hello! This is a streaming voice cloning benchmark.",
        help="Synthesis text.",
    )
    ap.add_argument(
        "--language",
        default="English",
        help="English/Chinese/... (or Auto if supported)",
    )

    # Streaming knobs
    ap.add_argument("--emit_every_frames", type=int, default=12)
    ap.add_argument("--decode_window_frames", type=int, default=80)
    ap.add_argument("--overlap_samples", type=int, default=512)

    # Two-phase first chunk knobs
    ap.add_argument("--first_chunk_emit_every", type=int, default=5)
    ap.add_argument("--first_chunk_decode_window", type=int, default=48)
    ap.add_argument("--first_chunk_frames", type=int, default=48)

    # Benchmark control
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--save_wav", action="store_true")
    ap.add_argument("--out_dir", default="bench_out")

    # Optional: approximate extra device buffer latency (audio backend)
    ap.add_argument(
        "--assume_output_buffer_ms",
        type=float,
        default=0.0,
        help="Add this many ms to TTFC to approximate when you actually start hearing.",
    )

    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"Loading model: {args.model}")
    t0 = time.perf_counter()
    kwargs = dict(
        device_map=args.device,
        torch_dtype=torch_dtype,
    )
    if args.attn.strip():
        kwargs["attn_implementation"] = args.attn.strip()

    model = Qwen3TTSModel.from_pretrained(args.model, **kwargs)
    cuda_sync()
    print(f"Model load time: {time.perf_counter() - t0:.3f}s")

    # Enable streaming optimizations (compile + CUDA graphs)
    if hasattr(model, "enable_streaming_optimizations"):
        print("Enabling streaming optimizations...")
        model.enable_streaming_optimizations(
            decode_window_frames=args.decode_window_frames,
            use_compile=True,
            compile_mode="reduce-overhead",
        )
    else:
        print(
            "WARNING: enable_streaming_optimizations() not found. "
            "Make sure you're using the streaming fork."
        )

    # Build reusable clone prompt once
    prompt_kwargs = dict(
        ref_audio=args.ref_audio, x_vector_only_mode=bool(args.xvec_only)
    )
    if not args.xvec_only:
        if not args.ref_text.strip():
            raise SystemExit("ref_text is required unless --xvec_only is set")
        prompt_kwargs["ref_text"] = args.ref_text

    print("Creating voice clone prompt (one-time)...")
    t0 = time.perf_counter()
    prompt = model.create_voice_clone_prompt(**prompt_kwargs)
    cuda_sync()
    prompt_time = time.perf_counter() - t0
    print(f"Prompt build time: {prompt_time:.3f}s")

    def run_once(text: str, run_idx: int, save: bool):
        chunks = []
        sr = None

        cuda_sync()
        t_start = time.perf_counter()

        first_chunk_time = None
        first_chunk_audio_dur = None

        for chunk, sr in model.stream_generate_voice_clone(
            text=text,
            language=args.language,
            voice_clone_prompt=prompt,
            emit_every_frames=args.emit_every_frames,
            decode_window_frames=args.decode_window_frames,
            overlap_samples=args.overlap_samples,
            first_chunk_emit_every=args.first_chunk_emit_every,
            first_chunk_decode_window=args.first_chunk_decode_window,
            first_chunk_frames=args.first_chunk_frames,
        ):
            # This is the moment you could start playing audio.
            if first_chunk_time is None:
                cuda_sync()
                first_chunk_time = time.perf_counter() - t_start
                first_chunk_audio_dur = audio_seconds(chunk, sr)

            chunks.append(chunk)

        cuda_sync()
        total_time = time.perf_counter() - t_start

        wav = (
            np.concatenate(chunks, axis=0)
            if chunks
            else np.zeros((0,), dtype=np.float32)
        )
        total_audio_dur = audio_seconds(wav, sr) if sr and wav.size else 0.0
        rtf = (total_time / total_audio_dur) if total_audio_dur > 0 else float("inf")

        if save and wav.size:
            outp = Path(args.out_dir) / f"out_run{run_idx}.wav"
            sf.write(str(outp), wav, sr)

        # “When you start hearing” ≈ first chunk arrives + output device buffer
        audible_ttfc = first_chunk_time + (args.assume_output_buffer_ms / 1000.0)

        return {
            "ttfc_s": first_chunk_time,
            "audible_ttfc_s": audible_ttfc,
            "first_chunk_audio_s": first_chunk_audio_dur or 0.0,
            "total_time_s": total_time,
            "audio_dur_s": total_audio_dur,
            "rtf": rtf,
            "sr": sr,
        }

    # Warmup (compile/graphs/etc.)
    for i in range(args.warmup):
        print(f"Warmup {i+1}/{args.warmup}...")
        _ = run_once("Warmup.", run_idx=-(i + 1), save=False)

    # Benchmark
    results = []
    for r in range(args.runs):
        print(f"\nRun {r+1}/{args.runs}")
        res = run_once(args.text, run_idx=r + 1, save=args.save_wav)

        print(f"  TTFC (chunk ready):         {ms(res['ttfc_s']):.1f} ms")
        if args.assume_output_buffer_ms > 0:
            print(
                f"  TTFC (audible est.):        {ms(res['audible_ttfc_s']):.1f} ms "
                f"(+{args.assume_output_buffer_ms:.1f} ms buffer)"
            )
        print(f"  First chunk audio duration: {ms(res['first_chunk_audio_s']):.1f} ms")
        print(f"  Total gen time:             {res['total_time_s']:.3f} s")
        print(f"  Total audio duration:       {res['audio_dur_s']:.3f} s")
        print(f"  RTF (lower=better):         {res['rtf']:.3f}")

        results.append(res)

    # Summary
    ttfc = np.array([x["ttfc_s"] for x in results], dtype=np.float64)
    audible = np.array([x["audible_ttfc_s"] for x in results], dtype=np.float64)
    rtf = np.array([x["rtf"] for x in results], dtype=np.float64)
    total = np.array([x["total_time_s"] for x in results], dtype=np.float64)

    def pct(a, p):
        return float(np.percentile(a, p))

    print("\n=== SUMMARY ===")
    print(f"Model: {args.model}")
    print(f"Prompt build (one-time): {prompt_time:.3f}s")
    print(
        f"TTFC p50/p90: {ms(pct(ttfc,50)):.1f} / {ms(pct(ttfc,90)):.1f} ms (chunk-ready)"
    )
    if args.assume_output_buffer_ms > 0:
        print(
            f"Audible TTFC p50/p90: {ms(pct(audible,50)):.1f} / {ms(pct(audible,90)):.1f} ms (estimated)"
        )
    print(f"Total time p50/p90: {pct(total,50):.3f} / {pct(total,90):.3f} s")
    print(f"RTF p50/p90: {pct(rtf,50):.3f} / {pct(rtf,90):.3f}")


if __name__ == "__main__":
    main()
