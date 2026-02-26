import argparse
import time
from pathlib import Path

import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr


def audio_seconds(path: str) -> float:
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=str, default="sample.wav")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timestamps", action="store_true")
    ap.add_argument(
        "--amp", action="store_true", help="Use autocast FP16/BF16 on GPU if supported"
    )
    args = ap.parse_args()

    audio_path = str(Path(args.audio).expanduser().resolve())
    dur_s = audio_seconds(audio_path)

    print("== Environment ==")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("cuda runtime:", torch.version.cuda)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )
    asr_model.eval()
    asr_model.to(device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    def run_once():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if args.amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = asr_model.transcribe([audio_path], timestamps=args.timestamps)
        else:
            out = asr_model.transcribe([audio_path], timestamps=args.timestamps)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        text = out[0].text if hasattr(out[0], "text") else str(out[0])
        return (t1 - t0), text, out

    print("\n== Warmup ==")
    for i in range(args.warmup):
        dt, text, _ = run_once()
        print(f"warmup {i+1}: {dt:.4f}s")

    print("\n== Timed runs ==")
    times = []
    last_text = None
    last_out = None
    for i in range(args.runs):
        dt, text, out = run_once()
        times.append(dt)
        last_text = text
        last_out = out
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

    if args.timestamps:
        ts = getattr(last_out[0], "timestamp", None)
        if ts and "segment" in ts:
            print("\n== Segment timestamps ==")
            for seg in ts["segment"]:
                print(f"{seg['start']:.2f}s - {seg['end']:.2f}s : {seg['segment']}")


if __name__ == "__main__":
    main()
