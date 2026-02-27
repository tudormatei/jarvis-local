import argparse
import os
import time
from pathlib import Path

import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr


def audio_seconds(path: str) -> float:
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def pick_device(device_arg: str) -> torch.device:
    device_arg = device_arg.lower()
    if device_arg == "cpu":
        # Optional: hide GPUs from torch so nothing CUDA-related can be used.
        # Do this BEFORE any CUDA checks / model creation.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Requested --device cuda but torch.cuda.is_available() is False"
            )
        return torch.device("cuda")

    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=str, default="sample.wav")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timestamps", action="store_true")
    ap.add_argument(
        "--amp", action="store_true", help="Use autocast FP16 on GPU if supported"
    )
    ap.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )

    # CPU control (optional)
    ap.add_argument(
        "--threads", type=int, default=None, help="torch.set_num_threads(N)"
    )
    ap.add_argument(
        "--interop", type=int, default=None, help="torch.set_num_interop_threads(N)"
    )

    args = ap.parse_args()

    # Thread settings must be set early (best before heavy work)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    if args.interop is not None:
        torch.set_num_interop_threads(args.interop)

    # Decide device
    device = pick_device(args.device)

    audio_path = str(Path(args.audio).expanduser().resolve())
    dur_s = audio_seconds(audio_path)

    print("== Environment ==")
    print("torch:", torch.__version__)
    print("requested device:", args.device)
    print("selected device:", device)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("cuda runtime:", torch.version.cuda)

    print(
        "threads:",
        torch.get_num_threads(),
        "| interop:",
        torch.get_num_interop_threads(),
    )

    # Load model explicitly onto chosen device
    # Note: map_location is helpful when forcing CPU to avoid any GPU tensors during load.
    kwargs = {}
    if device.type == "cpu":
        kwargs["map_location"] = "cpu"

    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2",
        **kwargs,
    )
    asr_model.eval()
    asr_model.to(device)

    if device.type == "cuda":
        # Good practice for performance (optional)
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

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
        dt, _, _ = run_once()
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
