# Pocket TTS Streaming Latency Benchmark

## Test Setup

- Model: Pocket TTS (voice cloning enabled)
- Device: CPU (no GPU)
- Voice reference: `reference.wav`
- Output device latency (PortAudio estimate): ~90 ms
- Mode: Streaming (`generate_audio_stream`)
- Model loaded once
- Voice state extracted once
- Multiple utterances generated sequentially in a single session

---

## Measured Metrics

Definitions:

- **submit→first_chunk**  
  Time from when text is submitted to the model until the first audio chunk is produced.

- **submit→first_audible (est)**  
  Estimated time from text submission until sound becomes audible at speakers.  
  Includes:
  - Model generation latency
  - First non-silence sample offset
  - Output device latency (~90 ms)

- **RTF (Real-Time Factor)**  
  Wall time / audio duration
  - `1.0` = real-time
  - `< 1.0` = faster than real-time
  - `> 1.0` = slower than real-time

---

## Results

### Utterance 1

Text:

> "Hello, this is utterance one."

- submit→first_chunk: **355.5 ms**
- submit→first_audible (est): **445.9 ms**
- Audio duration: 2.64 s
- Wall time: 2.91 s
- RTF: 1.10

---

### Utterance 2

Text:

> "This is utterance two, right after the first."

- submit→first_chunk: **187.7 ms**
- submit→first_audible (est): **277.9 ms**
- Audio duration: 2.96 s
- Wall time: 3.08 s
- RTF: 1.04

---

### Utterance 3

Text:

> "And this is utterance three."

- submit→first_chunk: **152.8 ms**
- submit→first_audible (est): **242.9 ms**
- Audio duration: 2.72 s
- Wall time: 2.80 s
- RTF: 1.03

---

## Observations

### 1️⃣ Warm-up Effect

The first utterance is significantly slower:

- First audible: **~446 ms**
- Later utterances: **~243–278 ms**

After the initial generation:

- CPU kernels are warm
- Memory is cached
- Model state is fully resident
- Audio stream already active

This cuts latency nearly in half.

---

### 2️⃣ Real-Time Factor

RTF improves slightly over time:

| Utterance | RTF  |
| --------- | ---- |
| 1         | 1.10 |
| 2         | 1.04 |
| 3         | 1.03 |

Pocket TTS approaches real-time generation after warm-up.

---

### 3️⃣ True Perceived Latency

For warmed system:

- **~240–280 ms total to audible speech**
- Including ~90 ms hardware output latency

Model-only generation latency (excluding device latency):

- ~150–190 ms

This is extremely low for CPU voice cloning.

---

## Why This Is Fast

- Small model (~100M parameters)
- CPU-optimized architecture
- Streaming generation
- No autoregressive diffusion
- No large conditioning encoder per utterance (voice state reused)

---

## Comparison Insight

Typical XTTSv2 streaming latency (CPU) is often:

- 800 ms – 1500+ ms to first audible speech

Pocket TTS here:

- ~250 ms (warm)
- ~450 ms (cold)

This is a **3–5× reduction in latency**.

---

## Final Conclusion

With:

- Model loaded once
- Voice state extracted once
- Persistent audio stream

Pocket TTS achieves:

- ~150 ms model generation latency
- ~250 ms perceived speech latency
- Near real-time generation (RTF ≈ 1.03)

For interactive assistants, this is effectively instant.

---

## Optimization Potential

Further reductions possible by:

- Lowering output device buffer
- Reducing sounddevice blocksize
- Using WASAPI exclusive mode
- Keeping process alive (no cold starts)
- Using exported `.safetensors` voice state

---

## Summary

Pocket TTS (CPU) delivers sub-300 ms interactive voice cloning latency in steady state.

This makes it highly suitable for:

- Real-time assistants
- Low-latency conversational agents
- Local voice interfaces
- Edge devices without GPUs
