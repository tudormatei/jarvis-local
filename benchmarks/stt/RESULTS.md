# Speech-to-Text Benchmark Results

## Hardware & Environment

- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU
- **CUDA Runtime:** 12.4
- **PyTorch:** 2.6.0+cu124
- **OS:** Windows
- **Audio Format:** 16 kHz mono WAV
- **Test Clip Duration:** 7.43 seconds
- **Runs per test:** 9 timed runs (after 2 warmup runs)
- **Metric reported:** Median

---

# Models Evaluated

1. **NVIDIA Parakeet TDT 0.6B v2** (NeMo)
2. **faster-whisper small.en**
   - int8, beam_size=1
   - int8, beam_size=5

---

# Speed Results

## NVIDIA Parakeet TDT 0.6B v2

| Metric      | Value              |
| ----------- | ------------------ |
| Median Time | **0.1708 s**       |
| RTF         | **0.0230**         |
| RTFx        | **43.5× realtime** |

**Interpretation:**  
Parakeet transcribes the 7.43s clip in ~0.17s, running **~43× faster than realtime**.

---

## faster-whisper small.en (int8, beam_size=1)

| Metric      | Value             |
| ----------- | ----------------- |
| Median Time | **0.9542 s**      |
| RTF         | **0.1283**        |
| RTFx        | **7.8× realtime** |

---

## faster-whisper small.en (int8, beam_size=5)

| Metric      | Value             |
| ----------- | ----------------- |
| Median Time | **0.9619 s**      |
| RTF         | **0.1294**        |
| RTFx        | **7.7× realtime** |

---

# Speed Comparison

| Model                     | Median Time | RTFx      | Relative Speed                |
| ------------------------- | ----------- | --------- | ----------------------------- |
| **Parakeet 0.6B**         | 0.1708 s    | **43.5×** | **~5.6× faster than Whisper** |
| Whisper small.en (beam=1) | 0.9542 s    | 7.8×      | Baseline                      |
| Whisper small.en (beam=5) | 0.9619 s    | 7.7×      | Slightly slower               |

**Observation:**  
Parakeet is approximately **5–6× faster** than faster-whisper small.en on this GPU for this clip.

---

# Accuracy Comparison (Qualitative)

### Reference Transcript

> Well, I don't wish to see it any more, observed Phebe, turning away her eyes. It is certainly very like the old portrait.

---

## Parakeet Output

> Well, I don't wish to see it any more, observed Phebe, turning away her eyes. It is certainly very like the old portrait.

**Notes:**

- Exact match
- Correct punctuation
- Correct capitalization
- Preserved original spelling

---

## faster-whisper Output (beam=1 & beam=5 identical)

> Well, I don't wish to see it anymore, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.

**Differences:**

- "any more" → "anymore"
- "Phebe" → "Phoebe"

**Notes:**

- Linguistically valid normalization
- Minor deviation from original spelling

---

# Accuracy Summary

| Model            | Transcript Fidelity | Notes                                         |
| ---------------- | ------------------- | --------------------------------------------- |
| **Parakeet**     | **Exact match**     | Perfect reproduction of spelling & formatting |
| Whisper small.en | Minor normalization | Modernized spelling + merged word             |

---

# Overall Conclusion

### 🚀 Speed Winner: NVIDIA Parakeet

- ~43× realtime
- ~5–6× faster than Whisper small.en
- Stable timing across runs

### 🧠 Accuracy

- Both models highly accurate
- Parakeet reproduced transcript exactly
- Whisper slightly normalized spelling

---

# Final Verdict (RTX 3050 Laptop GPU)

On this hardware:

- **Parakeet delivers significantly higher throughput**
- Accuracy is comparable across models
- faster-whisper small.en remains strong but substantially slower in this configuration
