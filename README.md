# JARVIS Local Assistant

**JARVIS Local** is a modular, privacy-focused voice/text assistant that runs entirely on your machine. It combines local Large Language Model (LLM) inference, high-quality Text-to-Speech (TTS), and fast Speech-to-Text (STT) for a seamless conversational AI experience—no cloud required.

[![JARVIS](docs/ui.gif)](docs/ui.gif)

## Features

- **Local LLM:** Fast, streaming responses with conversation memory using Ollama.
- **Text-to-Speech (TTS):** High-quality, low-latency voice synthesis using XTTSv2.
- **Speech-to-Text (STT):** Accurate, real-time transcription with NVIDIA Parakeet.
- **Push-to-Talk:** Optional mode for precise voice input control.
- **Text Mode:** CLI-based chat for keyboard-only interaction.
- **Configurable Logging:** Adjustable verbosity for debugging or silent operation.
- **Cross-platform:** Designed for Windows, but adaptable to Linux/Mac.
- **UI or CLI:** Choose between a graphical interface or command-line.

## Latency Evaluation

This evaluation measures system latency **after the user finishes speaking**.  
Microphone capture duration is excluded since the user is actively talking during that time.

### Measured Stages

**Speech-to-Text (transcription only)**

- Transcription time: **~617 ms**

**LLM Inference**

- Time to first speakable token (sentence): **~395 ms**

**Text-to-Speech**

- TTS inference (time to first audio output): **~873 ms**

### Time to First Audible Response

From end-of-speech → first spoken word:

- STT transcription: ~0.6 s
- LLM first token: ~0.40 s
- TTS first audio generation: ~0.9 s

**Total latency: ~1.9 – 2.0 seconds**

### Why It Feels Fast

The system streams intelligently:

- LLM streams tokens progressively
- TTS begins before LLM finishes
- Playback starts before full response is generated

Because of this overlap, JARVIS begins speaking in ~2 seconds while the rest of the response is still processing.

## Project Structure

```
jarvis-local/
│
├── jarvis.py                # Main entry point (service orchestrator)
├── jarvis_llm/
│   └── jarvis_llm.py        # LLM streaming and conversation logic
├── jarvis_tts/
│   └── jarvis_tts.py        # Text-to-Speech engine and playback
├── jarvis_stt/
│   └── jarvis_stt.py        # Speech-to-Text with push-to-talk and VAD
├── jarvis_ui/
│   └── jarvis_ui.py         # UI logic (webview)
│   └── ui/                  # UI frontend (HTML/JS/CSS)
└── utils/
    └── args.py              # Argument parsing and enums
```

## Installation

### Prerequisites

| Requirement | Version                           |
| ----------- | --------------------------------- |
| OS          | Windows (tested on Windows 10/11) |
| Python      | 3.11.11                           |
| CUDA Driver | 12.4+                             |
| Miniconda   | Latest                            |

---

### 1. Clone the repository

```sh
git clone https://github.com/yourusername/jarvis-local.git
cd jarvis-local
```

### 2. Recreate the conda environment

```sh
conda env create -f environment-lock.yml
conda activate jarvis-local
```

> If you run into issues with PyTorch not finding CUDA, reinstall it manually:
>
> ```sh
> pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
> ```

### 3. Download models

- **LLM** Uses Ollama to download and run models locally. You can download your own model from the archive, add personality to it using a Modelfile and replace `MODEL_NAME = "jarvis:1b"` inside `jarvis_llm/jarvis_llm.py` with the selected model
- **TTS:** The finetuned model is already included in the repository.
- **STT:** NVIDIA Parakeet model is auto-downloaded on first run.

---

### Updating the environment (maintainers only)

After installing or changing any dependencies, regenerate and commit the lockfiles:

```sh
# Full conda export (includes pip packages, Python version, and conda base packages)
conda env export > environment-lock.yml

# Pure pip lockfile (backup, useful if conda env create fails)
pip freeze > requirements-lock.txt

git add environment-lock.yml requirements-lock.txt
git commit -m "chore: update environment lockfiles"
```

> **Note:** `environment-lock.yml` is the source of truth. `requirements-lock.txt` is a fallback.  
> Do not manually edit either file — always regenerate them from the live environment.

## Usage

### **Basic Example**

```sh
python jarvis.py --input text --output voice --interface cli
```

### **Arguments**

| Argument         | Choices         | Required       | Description                                    |
| ---------------- | --------------- | -------------- | ---------------------------------------------- |
| `--input`        | `text`, `voice` | Yes            | Input mode: keyboard or microphone             |
| `--output`       | `text`, `voice` | Yes            | Output mode: text or spoken                    |
| `--interface`    | `cli`, `ui`     | Yes            | Output interface: command-line or graphical UI |
| `--push-to-talk` | `on`, `off`     | If input=voice | Enable/disable push-to-talk for voice input    |
| `--log`          | `info`, `none`  | No             | Set logging level (default: info)              |

#### **Examples**

- **Text input, spoken output, CLI:**
  ```sh
  python jarvis.py --input text --output voice --interface cli
  ```
- **Voice input, text output, UI, push-to-talk:**
  ```sh
  python jarvis.py --input voice --output text --interface ui --push-to-talk on
  ```
- **Voice input, spoken output, CLI, no push-to-talk:**
  ```sh
  python jarvis.py --input voice --output voice --interface cli --push-to-talk off
  ```

#### **Notes**

- `--push-to-talk` is **required** if `--input voice` is selected.
- Do **not** use `--push-to-talk` with `--input text`.
- For UI mode, a browser window will open for interaction.

## License

This project is for personal, non-commercial use. See individual model licenses for details.

## Acknowledgments

- [Ollama](https://ollama.com/) for local LLM serving
- [Coqui TTS](https://github.com/coqui-ai/TTS) for XTTS + finetuning
- [NVIDIA Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) for fast STT
- Open source community for tools and inspiration

---

**JARVIS Local** — Your private, local AI assistant.
