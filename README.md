# JARVIS Local Assistant

**JARVIS Local** is a modular, privacy-focused voice/text assistant that runs entirely on your machine. It combines local Large Language Model (LLM) inference, high-quality Text-to-Speech (TTS), and fast Speech-to-Text (STT) for a seamless conversational AI experience—no cloud required.

[![JARVIS](docs/ui.gif)](docs/ui.gif)

## TODO:

- Update TTS model to a quicker and better sounding one Qwen3 tts, f5 tts, maybe?
- MAKE UI SOUND WAVE PERFECT

## Features

- **Local LLM (Ollama/3B):** Fast, streaming responses with conversation memory.
- **Text-to-Speech (TTS):** High-quality, low-latency voice synthesis using XTTS.
- **Speech-to-Text (STT):** Accurate, real-time transcription with Whisper.
- **Push-to-Talk:** Optional mode for precise voice input control.
- **Text Mode:** CLI-based chat for keyboard-only interaction.
- **Configurable Logging:** Adjustable verbosity for debugging or silent operation.
- **Cross-platform:** Designed for Windows, but adaptable to Linux/Mac.
- **UI or CLI:** Choose between a graphical interface or command-line.

## Pipeline Latency

This evaluation measures system latency **after the user finishes speaking**.  
Microphone capture duration is excluded since the user is actively talking during that time.

---

### Measured Stages

**Speech-to-Text (transcription only)**

- Transcription time: **~617 ms**

**LLM Inference**

- Time to first token: **~395 ms**
- Full streaming duration: **~1.8 s**

**Text-to-Speech (first audio output)**

- First chunk received: ~5 ms
- TTS inference (first segment): ~873 ms
- Playback buffer: ~49 ms

### Time to First Audible Response

From end-of-speech → first spoken word:

- STT transcription: ~0.62 s
- LLM first token: ~0.40 s
- TTS first audio generation: ~0.92 s

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

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/jarvis-local.git
   cd jarvis-local
   ```

2. **Install dependencies:**  
   (Recommended: use conda or miniconda)

   ```sh
   conda env create -f environment.yml
   ```

3. **Download models:**
   - **LLM:** Uses Ollama (`ollama run jarvis:3b`) or your configured local model.
   - **TTS:** The finetuned model is already included in the repository.
   - **STT:** Whisper model is auto-downloaded on first run.

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
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for fast STT
- Open source community for tools and inspiration

---

**JARVIS Local** — Your private, local AI assistant.
