# JARVIS Local Assistant

**JARVIS Local** is a modular, privacy-focused voice/text assistant inspired by the iconic Iron Man AI that runs entirely on your machine. It combines local Large Language Model (LLM) inference, high-quality Text-to-Speech (TTS), and fast Speech-to-Text (STT) for a seamless conversational AI experience—no cloud required.

[![JARVIS](docs/ui.gif)](docs/ui.gif)

## Features

- **Local LLM:** Fast, streaming responses with conversation memory using Ollama.
- **Text-to-Speech (TTS):** High-quality, low-latency voice synthesis using Pocket TTS.
- **Speech-to-Text (STT):** Multilingual transcription with Faster-Whisper on the CPU.
- **Push-to-Talk:** Optional mode for precise voice input control.
- **Text Mode:** CLI-based chat for keyboard-only interaction.
- **Configurable Logging:** Adjustable verbosity for debugging or silent operation.
- **Cross-platform:** Designed for Windows, but adaptable to Linux/Mac.
- **UI or CLI:** Choose between a graphical interface or command-line.

## Voice Processing

Speech recognition uses Faster-Whisper's multilingual `small` model with CPU int8 inference, so it works without NVIDIA CUDA. Italian speech output uses Pocket TTS's built-in `giovanni` voice; it does not require voice-cloning access. Ollama runs the language model locally. The first voice-mode launch downloads the speech models and may take longer.

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
| Python      | 3.11                               |
| Ollama      | Latest                            |
| Microphone  | Required for voice input           |
| Speakers    | Required for voice output          |

---

### 1. Clone the repository

```sh
git clone https://github.com/tudormatei/jarvis-local.git
cd jarvis-local
```

### 2. Create and activate a virtual environment

```sh
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 3. Install project requirements

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch is installed without CUDA and speech recognition runs on the CPU, so an NVIDIA GPU is not required.

### 4. Install Ollama and download the language model

Install Ollama from [ollama.com/download](https://ollama.com/download), then run:

```sh
ollama pull qwen2.5:3b
```

JARVIS starts Ollama automatically if it is not already running. The Faster-Whisper `small` model and Pocket TTS model are downloaded automatically the first time voice mode starts.

## Usage

### **Basic Example**

```sh
python jarvis.py --input voice --output voice --interface ui --push-to-talk on
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

## Building a Windows Executable

You can package **JARVIS Local** into a standalone Windows executable using PyInstaller.

### 1. Install PyInstaller

Activate your conda environment:

```sh
conda activate jarvis
pip install pyinstaller
```

---

### 2. Build the Executable

From the project root directory, run:

```sh
pyinstaller Jarvis.spec --clean --noconfirm
```

Or to generate a new spec file:

```sh
pyinstaller jarvis_launcher.py ^
  --name Jarvis ^
  --onedir ^
  --clean ^
  --noconfirm ^
  --icon=jarvis_ui\ui\assets\icon.ico ^
  --add-data "jarvis_ui\ui;jarvis_ui\ui" ^
  --add-data "jarvis_tts\reference.wav;jarvis_tts" ^
  --collect-all pocket_tts ^
  --collect-all nemo
```

---

### 3. Run the Application

After building, the executable will be located at:

```
dist/Jarvis/Jarvis.exe
```

Double-click to launch.

---

### Notes

- Ollama must be installed on the system.
- Required models must be downloaded beforehand.
- The first launch may take longer due to model initialization.
- For debugging, remove `--windowed` to see console output, or change `console=True` in the `.spec` file.

## License

This project is for personal, non-commercial use. See individual model licenses for details.

## Acknowledgments

- [Ollama](https://ollama.com/) for local LLM serving
- [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) for TTS with streaming capabilities and voice cloning
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) for speech recognition
- Open source community for tools and inspiration

---

**JARVIS Local** — Your private, local AI assistant.
