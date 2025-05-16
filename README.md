# JARVIS Local Assistant

**JARVIS Local** is a modular, privacy-focused voice/text assistant that runs entirely on your machine. It combines local Large Language Model (LLM) inference, high-quality Text-to-Speech (TTS), and fast Speech-to-Text (STT) for a seamless conversational AI experience—no cloud required.

## Features

- **Local LLM (Ollama/3B):** Fast, streaming responses with conversation memory.
- **Text-to-Speech (TTS):** High-quality, low-latency voice synthesis using XTTS.
- **Speech-to-Text (STT):** Accurate, real-time transcription with Whisper.
- **Push-to-Talk:** Optional mode for precise voice input control.
- **Text Mode:** CLI-based chat for keyboard-only interaction.
- **Configurable Logging:** Adjustable verbosity for debugging or silent operation.
- **Cross-platform:** Designed for Windows, but adaptable to Linux/Mac.

## Project Structure

```
jarvis-local/
│
├── jarvis.py                # Main entry point (CLI interface)
├── jarvis_llm/
│   └── jarvis_llm.py        # LLM streaming and conversation logic
├── jarvis_tts/
│   └── jarvis_tts.py        # Text-to-Speech engine and playback
├── jarvis_stt/
    └── jarvis_stt.py        # Speech-to-Text with push-to-talk and VAD
```

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/jarvis-local.git
   cd jarvis-local
   ```

2. **Install dependencies:**
   I suggest using conda or miniconda for this.

   ```sh
   conda env create -f environment.yml
   ```

3. **Download models:**

   - **LLM:** Uses Ollama (`ollama run jarvis:3b`) or your configured local model.
   - **TTS:** The finetuned model is already included in the repository.
   - **STT:** Whisper model is auto-downloaded on first run.

## Usage

### **Text Mode**

```sh
python jarvis.py --mode text
```

Type your message and receive spoken responses.

### **Voice Mode**

1. Push to talk

```sh
python jarvis.py --mode voice --push-to-talk true
```

Hold the spacebar (default) to speak, release to transcribe and get a response.

2. Automatic voice detection (slower):

```sh
python jarvis.py --mode voice --push-to-talk false
```

#### **Command-line Options**

- `--mode [text|voice]` — Select input mode.
- `--push-to-talk [true|false]` — Enable/disable push-to-talk in voice mode.
- `--log [debug|info|warning|error|critical|none]` — Set logging level.

## License

This project is for personal, non-commercial use. See individual model licenses for details.

## Acknowledgments

- [Ollama](https://ollama.com/) for local LLM serving
- [Coqui TTS](https://github.com/coqui-ai/TTS) for XTTS + finetuning
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for fast STT
- Open source community for tools and inspiration

---

**JARVIS Local** — Your private, local AI assistant.
