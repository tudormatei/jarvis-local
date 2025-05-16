import argparse
import asyncio
import logging
import threading
from jarvis_llm.jarvis_llm import chat_with_jarvis
from jarvis_tts.jarvis_tts import JarvisTTS
from jarvis_stt.jarvis_stt import transcribe_user_audio
from jarvis_ui.jarvis_ui import JarvisUI

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run JARVIS in text or voice mode.")
    parser.add_argument(
        '--mode', choices=['text', 'voice'], required=True, help='Input mode: text or voice')
    parser.add_argument('--push-to-talk', choices=['true', 'false'],
                        default='true', help='Use push-to-talk in voice mode (default: true)')
    parser.add_argument(
        '--ui', choices=['true', 'false'], default='true', help='Display UI (default: true)')
    parser.add_argument('--log', choices=['info', 'none'],
                        default='info', help='Set logging level (default: info)')
    return parser.parse_args()


def configure_logger(log_level):
    if log_level == 'none':
        logging.disable(logging.CRITICAL)
    else:
        level = getattr(logging, log_level.upper(), logging.INFO)
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            level=level
        )

        for noisy_logger in [
            "websockets",
            "pywebview",
            "bottle",
            "werkzeug",  # Flask fallback
        ]:
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def run_jarvis_logic(args, jarvis_ui, jarvis_tts):
    push_to_talk_enabled = args.push_to_talk.lower() == 'true'

    async def handle_conversation():
        try:
            while True:
                if args.mode == 'text':
                    user_input = input("\nYou:  ")
                else:
                    user_input = transcribe_user_audio(
                        push_to_talk=push_to_talk_enabled)

                if jarvis_ui:
                    jarvis_ui.print_message(isUser=True, message=user_input)

                if "bye" in user_input.lower():
                    print("JARVIS: Shutting down. Goodbye!")
                    break

                async for sentence in chat_with_jarvis(user_input, is_streaming=True):
                    if jarvis_ui:
                        jarvis_ui.print_message(isUser=False, message=sentence)
                    else:
                        print(f"JARVIS: {sentence}")
                    jarvis_tts.speak(sentence)
        except KeyboardInterrupt:
            print("JARVIS: Interrupted by user. Exiting...")
        finally:
            jarvis_tts.stop()

    asyncio.run(handle_conversation())


def main():
    args = parse_arguments()
    configure_logger(args.log)
    use_ui = args.ui.lower() == 'true'

    jarvis_tts = JarvisTTS(
        model_path="jarvis_tts/models/jarvis_v2/",
        speaker_sample="jarvis_tts/models/jarvis_v2/reference.wav",
        should_stream=use_ui
    )

    if use_ui:
        jarvis_ui = JarvisUI(html_path="jarvis_ui/ui/index.html")
        logic_thread = threading.Thread(target=run_jarvis_logic, args=(
            args, jarvis_ui, jarvis_tts), daemon=True)
        logic_thread.start()
        jarvis_ui.start()  # Must be on main thread
    else:
        run_jarvis_logic(args, None, jarvis_tts)


if __name__ == "__main__":
    main()
