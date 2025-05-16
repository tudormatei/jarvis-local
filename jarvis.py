import argparse
import asyncio
import logging
from jarvis_llm.jarvis_llm import chat_with_jarvis
from jarvis_tts.jarvis_tts import JarvisTTS
from jarvis_stt.jarvis_stt import transcribe_user_audio


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run JARVIS in text or voice mode.")
    parser.add_argument(
        '--mode', choices=['text', 'voice'], required=True, help='Input mode: text or voice')
    parser.add_argument('--push-to-talk', choices=['true', 'false'], default='true',
                        help='Use push-to-talk in voice mode (default: false)')
    parser.add_argument('--log', choices=['debug', 'info', 'warning', 'error', 'critical', 'none'],
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


logger = logging.getLogger(__name__)


async def main():
    args = parse_arguments()
    configure_logger(args.log)

    jarvis_tts = JarvisTTS()
    push_to_talk_enabled = args.push_to_talk.lower() == 'true'

    try:
        while True:
            if args.mode == 'text':
                user_input = input(f"\nYou:  ")
            else:
                user_input = transcribe_user_audio(
                    push_to_talk=push_to_talk_enabled)

            if "bye" in user_input.lower():
                print("JARVIS: Shutting down. Goodbye!")
                break

            async for sentence in chat_with_jarvis(user_input, is_streaming=True):
                print(f"JARVIS: {sentence}")
                await asyncio.to_thread(jarvis_tts.speak, sentence)

    except KeyboardInterrupt:
        print("JARVIS: Interrupted by user. Exiting...")
    finally:
        jarvis_tts.stop()

if __name__ == "__main__":
    asyncio.run(main())
