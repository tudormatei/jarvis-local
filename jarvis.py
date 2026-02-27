import asyncio
import logging
import threading
from jarvis_llm.jarvis_llm import chat_with_jarvis
from jarvis_tts.jarvis_tts import JarvisTTS
from jarvis_stt.jarvis_stt import JarvisSTT
from jarvis_ui.jarvis_ui import JarvisUI
from utils.args import (
    parse_arguments,
    InputMode,
    OutputMode,
    OutputInterface,
    PushToTalk,
    LogLevel,
)
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def run_jarvis_logic(
    input_voice_enabled,
    output_voice_enabled,
    ui_enabled,
    push_to_talk_enabled,
    jarvis_ui,
    jarvis_tts,
    jarvis_stt,
):
    async def handle_conversation():
        try:
            while True:
                if not input_voice_enabled:
                    user_input = input("You: ")
                else:
                    user_input = jarvis_stt.transcribe_user_audio(
                        push_to_talk=push_to_talk_enabled
                    )

                if ui_enabled and jarvis_ui:
                    jarvis_ui.print_message(isUser=True, message=user_input)
                elif (not ui_enabled) and input_voice_enabled:
                    print(f"You: {user_input}")

                if "bye" in user_input.lower() and (not ui_enabled):
                    print("JARVIS: Shutting down. Goodbye!")
                    break

                async for sentence in chat_with_jarvis(user_input):
                    if ui_enabled:
                        jarvis_ui.print_message(isUser=False, message=sentence)
                    elif not ui_enabled:
                        print(f"JARVIS: {sentence}")

                    if output_voice_enabled:
                        jarvis_tts.speak(sentence)

        except (KeyboardInterrupt, EOFError):
            print("JARVIS: Interrupted by user. Exiting...")
        finally:
            if jarvis_tts:
                jarvis_tts.stop()

    try:
        asyncio.run(handle_conversation())
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("JARVIS: Async loop interrupted by user. Exiting...")


def main():
    args = parse_arguments()
    setup_logging(args.log.name)

    input_voice_enabled = args.input == InputMode.VOICE
    output_voice_enabled = args.output == OutputMode.VOICE
    ui_enabled = args.interface == OutputInterface.UI
    push_to_talk_enabled = args.push_to_talk == PushToTalk.ON

    logger.info("Starting JARVIS with the following configuration:")
    logger.info(f"Input Mode: {args.input.name}")
    logger.info(f"Output Mode: {args.output.name}")
    logger.info(f"Output Interface: {args.interface.name}")
    logger.info(
        f"Push to Talk: {args.push_to_talk.name if args.push_to_talk else 'N/A'}"
    )

    jarvis_ui = None
    jarvis_tts = None

    if output_voice_enabled:
        jarvis_tts = JarvisTTS(
            speaker_sample="jarvis_tts/reference.wav",
            ui_enabled=ui_enabled,
        )

    if input_voice_enabled:
        jarvis_stt = JarvisSTT()

    if ui_enabled:
        jarvis_ui = JarvisUI(html_path="jarvis_ui/ui/index.html")
        logic_thread = threading.Thread(
            target=run_jarvis_logic,
            args=(
                input_voice_enabled,
                output_voice_enabled,
                ui_enabled,
                push_to_talk_enabled,
                jarvis_ui,
                jarvis_tts,
                jarvis_stt,
            ),
            daemon=True,
        )
        logic_thread.start()
        jarvis_ui.start()
    else:
        run_jarvis_logic(
            input_voice_enabled,
            output_voice_enabled,
            ui_enabled,
            push_to_talk_enabled,
            jarvis_ui,
            jarvis_tts,
            jarvis_stt,
        )


if __name__ == "__main__":
    main()
