import ast
import asyncio
import re
import ollama
import logging

from jarvis_llm.tools.tools import get_user_info, get_weather_report, play_song

logger = logging.getLogger(__name__)

MODEL_NAME = "jarvis:1b-new"  # jarvis:1b, jarvis:3b, jarvis-tool
TOOLS_ENABLED = MODEL_NAME == "jarvis-tool"

conversation_history = []
MAX_INTERACTIONS = 3


# make sure to keep this in sync with the tools.py file and Modelfile defined functions
tool_registry = {
    "get_user_info": get_user_info,
    "get_weather_report": get_weather_report,
    "play_song": play_song,
}

TERMINATORS = {".", "!", "?"}
CLOSERS = {'"', "'", ")", "]", "}"}

ABBREV_SUFFIXES = (
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "jr.",
    "sr.",
    "e.g.",
    "i.e.",
    "vs.",
    "etc.",
    "approx.",
)


def detect_and_handle_tool_call(response_text):
    tool_call_pattern = r"\[(\w+)\((.*?)\)\]"
    match = re.search(tool_call_pattern, response_text)
    if not match:
        return None

    func_name = match.group(1)
    params_str = match.group(2)

    try:
        fake_call = f"f({params_str})"
        parsed = ast.parse(fake_call, mode="eval")
        if not isinstance(parsed.body, ast.Call):
            raise ValueError("Not a valid function call")

        param_dict = {kw.arg: ast.literal_eval(kw.value) for kw in parsed.body.keywords}

    except Exception as e:
        logger.info("LLM Failed parsing tool parameters.")
        return None

    tool_func = tool_registry.get(func_name)
    if not tool_func:
        logger.info("LLM Unknown tool.")
        return None

    try:
        logger.info(f"LLM Calling tool {func_name} with params {param_dict}.")
        return tool_func(**param_dict)
    except Exception as e:
        logger.info(f"LLM Error while executing tool {func_name}.")
        return None


def update_conversation_history(user_input, assistant_response):
    # Append user input and assistant response to the conversation history
    conversation_history.append({"role": "User", "content": user_input})
    conversation_history.append({"role": "Assistant", "content": assistant_response})

    # Keep only the last N interactions (where each interaction is a user-assistant pair)
    while len(conversation_history) > MAX_INTERACTIONS * 2:
        # Remove the oldest exchange (both user and assistant)
        conversation_history.pop(0)


def handle_sentence_endings(buf: str):
    if not buf:
        return None, buf

    n = len(buf)
    i = 0

    while i < n:
        ch = buf[i]

        if ch in TERMINATORS:
            if (
                ch == "."
                and i > 0
                and i + 1 < n
                and buf[i - 1].isdigit()
                and buf[i + 1].isdigit()
            ):
                i += 1
                continue

            if ch == "." and i + 1 < n and buf[i + 1] == ".":
                # consume the whole run of dots
                k = i + 2
                while k < n and buf[k] == ".":
                    k += 1
                i = k
                continue

            j = i + 1
            while j < n and buf[j] in CLOSERS:
                j += 1

            sentence = buf[:j].strip()

            lower = sentence.lower()
            if lower.endswith(ABBREV_SUFFIXES):
                return None, buf  # wait for more context

            remaining = buf[j:].lstrip()
            return sentence, remaining

        i += 1

    return None, buf


async def chat_with_jarvis(input_text):
    messages = conversation_history + [{"role": "user", "content": input_text}]
    current_characters = ""
    full_response = ""

    try:
        logger.info("LLM Started inference.")
        response_stream = ollama.chat(MODEL_NAME, messages=messages, stream=True)

        first_chunk = False
        first_sentence = False
        for part in response_stream:
            if not first_chunk:
                logger.info("LLM First chunk received.")
                first_chunk = True

            full_response += part["message"]["content"]
            current_characters += part["message"]["content"]

            sentence, current_characters = handle_sentence_endings(current_characters)
            if sentence:
                if not first_sentence:
                    logger.info("LLM First sentence sent.")
                    first_sentence = True

                # here we already send first sentence to voice generation module
                yield sentence

        logger.info("LLM Finished streaming.")
        if TOOLS_ENABLED:
            tool_response = detect_and_handle_tool_call(full_response)
            if tool_response:
                full_response = tool_response
                yield tool_response
        update_conversation_history(input_text, full_response)

    except Exception as e:
        print(f"LLM Error: {e}")


async def main():
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("JARVIS: Shutting down. Goodbye!")
            break

        async for sentence in chat_with_jarvis(user_input):
            print(
                f"JARVIS: {sentence}",
            )


if __name__ == "__main__":
    asyncio.run(main())
