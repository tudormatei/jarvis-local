import asyncio
import re
import ollama
import logging

logger = logging.getLogger(__name__)

# Initialize Ollama model
MODEL_NAME = "jarvis:3b"

# Conversation memory settings
conversation_history = []
MAX_INTERACTIONS = 5


def update_conversation_history(user_input, assistant_response):
    # Append user input and assistant response to the conversation history
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append(
        {"role": "assistant", "content": assistant_response})

    # Keep only the last N interactions (where each interaction is a user-assistant pair)
    while len(conversation_history) > MAX_INTERACTIONS * 2:
        # Remove the oldest exchange (both user and assistant)
        conversation_history.pop(0)


def handle_sentence_endings(full_response):
    # Regex to detect a sentence ending with ., !, or ? — possibly followed by quotes or brackets
    sentence_endings_pattern = r'^(.*?[.!?]["\')\]]?)(?=\s|$)'

    # Find the first match of a complete sentence
    match = re.match(sentence_endings_pattern, full_response)

    if match:
        sentence = match.group(1).strip()
        remaining_text = full_response[len(match.group(0)):].strip()
    else:
        sentence = None
        remaining_text = full_response.strip()

    return sentence, remaining_text


async def chat_with_jarvis(input_text, is_streaming=True):
    """Handles streaming responses efficiently."""
    messages = conversation_history + [{"role": "user", "content": input_text}]
    current_characters = ""
    full_response = ""

    try:
        logger.info("LLM Started inference.")
        response_stream = ollama.chat(
            MODEL_NAME, messages=messages, stream=True)

        first_chunk = False
        first_sentence = False
        for part in response_stream:
            if not first_chunk:
                logger.info("LLM First chunk received.")
                first_chunk = True

            full_response += part["message"]["content"]
            current_characters += part["message"]["content"]

            if is_streaming:
                sentence, current_characters = handle_sentence_endings(
                    current_characters)
                if sentence:
                    if not first_sentence:
                        logger.info("LLM First sentence sent.")
                        first_sentence = True

                    # Here we could actually send this to voice generation module
                    yield sentence
        logger.info("LLM Finished streaming.")
        if not is_streaming:
            yield full_response
        update_conversation_history(input_text, full_response)

    except Exception as e:
        print(f"Error: {e}")


async def main():
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("JARVIS: Shutting down. Goodbye!")
            break

        async for sentence in chat_with_jarvis(user_input):
            print(
                f"{sentence}",)

if __name__ == "__main__":
    asyncio.run(main())
