import asyncio
import ollama
import time
import re
import ast
import statistics

MODELS = [
    "granite4:350m",
    "deepseek-r1:1.5b",
    "qwen3:1.7b",
    "qwen3:0.6b",
    "gemma3:1b",
    "llama3.2:3b",
    "llama3.2:1b",
]

REPEATS_PER_TEST = 3
STREAM = True


QA_TESTS = [
    # Simple factual
    ("What is the capital of France?", "paris"),
    ("Who wrote Romeo and Juliet?", "shakespeare"),
    ("What planet is known as the Red Planet?", "mars"),
    # Basic reasoning
    ("If you have 3 apples and buy 2 more, how many apples do you have?", "5"),
    ("What is 12 divided by 3?", "4"),
    # Multi-step reasoning
    (
        "If John is taller than Mike and Mike is taller than Sam, who is the tallest?",
        "john",
    ),
    ("A train leaves at 3pm and arrives at 5pm. How long was the journey?", "2"),
    # Small world knowledge
    ("What gas do plants absorb from the atmosphere?", "carbon dioxide"),
]

INSTRUCTION_TESTS = [
    # Exact word
    ("Reply with exactly one word: cat", "cat"),
    # Exact number
    ("Say only the number 42", "42"),
    # Case sensitivity
    ("Reply with exactly: HELLO", "HELLO"),
    # No explanation
    ("Respond with only the word 'blue' and nothing else.", "blue"),
    # Character constraint
    ("Respond with exactly three characters: yes", "yes"),
    # No punctuation
    ("Reply with the word apple without punctuation.", "apple"),
]

STRUCTURED_TESTS = [
    (
        'Respond ONLY with valid JSON: {"name": "John", "age": 30}',
        {"name": "John", "age": 30},
    ),
    (
        'Respond ONLY with valid JSON: {"numbers": [1, 2, 3]}',
        {"numbers": [1, 2, 3]},
    ),
    (
        'Respond ONLY with valid JSON: {"person": {"name": "Alice", "city": "Paris"}}',
        {"person": {"name": "Alice", "city": "Paris"}},
    ),
    (
        'Respond ONLY with valid JSON: {"valid": true}',
        {"valid": True},
    ),
    (
        'Respond ONLY with valid JSON: {"a": 1, "b": 2, "c": 3}',
        {"a": 1, "b": 2, "c": 3},
    ),
]

TOOL_TESTS = [
    (
        "What's the weather in London?",
        "get_weather_report",
    ),
    (
        "Play song Imagine",
        "play_song",
    ),
]


async def warmup_model(model):
    print(f"Warming up {model}...")
    await run_single_prompt(model, "Hello.")
    await asyncio.sleep(0.5)  # allow GPU settle


def extract_tool_call(text):
    pattern = r"\[(\w+)\((.*?)\)\]"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def try_parse_json(text):
    try:
        return ast.literal_eval(text)
    except:
        return None


async def run_single_prompt(model: str, prompt: str):
    start_time = time.perf_counter()

    ttft = None
    full_response = ""

    stream = ollama.chat(
        model,
        messages=[{"role": "user", "content": prompt}],
        stream=STREAM,
    )

    first_sentence_time = None
    sentence_buffer = ""

    for chunk in stream:
        now = time.perf_counter()

        if ttft is None:
            ttft = now - start_time

        content = chunk["message"]["content"]
        full_response += content
        sentence_buffer += content

        if first_sentence_time is None:
            if re.search(r"[.!?]", sentence_buffer):
                first_sentence_time = now - start_time

    total_time = time.perf_counter() - start_time

    tokens_estimate = len(full_response.split())
    tps = tokens_estimate / total_time if total_time > 0 else 0

    return {
        "ttft": ttft,
        "first_sentence": first_sentence_time,
        "total_time": total_time,
        "tps": tps,
        "response": full_response.strip(),
    }


async def run_qa_tests(model):
    correct = 0
    for prompt, answer in QA_TESTS:
        result = await run_single_prompt(model, prompt)
        if answer.lower() in result["response"].lower():
            correct += 1
    return correct / len(QA_TESTS)


async def run_instruction_tests(model):
    correct = 0
    for prompt, answer in INSTRUCTION_TESTS:
        result = await run_single_prompt(model, prompt)
        if result["response"].strip().lower() == answer.lower():
            correct += 1
    return correct / len(INSTRUCTION_TESTS)


async def run_structured_tests(model):
    correct = 0
    for prompt, expected in STRUCTURED_TESTS:
        result = await run_single_prompt(model, prompt)
        parsed = try_parse_json(result["response"])
        if parsed == expected:
            correct += 1
    return correct / len(STRUCTURED_TESTS)


async def run_tool_tests(model):
    correct = 0
    for prompt, expected_tool in TOOL_TESTS:
        result = await run_single_prompt(model, prompt)
        tool_called = extract_tool_call(result["response"])
        if tool_called == expected_tool:
            correct += 1
    return correct / len(TOOL_TESTS)


async def benchmark_model(model: str):
    print(f"\n=== Benchmarking {model} ===")

    await warmup_model(model)

    latencies = []
    ttfts = []
    tps_list = []

    for _ in range(REPEATS_PER_TEST):
        result = await run_single_prompt(model, "Say hello.")
        latencies.append(result["total_time"])
        ttfts.append(result["ttft"])
        tps_list.append(result["tps"])

    qa_score = await run_qa_tests(model)
    instr_score = await run_instruction_tests(model)
    struct_score = await run_structured_tests(model)
    # tool_score = await run_tool_tests(model)

    return {
        "model": model,
        "avg_latency": statistics.mean(latencies),
        "avg_ttft": statistics.mean(ttfts),
        "avg_tps": statistics.mean(tps_list),
        "qa_accuracy": qa_score,
        "instruction_accuracy": instr_score,
        "structured_accuracy": struct_score,
        # "tool_accuracy": tool_score,
    }


async def main():
    results = []

    for model in MODELS:
        result = await benchmark_model(model)
        results.append(result)

    # Print results nicely
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 80)

    sorted_results = sorted(results, key=lambda x: (x["avg_ttft"], -x["qa_accuracy"]))

    header = (
        f"{'Rank':<5}"
        f"{'Model':<18}"
        f"{'TTFT (ms)':>12}"
        f"{'Latency (ms)':>14}"
        f"{'Tok/s':>10}"
        f"{'QA %':>8}"
        f"{'Instr %':>10}"
        f"{'Struct %':>10}"
        # f"{'Tool %':>8}"
    )

    print(header)
    print("-" * len(header))

    for idx, r in enumerate(sorted_results, start=1):
        print(
            f"{idx:<5}"
            f"{r['model']:<18}"
            f"{r['avg_ttft'] * 1000:>12.0f}"
            f"{r['avg_latency'] * 1000:>14.0f}"
            f"{r['avg_tps']:>10.1f}"
            f"{r['qa_accuracy'] * 100:>8.0f}"
            f"{r['instruction_accuracy'] * 100:>10.0f}"
            f"{r['structured_accuracy'] * 100:>10.0f}"
            # f"{r['tool_accuracy'] * 100:>8.0f}"
        )

    print("=" * 80)

    # Highlight winner
    winner = sorted_results[0]
    print(
        f"\n🏆 Fastest Model: {winner['model']} "
        f"(TTFT: {winner['avg_ttft']*1000:.0f} ms)"
    )


if __name__ == "__main__":
    asyncio.run(main())
