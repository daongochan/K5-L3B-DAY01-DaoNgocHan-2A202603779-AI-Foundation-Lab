"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to src/optional_llm_api/solution/solution.py when done.
    4. Run from the repo root: pytest tests/test_solution.py -v

Models (11/2026): gpt-4o / gpt-4o-mini are removed from the OpenAI API on 2027-01-14
(https://developers.openai.com/api/docs/deprecations), so this lab uses gpt-6.1-terra / gpt-6-luna.
The result keys and table headers below still say "gpt4o" / "GPT-4o" / "Mini": they are a test
contract kept for compatibility. Read "gpt4o" as "the large model" and "mini" as "the small model".
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1K OUTPUT tokens (USD) — OpenAI pricing page, 11/2026. Update if pricing changes.
# ---------------------------------------------------------------------------
COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-6.1-terra": 0.005,   # $5.00 / 1M output tokens
    "gpt-6-luna": 0.0005,     # $0.50 / 1M output tokens
}

OPENAI_MODEL = "gpt-6.1-terra"
OPENAI_MINI_MODEL = "gpt-6-luna"


# ---------------------------------------------------------------------------
# Task 1 — Call the large model (OPENAI_MODEL)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API and return the response text + latency.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="none",
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_tokens,
    )
    latency = max(time.perf_counter() - t0, 1e-9)
    text = resp.choices[0].message.content or ""
    return text, float(latency)


# ---------------------------------------------------------------------------
# Task 2 — Call the small model (OPENAI_MINI_MODEL)
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API using OPENAI_MINI_MODEL and return the
    response text + latency.
    """
    return call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Task 3 — Compare the large vs the small model
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call both OPENAI_MODEL and OPENAI_MINI_MODEL with the same prompt and return a
    comparison dictionary.
    """
    gpt4o_resp, gpt4o_lat = call_openai(prompt)
    mini_resp, mini_lat = call_openai_mini(prompt)
    cost_est = (len(gpt4o_resp.split()) / 0.75) / 1000.0 * COST_PER_1K_OUTPUT_TOKENS[OPENAI_MODEL]
    return {
        "gpt4o_response": gpt4o_resp,
        "mini_response": mini_resp,
        "gpt4o_latency": float(gpt4o_lat),
        "mini_latency": float(mini_lat),
        "gpt4o_cost_estimate": float(cost_est),
    }


# ---------------------------------------------------------------------------
# Task 4 — Streaming chatbot with conversation history
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history: list[dict[str, str]] = []

    while True:
        user_msg = input("You: ").strip()
        if user_msg.lower() in ("quit", "exit"):
            break
        if not user_msg:
            continue

        history.append({"role": "user", "content": user_msg})
        stream = client.chat.completions.create(
            model=OPENAI_MINI_MODEL,
            messages=history,
            reasoning_effort="none",
            stream=True,
        )

        reply_chunks: list[str] = []
        print("Assistant: ", end="", flush=True)
        for chunk in stream:
            delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            print(delta, end="", flush=True)
            reply_chunks.append(delta)
        print()

        history.append({"role": "assistant", "content": "".join(reply_chunks)})
        history = history[-6:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (base_delay * 2^attempt).
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.
    """
    results: list[dict] = []
    for p in prompts:
        item = dict(compare_models(p))
        item["prompt"] = p
        results.append(item)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of compare_models results as a readable text table.
    """
    def _trunc(s: str, width: int = 40) -> str:
        s_clean = " ".join(str(s).splitlines()).strip()
        if len(s_clean) <= width:
            return s_clean
        return s_clean[: width - 3] + "..."

    headers = ["Prompt", "GPT-4o Response", "Mini Response", "GPT-4o Latency", "Mini Latency"]
    widths = [40, 40, 40, 14, 12]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep_line = "-+-".join("-" * w for w in widths)
    rows = [header_line, sep_line]

    for r in results:
        cols = [
            _trunc(r.get("prompt", ""), 40).ljust(40),
            _trunc(r.get("gpt4o_response", ""), 40).ljust(40),
            _trunc(r.get("mini_response", ""), 40).ljust(40),
            f"{float(r.get('gpt4o_latency', 0.0)):.3f}s".ljust(14),
            f"{float(r.get('mini_latency', 0.0)):.3f}s".ljust(12),
        ]
        rows.append(" | ".join(cols))

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_prompt = "Explain the difference between temperature and top_p in one sentence."
    print("=== Comparing models ===")
    result = compare_models(test_prompt)
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Starting chatbot (type 'quit' to exit) ===")
    streaming_chatbot()

