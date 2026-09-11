"""LLM-judge grading for groundedness, relevance, and correctness.

Uses a Groq model distinct from both app backends (openai/gpt-oss-120b and
llama3.2:3b) so the judge never grades its own or a same-provider sibling's
output on the exact model under test.
"""

import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

JUDGE_MODEL = "qwen/qwen3.8-27b"

SYSTEM_PROMPT = (
    "You are grading a RAG system's answer to a question about SEC 10-K filings. "
    "You will be given the question, the excerpts the system retrieved and cited, "
    "the system's generated answer, and a human-verified gold answer. Grade three "
    "independent properties and respond with ONLY a JSON object matching this "
    "schema: "
    '{"faithful": bool, "relevant": bool, "correct": bool, "reasoning": string}. '
    "faithful: every factual claim in the answer is actually supported by the "
    "retrieved excerpts (not fabricated, not from outside knowledge). "
    "relevant: the answer actually addresses what the question asked, rather than "
    "talking about something else. "
    "correct: the answer's content matches the gold answer's key facts, allowing "
    "for paraphrasing, formatting, and rounding differences -- for an unanswerable "
    "question (gold answer starts with 'Not disclosed'), correct means the system "
    "appropriately declined to answer or stated the information isn't available, "
    "rather than fabricating a figure. "
    "reasoning: one or two sentences justifying the three verdicts."
)


def build_user_prompt(question: str, excerpts: list[dict], answer: str, gold_answer: str) -> str:
    lines = ["Question:", question, "", "Retrieved excerpts:"]
    for i, ex in enumerate(excerpts, 1):
        lines.append(f"[{i}] ({ex['ticker']}, {ex['item_section']}) {ex['text']}")
    lines += ["", "System's answer:", answer, "", "Gold answer:", gold_answer]
    return "\n".join(lines)


def judge_answer(question: str, excerpts: list[dict], answer: str, gold_answer: str) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set (checked environment and .env file)")
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, excerpts, answer, gold_answer)},
        ],
        response_format={"type": "json_object"},
        max_tokens=1024,
    )
    return json.loads(response.choices[0].message.content)
