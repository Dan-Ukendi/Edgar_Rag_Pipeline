"""Detects which of the 6 known companies (and comparison intent) a question
is about, so RagService can decide retrieval scope and answer structure
without an extra LLM call. Deterministic and reproducible, which matters for
Phase 5 eval design.

Aliases deliberately exclude ambiguous single words that could false-positive
on ordinary language (e.g. "Chase" for JPMorgan Chase).
"""

import re

COMPANY_ALIASES: dict[str, list[str]] = {
    "AAPL": ["AAPL", "Apple"],
    "JPM": ["JPM", "JPMorgan Chase", "JPMorgan", "JP Morgan"],
    "XOM": ["XOM", "ExxonMobil", "Exxon Mobil", "Exxon"],
    "JNJ": ["JNJ", "Johnson & Johnson", "Johnson and Johnson", "J&J"],
    "WMT": ["WMT", "Walmart", "Wal-Mart"],
    "TSLA": ["TSLA", "Tesla"],
}

COMPARISON_KEYWORDS = [
    "compare",
    "comparison",
    "versus",
    " vs ",
    " vs.",
    "difference between",
    "differ",
    "which company",
    "which one",
    "which has",
    "which is higher",
    "which is lower",
    "higher than",
    "lower than",
    "more than",
    "less than",
    "better than",
    "worse than",
    "relative to",
    "compared to",
    "compared with",
]


def detect_companies(question: str) -> list[str]:
    """Return the tickers explicitly named in the question, in the order they
    first appear, deduplicated. Empty list means no specific company was named."""
    matches: list[tuple[int, str]] = []
    for ticker, aliases in COMPANY_ALIASES.items():
        for alias in aliases:
            match = re.search(rf"\b{re.escape(alias)}\b", question, re.IGNORECASE)
            if match:
                matches.append((match.start(), ticker))
                break

    matches.sort(key=lambda m: m[0])
    seen: set[str] = set()
    ordered: list[str] = []
    for _, ticker in matches:
        if ticker not in seen:
            seen.add(ticker)
            ordered.append(ticker)
    return ordered


def detect_comparison_intent(question: str) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in COMPARISON_KEYWORDS)
