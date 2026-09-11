"""Parses eval_set.json's human-written `expected_source` field into
machine-checkable (ticker, item_section) pairs for retrieval scoring.

Ground truth is deliberately (ticker, item_section) rather than exact chunk
IDs, because chunk boundaries shift under Phase 6's chunking sweeps -- an
ID-pinned ground truth would silently break on every config except today's.
"""

import re

TICKERS = ["AAPL", "TSLA", "WMT", "JPM", "XOM", "JNJ"]
_TICKER_ALT = "|".join(TICKERS)

# "Item 7", "Item 1A", "Item 1/2" (-> Item 1 and Item 2)
_ITEM_PATTERN = re.compile(r"Item\s+(\d{1,2})([A-Za-z]?)(?:/(\d{1,2}))?")
_MULTI_ITEM_PATTERN = re.compile(rf"({_TICKER_ALT})\s+Item\s+(\d{{1,2}})([A-Za-z]?)")


def parse_expected_source(doc: str, expected_source: str) -> list[tuple[str, str]]:
    """Returns [(ticker, item_label), ...], e.g. [("AAPL", "Item 7"), ("AAPL", "Item 8")].
    Empty list means expected_source doesn't name specific items (e.g. prose
    like "Human capital sections of all six filings", or an unanswerable
    question) -- retrieval scoring should be skipped for those, not failed."""
    pairs: list[tuple[str, str]] = []

    if doc == "MULTI":
        for m in _MULTI_ITEM_PATTERN.finditer(expected_source):
            ticker, num, suffix = m.group(1), m.group(2), m.group(3)
            pairs.append((ticker, f"Item {num}{suffix.upper()}"))
        return pairs

    ticker = doc.split("_")[0]
    for m in _ITEM_PATTERN.finditer(expected_source):
        num, suffix, alt_num = m.group(1), m.group(2), m.group(3)
        pairs.append((ticker, f"Item {num}{suffix.upper()}"))
        if alt_num:
            pairs.append((ticker, f"Item {alt_num}"))
    return pairs
