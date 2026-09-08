"""Parse a raw 10-K HTML filing into plain text tagged with SEC 'Item' sections.

Section tagging is a best-effort heuristic: 10-K HTML formatting is
inconsistent across filers, and every filing mentions each Item multiple
times outside its real heading -- once in the table of contents, and
sometimes again later as a prose cross-reference (e.g. "as discussed in
Item 1A, ..."). A real Item heading sequence runs in strictly increasing
order through the document body (1, 1A, 1B, 2, 3, ... 16); the ToC forms
its own increasing run before it, and stray cross-references break the
order rather than extending it. So we group all matches into maximal
runs of increasing item order and keep the longest one -- ties broken
toward the later run, since real content always follows the ToC.
"""

import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ITEM_PATTERN = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)\.?\s*[-–:]?\s*(.*)$", re.IGNORECASE)

ITEM_ORDER = [
    "1", "1A", "1B", "1C", "2", "3", "4", "5", "6",
    "7", "7A", "8", "9", "9A", "9B", "9C",
    "10", "11", "12", "13", "14", "15", "16",
]
ITEM_RANK = {f"Item {label}": rank for rank, label in enumerate(ITEM_ORDER)}


def _select_real_headings(matches: list[tuple[int, str]]) -> list[tuple[int, str]]:
    runs: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    last_rank = -1
    for offset, label in matches:
        rank = ITEM_RANK[label]
        if not current or rank > last_rank:
            current.append((offset, label))
        else:
            runs.append(current)
            current = [(offset, label)]
        last_rank = rank
    if current:
        runs.append(current)

    if not runs:
        return []

    max_len = max(len(run) for run in runs)
    best_run = next(run for run in reversed(runs) if len(run) == max_len)
    return best_run


def parse_filing(html_path: Path) -> tuple[str, list[tuple[int, str]]]:
    """Return (plain_text, item_offsets) where item_offsets is a list of
    (char_offset, item_label) sorted by offset, one entry per Item heading."""
    raw_html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw_html, "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    matches: list[tuple[int, str]] = []
    offset = 0
    for line in text.split("\n"):
        match = ITEM_PATTERN.match(line.strip())
        if match:
            label = f"Item {match.group(1).upper()}"
            if label in ITEM_RANK:
                matches.append((offset, label))
        offset += len(line) + 1

    item_offsets = _select_real_headings(matches)
    return text, item_offsets
