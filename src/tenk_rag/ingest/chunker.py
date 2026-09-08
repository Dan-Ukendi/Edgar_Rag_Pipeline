"""Split parsed filing text into overlapping, Item-tagged chunks."""

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    ticker: str
    filing_date: str
    source_file: str
    item_section: str | None
    chunk_index: int
    char_start: int
    char_end: int
    text: str


def _find_item_section(offset: int, item_offsets: list[tuple[int, str]]) -> str | None:
    section = None
    for item_offset, label in item_offsets:
        if item_offset <= offset:
            section = label
        else:
            break
    return section


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int, str]]:
    """Sliding-window split that snaps boundaries to the nearest paragraph
    or sentence break within the window, to avoid cutting mid-sentence."""
    spans = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            boundary = text.rfind("\n\n", start, end)
            if boundary == -1 or boundary <= start + chunk_size // 2:
                boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk_str = text[start:end].strip()
        if chunk_str:
            spans.append((start, end, chunk_str))
        if end >= n:
            break
        start = max(end - chunk_overlap, start + 1)
    return spans


def build_chunks(
    text: str,
    item_offsets: list[tuple[int, str]],
    ticker: str,
    filing_date: str,
    source_file: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks = []
    for i, (start, end, chunk_text) in enumerate(_split_text(text, chunk_size, chunk_overlap)):
        chunks.append(
            Chunk(
                chunk_id=f"{ticker}:{source_file}:{i}",
                ticker=ticker,
                filing_date=filing_date,
                source_file=source_file,
                item_section=_find_item_section(start, item_offsets),
                chunk_index=i,
                char_start=start,
                char_end=end,
                text=chunk_text,
            )
        )
    return chunks
