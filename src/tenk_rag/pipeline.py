from pathlib import Path

from .config import PipelineConfig
from .embeddings.local import SentenceTransformerEmbedder
from .ingest.chunker import build_chunks
from .ingest.parser import parse_filing
from .store.qdrant_store import QdrantStore


def _filename_metadata(path: Path) -> tuple[str, str]:
    """Filenames are TICKER_YYYY-MM-DD_10K.<ext> (see scripts/fetch_filings.py)."""
    parts = path.stem.split("_")
    ticker = parts[0]
    filing_date = parts[1] if len(parts) > 1 else ""
    return ticker, filing_date


def run_pipeline(config: PipelineConfig) -> None:
    raw_dir = Path(config.raw_data_dir)
    files = sorted(raw_dir.glob("*.htm")) + sorted(raw_dir.glob("*.html"))
    if not files:
        raise FileNotFoundError(f"No filings found in {raw_dir}. Run scripts/fetch_filings.py first.")

    embedder = SentenceTransformerEmbedder(config.embedding.model_name, config.embedding.batch_size)
    store = QdrantStore(
        collection_name=config.vector_store.collection_name,
        dimension=embedder.dimension,
        mode=config.vector_store.mode,
        url=config.vector_store.url,
        local_path=config.vector_store.local_path,
    )
    store.recreate_collection()

    total_chunks = 0
    for path in files:
        ticker, filing_date = _filename_metadata(path)
        text, item_offsets = parse_filing(path)
        chunks = build_chunks(
            text=text,
            item_offsets=item_offsets,
            ticker=ticker,
            filing_date=filing_date,
            source_file=path.name,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
        )
        vectors = embedder.embed([c.text for c in chunks])
        store.upsert(chunks, vectors)
        total_chunks += len(chunks)
        print(f"[ok] {path.name}: {len(chunks)} chunks")

    print(f"Indexed {total_chunks} chunks total into collection '{config.vector_store.collection_name}'")
