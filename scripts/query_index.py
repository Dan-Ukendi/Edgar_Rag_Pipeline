"""Manually query the index and eyeball the retrieved chunks.

Usage:
    uv run scripts/query_index.py "What are Apple's main risk factors?" [TICKER] [TOP_K]
"""

import sys

from tenk_rag.config import PipelineConfig
from tenk_rag.embeddings.local import SentenceTransformerEmbedder
from tenk_rag.store.qdrant_store import QdrantStore


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: uv run scripts/query_index.py "<question>" [TICKER] [TOP_K]')
    question = sys.argv[1]
    ticker = sys.argv[2] if len(sys.argv) > 2 else None
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    config = PipelineConfig.from_yaml("configs/pipeline.yaml")
    embedder = SentenceTransformerEmbedder(config.embedding.model_name, config.embedding.batch_size)
    store = QdrantStore(
        collection_name=config.vector_store.collection_name,
        dimension=embedder.dimension,
        mode=config.vector_store.mode,
        url=config.vector_store.url,
        local_path=config.vector_store.local_path,
    )

    vector = embedder.embed([question])[0]
    results = store.query(vector, top_k=top_k, ticker=ticker)

    print(f"\nQuery: {question}")
    if ticker:
        print(f"Filtered to ticker: {ticker}")
    print()
    for i, r in enumerate(results, 1):
        p = r.payload
        print(f"--- #{i} | score={r.score:.4f} | {p['ticker']} | {p['filing_date']} | {p['item_section']} ---")
        print(p["text"][:400].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
