"""Ask a question end-to-end: retrieve chunks, generate a grounded, cited answer.

Usage:
    uv run scripts/ask.py "What are Apple's main risk factors?" [TICKER] [--backend groq|local]
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from tenk_rag.config import PipelineConfig
from tenk_rag.embeddings.local import SentenceTransformerEmbedder
from tenk_rag.generation.answer import answer_question
from tenk_rag.llm.factory import build_llm_client
from tenk_rag.store.qdrant_store import QdrantStore


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit('Usage: uv run scripts/ask.py "<question>" [TICKER] [--backend groq|local]')
    question = args[0]
    ticker = args[1] if len(args) > 1 else None

    backend_override = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--backend=")), None)

    config = PipelineConfig.from_yaml("configs/pipeline.yaml")
    if backend_override:
        config.llm.backend = backend_override

    embedder = SentenceTransformerEmbedder(config.embedding.model_name, config.embedding.batch_size)
    store = QdrantStore(
        collection_name=config.vector_store.collection_name,
        dimension=embedder.dimension,
        mode=config.vector_store.mode,
        url=config.vector_store.url,
        local_path=config.vector_store.local_path,
    )

    query_vector = embedder.embed([question])[0]
    results = store.query(query_vector, top_k=config.retrieval.top_k, ticker=ticker)
    excerpts = [
        {
            "ticker": r.payload["ticker"],
            "filing_date": r.payload["filing_date"],
            "item_section": r.payload["item_section"],
            "text": r.payload["text"],
        }
        for r in results
    ]

    llm_client = build_llm_client(config.llm)
    result = answer_question(question, excerpts, llm_client, backend=config.llm.backend)

    print(f"\nQuestion: {result.question}")
    print(f"Backend: {result.backend}\n")
    print(result.answer)
    print(f"\nCited excerpts: {result.cited_indices or 'none'}")
    print("\nRetrieved excerpts:")
    for i, ex in enumerate(excerpts, 1):
        cited = "cited" if i in result.cited_indices else "not cited"
        print(f"  [{i}] {ex['ticker']} | {ex['filing_date']} | {ex['item_section']} | {cited}")


if __name__ == "__main__":
    main()
