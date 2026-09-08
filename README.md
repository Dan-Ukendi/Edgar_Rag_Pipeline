# Edgar_Rag_Pipeline

Benchmarked RAG pipeline for SEC 10-K filings — configurable chunking/embedding/LLM backends, containerized with Docker, evaluated on retrieval precision, faithfulness, and cost/latency trade-offs.

## Status

Phase 1 (core retrieval pipeline) complete. See project plan for the full phase breakdown.

**Known corpus/parsing characteristics** (found during manual testing, worth remembering when writing eval questions in Phase 5):
- Item-section tags on chunks are a best-effort heuristic (nearest preceding "Item N" heading). Some filings place their full financial statements or extended risk discussion in an appendix after their last numbered Item, so those chunks inherit that Item's label rather than the semantically obvious one (e.g. financial statements tagged "Item 16" in one filing). Retrieved *content* is still correct in these cases — only the section label is approximate.
- Items 11-14 (executive compensation, ownership, related-party transactions, accountant fees) are typically just "incorporated by reference to the Proxy Statement" boilerplate with no real data — a 10-K corpus limitation, not a pipeline bug. Avoid eval questions targeting those topics.

## Corpus

Most recent 10-K filing for 6 companies spanning distinct sectors, sourced from SEC EDGAR:

| Ticker | Company | Sector |
|---|---|---|
| AAPL | Apple | Tech / Consumer Electronics |
| JPM | JPMorgan Chase | Financial Services |
| XOM | ExxonMobil | Energy |
| JNJ | Johnson & Johnson | Healthcare / Pharma |
| WMT | Walmart | Retail |
| TSLA | Tesla | Auto / Tech |

## Setup

```bash
uv sync
uv run scripts/fetch_filings.py           # download the 6 filings from SEC EDGAR
docker run -d -p 6333:6333 -p 6334:6334 \
  -v tenk_rag_qdrant_storage:/qdrant/storage qdrant/qdrant   # start the vector store
uv run scripts/build_index.py             # parse -> chunk -> embed -> index
uv run scripts/query_index.py "What are Apple's main risk factors?" AAPL 3
```

## Structure

```
src/tenk_rag/
├── config.py           # pydantic config models, loaded from configs/pipeline.yaml
├── ingest/
│   ├── parser.py         # HTML -> plain text + Item-section tags
│   └── chunker.py         # tagged text -> overlapping Chunk objects
├── embeddings/
│   ├── base.py             # EmbeddingModel interface
│   └── local.py             # sentence-transformers backend
└── store/
    └── qdrant_store.py       # Qdrant collection management, upsert, query

configs/pipeline.yaml     # chunk size/overlap, embedding model, vector store settings
scripts/
├── fetch_filings.py       # download filings from SEC EDGAR
├── build_index.py          # run the full ingest pipeline
└── query_index.py          # manually query the index and inspect results
data/raw/                 # downloaded filings (gitignored, reproducible via fetch script)
data/processed/           # generated artifacts (gitignored)
tests/                    # test suite
```
