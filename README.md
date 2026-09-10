# Edgar_Rag_Pipeline

Benchmarked RAG pipeline for SEC 10-K filings — configurable chunking/embedding/LLM backends, containerized with Docker, evaluated on retrieval precision, faithfulness, and cost/latency trade-offs.

## Status

Phase 3 (service layer) complete. See project plan for the full phase breakdown.

**Known corpus/parsing characteristics** (found during manual testing, worth remembering when writing eval questions in Phase 5):
- Item-section tags on chunks are a best-effort heuristic (nearest preceding "Item N" heading). Some filings place their full financial statements or extended risk discussion in an appendix after their last numbered Item, so those chunks inherit that Item's label rather than the semantically obvious one (e.g. financial statements tagged "Item 16" in one filing). Retrieved *content* is still correct in these cases — only the section label is approximate.
- Items 11-14 (executive compensation, ownership, related-party transactions, accountant fees) are typically just "incorporated by reference to the Proxy Statement" boilerplate with no real data — a 10-K corpus limitation, not a pipeline bug. Avoid eval questions targeting those topics.
- Some open-weight models (observed on Groq's `openai/gpt-oss-120b`) will use CJK-style brackets (`【1】`) instead of ASCII `[1]` for citations despite explicit instructions. The prompt now spells out "plain ASCII square brackets" with an example, and citation parsing accepts both bracket styles as a safety net.
- Streamlit's markdown renderer treats text between two `$` signs as LaTeX math, which mangles answers/excerpts containing two or more dollar amounts. Fixed by escaping `$` before display in `ui/app.py` — doesn't affect the CLI or raw API JSON, which were never mangled.
- Comparison questions can hit `top_k_per_company`'s shallow depth (default 2) — e.g. asking to compare two companies' revenue can retrieve one company's income-statement chunk but not the other's, and the model correctly says the figure isn't in the excerpts rather than guessing. A worthwhile Phase 6 sweep target if comparison quality matters.

## Multi-company questions

Leaving the `ticker` filter unset doesn't just mean "search across everyone's chunks" — `RagService.ask()` first runs deterministic company detection (`companies.py`, a small alias table, no extra LLM call) on the question text itself, then picks one of three modes:
- **No company named** → `bullets` mode across all 6 companies (Qdrant's `query_points_groups`, grouped by ticker, `top_k_per_company` chunks each so every company is represented) — one bullet per company, each citing only its own excerpts.
- **1+ companies named, no comparison language detected** → same `bullets` structure, but retrieval and the answer are restricted to just the named companies (so "What is Apple's revenue?" only ever pulls and discusses Apple).
- **2+ companies named *and* comparison language detected** (`compare`, `versus`, `which is higher`, etc.) → `comparison` mode: a single unified answer that directly compares the named companies, not independent parallel bullets.

An explicit `ticker` from the API/UI always overrides detection entirely (unchanged `single` mode, normal prose). The mode actually used is returned in every response (`mode` field / CLI output) for transparency.

## LLM backends

Two interchangeable backends behind one `LLMClient` interface, selectable via `llm.backend` in `configs/pipeline.yaml` or `--backend=groq|local` on `scripts/ask.py`:
- **groq** (hosted, free tier) — `openai/gpt-oss-120b` via the Groq API. Needs a free `GROQ_API_KEY` from console.groq.com in a local `.env` file (gitignored).
- **local** — Ollama running in Docker, `llama3.2:3b`. No API key, fully offline, but noticeably slower on CPU and less consistent about citation formatting than the hosted model.

## API + UI

FastAPI service (`POST /query`, `GET /health`) backed by a shared `RagService` (also used by `scripts/ask.py`, so the CLI and API never drift), plus a minimal Streamlit demo UI.

```bash
uv run uvicorn tenk_rag.api.app:app --reload --port 8000   # API; Swagger docs at /docs
uv run streamlit run ui/app.py                               # UI at :8501, talks to the API over HTTP
```

`GET /health` does a real dependency check: Qdrant reachability + point count, and each LLM backend's actual availability (Groq via API-key presence, Ollama via a live ping to its `/api/tags` endpoint — no generation call, so health checks stay free and fast).

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

# LLM backends
echo "GROQ_API_KEY=your-key-here" > .env       # free key from console.groq.com
docker run -d -p 11434:11434 \
  -v tenk_rag_ollama_data:/root/.ollama ollama/ollama   # local backend (optional)
docker exec <ollama-container-name> ollama pull llama3.2:3b

uv run scripts/ask.py "What are Apple's main risk factors?" AAPL --backend=groq
uv run scripts/ask.py "What are Apple's main risk factors?" AAPL --backend=local
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
├── store/
│   └── qdrant_store.py       # Qdrant collection management, upsert, query, health info
├── llm/
│   ├── base.py                # LLMClient interface
│   ├── groq_client.py          # hosted (free tier) backend
│   └── ollama_client.py         # local backend (+ is_reachable() for health checks)
├── generation/
│   ├── prompt.py                # grounded-answer prompt (forces inline citations)
│   └── answer.py                  # orchestration + citation parsing
├── service.py             # RagService: shared orchestration used by both the CLI and the API
└── api/
    ├── app.py               # FastAPI app, lifespan builds RagService once, CORS enabled
    ├── schemas.py             # request/response pydantic models
    └── routes.py               # POST /query, GET /health

ui/app.py                 # Streamlit demo UI

configs/pipeline.yaml     # chunk size/overlap, embedding model, vector store, LLM settings
scripts/
├── fetch_filings.py       # download filings from SEC EDGAR
├── build_index.py          # run the full ingest pipeline
├── query_index.py           # manually query the index and inspect results
└── ask.py                    # end-to-end: retrieve -> generate a grounded, cited answer
data/raw/                 # downloaded filings (gitignored, reproducible via fetch script)
data/processed/           # generated artifacts (gitignored)
tests/                    # test suite
```
