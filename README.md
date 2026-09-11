# Edgar_Rag_Pipeline

Benchmarked RAG pipeline for SEC 10-K filings — configurable chunking/embedding/LLM backends, containerized with Docker, evaluated on retrieval precision, faithfulness, and cost/latency trade-offs.

## Status

Phase 5 (evaluation set) complete. See project plan for the full phase breakdown.

**Known corpus/parsing characteristics** (found during manual testing, worth remembering when writing eval questions in Phase 5):
- Item-section tags on chunks are a best-effort heuristic (nearest preceding "Item N" heading). Some filings place their full financial statements or extended risk discussion in an appendix after their last numbered Item, so those chunks inherit that Item's label rather than the semantically obvious one (e.g. financial statements tagged "Item 16" in one filing). Retrieved *content* is still correct in these cases — only the section label is approximate.
- Items 11-14 (executive compensation, ownership, related-party transactions, accountant fees) are typically just "incorporated by reference to the Proxy Statement" boilerplate with no real data — a 10-K corpus limitation, not a pipeline bug. Avoid eval questions targeting those topics.
- Some open-weight models (observed on Groq's `openai/gpt-oss-120b`) will use CJK-style brackets (`【1】`) instead of ASCII `[1]` for citations despite explicit instructions. The prompt now spells out "plain ASCII square brackets" with an example, and citation parsing accepts both bracket styles as a safety net.
- Streamlit's markdown renderer treats text between two `$` signs as LaTeX math, which mangles answers/excerpts containing two or more dollar amounts. Fixed by escaping `$` before display in `ui/app.py` — doesn't affect the CLI or raw API JSON, which were never mangled.
- Comparison questions can hit `top_k_per_company`'s shallow depth (default 2) — e.g. asking to compare two companies' revenue can retrieve one company's income-statement chunk but not the other's, and the model correctly says the figure isn't in the excerpts rather than guessing. A worthwhile Phase 6 sweep target if comparison quality matters.
- Neither the official Qdrant nor Ollama Docker images include curl/wget, only bash — their `docker-compose.yml` healthchecks use bash's `/dev/tcp` pseudo-device for a plain TCP-port-open check instead.
- Short factual sentences (e.g. an employee headcount buried in a "Human Capital" paragraph) can rank behind dense financial-statement boilerplate in cosine similarity, even at `top_k=8+` — a real embedding/chunking limitation the eval set caught, not a retrieval-logic bug. Worth a Phase 6 sweep target (chunk size, embedding model).
- Building the eval set (Phase 5) surfaced and fixed two real bugs: company-mention detection missed "JPMorganChase" as one word (only had "JPMorgan Chase" with a space) and comparison-intent detection missed phrasings like "Rank X, Y and Z by..." or "which reported a decline" (`companies.py`). It also caught a structural regression: single-company questions asked without an explicit `ticker` were retrieving only `top_k_per_company` (2) chunks via the multi-company code path instead of the deeper `top_k` (5) a single-company query should get — fixed in `RagService.ask()`.

## Evaluation

`eval/eval_set.json` — 40 hand-written, source-verified Q&A pairs (every gold answer checked against the actual downloaded filing text) spanning all 6 companies, three question types (factual, multi-hop, cross-document comparison) plus 2 deliberately unanswerable questions. Ground truth for retrieval is `(ticker, item_section)` pairs parsed from a human-written section reference, not exact chunk IDs — chunk boundaries shift under Phase 6's chunking sweeps, so ID-pinned ground truth would silently break on every config except today's.

```bash
uv run scripts/run_eval.py --backend=groq   # or --backend=local, --limit=N for a quick subset
```

Scores three independent properties per question, using a Groq model (`qwen/qwen3.8-27b`) distinct from both app backends as judge — never self-grading:
- **Retrieval hit** (programmatic): did any retrieved chunk match the expected `(ticker, item_section)`?
- **Faithful** (LLM-judged): is every claim in the answer actually supported by what was retrieved?
- **Relevant** (LLM-judged): does the answer address what was actually asked?
- **Correct** (LLM-judged): does the answer match the gold answer's key facts (or, for unanswerable questions, correctly decline rather than fabricate)?

Baseline (Groq backend, default config): 46% retrieval hit, 95% faithful, 98% relevant, 58% correct. The gap between "faithful" and "correct" is the story: the system rarely hallucinates, but a lot of the miss rate is retrieval not surfacing the right chunk in the first place — exactly what Phase 6's sweeps exist to improve on.

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

FastAPI service (`POST /query`, `GET /health`) backed by a shared `RagService` (also used by `scripts/ask.py`, so the CLI and API never drift), plus a minimal Streamlit demo UI. See Setup above for how to run them (Docker or local).

`GET /health` does a real dependency check: Qdrant reachability + point count, and each LLM backend's actual availability (Groq via API-key presence, Ollama via a live ping to its `/api/tags` endpoint — no generation call, so health checks stay free and fast).

## Containerization

`docker-compose.yml` runs the whole system as one stack: `qdrant` and `ollama` (official images, named volumes for persistence), a one-shot `ingest` service (idempotent — fetches filings and builds the index only if they don't already exist) and `ollama-pull` (pulls `llama3.2:3b`, doesn't block the rest of the stack since the default backend is Groq), then `api` and `ui` built from a single shared `Dockerfile`. Cross-container addressing (`http://qdrant:6333`, `http://ollama:11434` instead of `localhost`) is handled by `QDRANT_URL`/`OLLAMA_BASE_URL` env-var overrides in `PipelineConfig.from_yaml()` — the same `pipeline.yaml` works unmodified in both local dev and Docker. Downloaded filings are bind-mounted to `./data` on the host rather than hidden inside a Docker volume, so they're inspectable and reused across rebuilds without a network fetch.

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

**Docker (recommended)** — one command gets a fully working, queryable stack from a clean clone:

```bash
echo "GROQ_API_KEY=your-key-here" > .env     # free key from console.groq.com
docker compose up --build
```

That single command builds the API/UI image, starts Qdrant and Ollama, automatically fetches the 6 filings and builds the vector index (skipped on future runs if already done), pulls the `llama3.2:3b` local model, then starts the API (`:8000`, Swagger docs at `/docs`) and the Streamlit UI (`:8501`). First run takes several minutes (downloading filings, embedding ~4,700 chunks, pulling the Ollama model); `docker compose down` followed by `docker compose up` again is fast, since Qdrant/Ollama data persists in named volumes and downloaded filings persist in the bind-mounted `./data`.

**Local dev (no Docker)** — for iterating on code directly:

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
uv run uvicorn tenk_rag.api.app:app --reload --port 8000
uv run streamlit run ui/app.py
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
├── companies.py           # deterministic company/comparison-intent detection
├── service.py             # RagService: shared orchestration used by both the CLI and the API
└── api/
    ├── app.py               # FastAPI app, lifespan builds RagService once, CORS enabled
    ├── schemas.py             # request/response pydantic models
    └── routes.py               # POST /query, GET /health

ui/app.py                 # Streamlit demo UI

Dockerfile                # shared image for api/ui/ingest services
docker-compose.yml        # full stack: qdrant, ollama, ingest, ollama-pull, api, ui
.dockerignore

configs/pipeline.yaml     # chunk size/overlap, embedding model, vector store, LLM settings
scripts/
├── fetch_filings.py       # download filings from SEC EDGAR
├── build_index.py          # run the full ingest pipeline
├── query_index.py           # manually query the index and inspect results
├── ask.py                    # end-to-end: retrieve -> generate a grounded, cited answer
├── docker_ingest.py            # idempotent fetch+build step run by the `ingest` compose service
└── run_eval.py                  # runs eval/eval_set.json end to end, writes eval/results.json

eval/
├── eval_set.json           # 40 hand-written, source-verified Q&A pairs (ground truth)
└── results.json            # latest run's per-question scores (generated)
src/tenk_rag/eval/
├── ground_truth.py         # parses expected_source -> (ticker, item_section) pairs
├── judge.py                  # Groq LLM-judge: faithful / relevant / correct
└── scorer.py                   # combines retrieval scoring + judge per question

data/raw/                 # downloaded filings (gitignored, reproducible via fetch script)
data/processed/           # generated artifacts (gitignored)
tests/                    # test suite
```
