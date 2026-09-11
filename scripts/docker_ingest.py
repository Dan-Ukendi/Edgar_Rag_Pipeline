"""Idempotent ingest step for the Docker stack: fetches filings if missing,
builds the index if the Qdrant collection is empty. Safe to run on every
`docker compose up` -- skips work that's already done.
"""

import subprocess
import sys
from pathlib import Path

from tenk_rag.config import PipelineConfig
from tenk_rag.store.qdrant_store import QdrantStore


def main() -> None:
    config = PipelineConfig.from_yaml("configs/pipeline.yaml")

    raw_dir = Path(config.raw_data_dir)
    existing_filings = list(raw_dir.glob("*.htm"))
    if existing_filings:
        print(f"[ingest] Found {len(existing_filings)} existing filings, skipping fetch.")
    else:
        print("[ingest] No filings found, fetching from SEC EDGAR...")
        subprocess.run([sys.executable, "scripts/fetch_filings.py"], check=True)

    # dimension is unused by collection_info() (get_collection doesn't need
    # it) -- a placeholder here avoids loading the embedding model just to
    # check whether the index already exists.
    store = QdrantStore(
        collection_name=config.vector_store.collection_name,
        dimension=1,
        mode=config.vector_store.mode,
        url=config.vector_store.url,
        local_path=config.vector_store.local_path,
    )
    try:
        points_count = store.collection_info()["points_count"]
    except Exception:
        points_count = 0

    if points_count > 0:
        print(f"[ingest] Collection already has {points_count} points, skipping index build.")
    else:
        print("[ingest] Qdrant collection is empty, building index...")
        subprocess.run([sys.executable, "scripts/build_index.py"], check=True)


if __name__ == "__main__":
    main()
