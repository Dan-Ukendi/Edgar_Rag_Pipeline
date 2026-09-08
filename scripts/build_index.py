"""Run the full ingest pipeline: raw filings -> parse -> chunk -> embed -> upsert.

Usage:
    uv run scripts/build_index.py
"""

from tenk_rag.config import PipelineConfig
from tenk_rag.pipeline import run_pipeline


def main() -> None:
    config = PipelineConfig.from_yaml("configs/pipeline.yaml")
    run_pipeline(config)


if __name__ == "__main__":
    main()
