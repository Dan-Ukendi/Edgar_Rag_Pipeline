"""Pydantic config models loaded from configs/pipeline.yaml.

Kept as plain, serializable settings so Phase 6 can generate many variants
of this config programmatically for benchmarking sweeps.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(1000, gt=0, description="Target chunk size, in characters")
    chunk_overlap: int = Field(150, ge=0, description="Overlap between consecutive chunks, in characters")

    @model_validator(mode="after")
    def _check_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class EmbeddingConfig(BaseModel):
    backend: Literal["local"] = "local"
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = Field(32, gt=0)


class VectorStoreConfig(BaseModel):
    backend: Literal["qdrant"] = "qdrant"
    collection_name: str = "tenk_filings"
    mode: Literal["server", "local"] = "server"
    url: str = "http://localhost:6333"
    local_path: str = "data/processed/qdrant_local"


class PipelineConfig(BaseModel):
    raw_data_dir: str = "data/raw"
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
