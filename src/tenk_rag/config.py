"""Pydantic config models loaded from configs/pipeline.yaml.

Kept as plain, serializable settings so Phase 6 can generate many variants
of this config programmatically for benchmarking sweeps.
"""

import os
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


class GroqLLMConfig(BaseModel):
    model: str = "openai/gpt-oss-120b"
    max_tokens: int = Field(2048, gt=0)


class LocalLLMConfig(BaseModel):
    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    max_tokens: int = Field(2048, gt=0)


class LLMConfig(BaseModel):
    backend: Literal["groq", "local"] = "groq"
    groq: GroqLLMConfig = GroqLLMConfig()
    local: LocalLLMConfig = LocalLLMConfig()


class RetrievalConfig(BaseModel):
    top_k: int = Field(5, gt=0)
    top_k_per_company: int = Field(2, gt=0, description="Chunks retrieved per company when no ticker filter is set")


class PipelineConfig(BaseModel):
    raw_data_dir: str = "data/raw"
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    llm: LLMConfig = LLMConfig()
    retrieval: RetrievalConfig = RetrievalConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        config = cls(**data)

        # Service-address overrides for containerized environments, where
        # "localhost" from the YAML doesn't resolve to other containers.
        if qdrant_url := os.environ.get("QDRANT_URL"):
            config.vector_store.url = qdrant_url
        if ollama_url := os.environ.get("OLLAMA_BASE_URL"):
            config.llm.local.base_url = ollama_url

        return config
