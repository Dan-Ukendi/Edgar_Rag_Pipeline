from typing import Literal

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    ticker: str | None = None
    top_k: int | None = None
    backend: Literal["groq", "local"] | None = None


class ExcerptOut(BaseModel):
    index: int
    ticker: str
    filing_date: str
    item_section: str | None
    text: str
    cited: bool


class QueryResponse(BaseModel):
    question: str
    answer: str
    backend: str
    cited_indices: list[int]
    excerpts: list[ExcerptOut]


class VectorStoreHealth(BaseModel):
    reachable: bool
    points_count: int | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    embedding_model: Literal["loaded"]
    vector_store: VectorStoreHealth
    llm_backends: dict[str, bool]
