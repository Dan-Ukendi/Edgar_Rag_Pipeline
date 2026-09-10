"""Shared retrieve-then-generate orchestration, used by both the CLI
(scripts/ask.py) and the API (src/tenk_rag/api) so the two never drift."""

from dataclasses import dataclass

from .config import PipelineConfig
from .embeddings.local import SentenceTransformerEmbedder
from .generation.answer import AnswerResult, answer_question
from .llm.base import LLMClient
from .llm.groq_client import GroqLLM
from .llm.ollama_client import OllamaLLM
from .store.qdrant_store import QdrantStore


@dataclass
class RagService:
    config: PipelineConfig
    embedder: SentenceTransformerEmbedder
    store: QdrantStore
    llm_clients: dict[str, LLMClient]

    @classmethod
    def build(cls, config: PipelineConfig) -> "RagService":
        embedder = SentenceTransformerEmbedder(config.embedding.model_name, config.embedding.batch_size)
        store = QdrantStore(
            collection_name=config.vector_store.collection_name,
            dimension=embedder.dimension,
            mode=config.vector_store.mode,
            url=config.vector_store.url,
            local_path=config.vector_store.local_path,
        )

        llm_clients: dict[str, LLMClient] = {}
        try:
            llm_clients["groq"] = GroqLLM(model=config.llm.groq.model, max_tokens=config.llm.groq.max_tokens)
        except RuntimeError:
            pass  # no GROQ_API_KEY -- backend simply unavailable
        llm_clients["local"] = OllamaLLM(
            model=config.llm.local.model,
            base_url=config.llm.local.base_url,
            max_tokens=config.llm.local.max_tokens,
        )

        return cls(config=config, embedder=embedder, store=store, llm_clients=llm_clients)

    def ask(
        self,
        question: str,
        ticker: str | None = None,
        top_k: int | None = None,
        backend: str | None = None,
    ) -> AnswerResult:
        backend = backend or self.config.llm.backend
        if backend not in self.llm_clients:
            raise ValueError(f"Backend '{backend}' is not available (available: {list(self.llm_clients)})")

        query_vector = self.embedder.embed([question])[0]
        multi_company = ticker is None

        if multi_company:
            # No ticker filter means "All companies" -- retrieve top_k_per_company
            # chunks for each distinct ticker so every company is represented,
            # rather than whichever companies happen to score highest overall.
            # num_groups is a generous cap above the current 6-company corpus.
            results = self.store.query_grouped_by_ticker(
                query_vector, group_size=self.config.retrieval.top_k_per_company, num_groups=10
            )
        else:
            top_k = top_k or self.config.retrieval.top_k
            results = self.store.query(query_vector, top_k=top_k, ticker=ticker)

        excerpts = [
            {
                "ticker": r.payload["ticker"],
                "filing_date": r.payload["filing_date"],
                "item_section": r.payload["item_section"],
                "text": r.payload["text"],
            }
            for r in results
        ]

        return answer_question(
            question, excerpts, self.llm_clients[backend], backend=backend, multi_company=multi_company
        )
