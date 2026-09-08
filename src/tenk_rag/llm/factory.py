from ..config import LLMConfig
from .base import LLMClient
from .groq_client import GroqLLM
from .ollama_client import OllamaLLM


def build_llm_client(config: LLMConfig) -> LLMClient:
    if config.backend == "groq":
        return GroqLLM(model=config.groq.model, max_tokens=config.groq.max_tokens)
    if config.backend == "local":
        return OllamaLLM(
            model=config.local.model,
            base_url=config.local.base_url,
            max_tokens=config.local.max_tokens,
        )
    raise ValueError(f"Unknown LLM backend: {config.backend}")
