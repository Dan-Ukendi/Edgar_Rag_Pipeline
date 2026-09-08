"""Interface every LLM backend implements, so answer-generation code never
needs to know whether it's talking to a hosted API or a local model."""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str: ...
