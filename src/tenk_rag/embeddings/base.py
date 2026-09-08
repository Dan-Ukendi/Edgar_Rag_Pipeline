"""Interface every embedding backend implements, so callers (pipeline, query
script) never need to know whether they're talking to a local model or a
hosted API."""

from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...
