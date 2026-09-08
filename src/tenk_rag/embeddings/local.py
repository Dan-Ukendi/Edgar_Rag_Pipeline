from sentence_transformers import SentenceTransformer

from .base import EmbeddingModel


class SentenceTransformerEmbedder(EmbeddingModel):
    def __init__(self, model_name: str, batch_size: int = 32):
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, batch_size=self._batch_size, show_progress_bar=False)
        return vectors.tolist()
