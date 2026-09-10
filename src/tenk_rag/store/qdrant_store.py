import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from ..ingest.chunker import Chunk


class QdrantStore:
    def __init__(
        self,
        collection_name: str,
        dimension: int,
        mode: str = "server",
        url: str = "http://localhost:6333",
        local_path: str = "data/processed/qdrant_local",
    ):
        self._client = QdrantClient(url=url) if mode == "server" else QdrantClient(path=local_path)
        self._collection_name = collection_name
        self._dimension = dimension

    def recreate_collection(self) -> None:
        self._client.recreate_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "ticker": chunk.ticker,
                    "filing_date": chunk.filing_date,
                    "source_file": chunk.source_file,
                    "item_section": chunk.item_section,
                    "chunk_index": chunk.chunk_index,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)

    def collection_info(self) -> dict:
        info = self._client.get_collection(self._collection_name)
        return {"points_count": info.points_count}

    def query(self, vector: list[float], top_k: int = 5, ticker: str | None = None):
        query_filter = None
        if ticker:
            query_filter = Filter(must=[FieldCondition(key="ticker", match=MatchValue(value=ticker))])
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
        )
        return response.points

    def query_grouped_by_ticker(self, vector: list[float], group_size: int, num_groups: int):
        """Retrieve the top `group_size` chunks per distinct ticker, so every
        company is represented instead of whichever scores highest overall."""
        response = self._client.query_points_groups(
            collection_name=self._collection_name,
            query=vector,
            group_by="ticker",
            group_size=group_size,
            limit=num_groups,
        )
        hits = []
        for group in sorted(response.groups, key=lambda g: g.id):
            hits.extend(group.hits)
        return hits
