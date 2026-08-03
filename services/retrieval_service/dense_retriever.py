import logging
from typing import List
from shared.config import settings
from shared.models.schemas import RetrievedChunk, Chunk, ChunkMetadata

logger = logging.getLogger(__name__)

class DenseRetriever:
    def __init__(self, qdrant_host: str = settings.QDRANT_HOST, qdrant_port: int = settings.QDRANT_PORT):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port, timeout=3.0)
        except Exception as e:
            logger.warning(f"DenseRetriever: Qdrant client connection unavailable ({e}).")

    async def search_facet(self, query_text: str, embedder, top_k: int = 50, collection_name: str = "clinical_reference") -> List[RetrievedChunk]:
        """Runs dense vector search on Qdrant."""
        emb = embedder.embed_text(query_text)
        retrieved: List[RetrievedChunk] = []

        if self.client:
            try:
                hits = []
                if hasattr(self.client, "query_points"):
                    res = self.client.query_points(collection_name=collection_name, query=emb, limit=top_k)
                    hits = res.points
                elif hasattr(self.client, "search"):
                    hits = self.client.search(collection_name=collection_name, query_vector=emb, limit=top_k)

                for hit in hits:
                    payload = hit.payload
                    metadata = ChunkMetadata(**payload)
                    chunk = Chunk(
                        chunk_id=metadata.chunk_id,
                        text=payload.get("text", ""),
                        parent_text=payload.get("parent_text"),
                        metadata=metadata
                    )
                    retrieved.append(RetrievedChunk(chunk=chunk, dense_score=float(hit.score)))
                return retrieved
            except Exception as e:
                logger.debug(f"Qdrant search error: {e}")

        return retrieved
