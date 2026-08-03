import os
import json
import logging
from typing import List, Dict, Any
from shared.models.schemas import RetrievedChunk, Chunk, ChunkMetadata, AuthorityTier, ContentType

logger = logging.getLogger(__name__)

class SparseRetriever:
    def __init__(self, collection_name: str = "clinical_reference"):
        self.collection_name = collection_name
        self.bm25_index = None
        self.documents: List[Dict[str, Any]] = []
        self._load_index()

    def _load_index(self):
        docs_file = f".bm25_store/{self.collection_name}_docs.json"
        if os.path.exists(docs_file):
            try:
                import bm25s
                self.bm25_index = bm25s.BM25.load(f".bm25_store/{self.collection_name}_index", load_corpus=False)
                with open(docs_file, "r") as f:
                    self.documents = json.load(f)
                logger.info(f"SparseRetriever: Loaded BM25 index with {len(self.documents)} documents.")
            except Exception as e:
                logger.warning(f"Could not load BM25 index ({e}).")

    async def search_facet(self, query_text: str, top_k: int = 50) -> List[RetrievedChunk]:
        """Runs BM25 sparse exact keyword search."""
        retrieved: List[RetrievedChunk] = []

        if self.bm25_index and self.documents:
            try:
                import bm25s
                query_tokens = bm25s.tokenize([query_text])
                results, scores = self.bm25_index.retrieve(query_tokens, k=min(top_k, len(self.documents)))
                
                for idx, doc_idx in enumerate(results[0]):
                    doc = self.documents[doc_idx]
                    metadata = ChunkMetadata(**doc["metadata"])
                    chunk = Chunk(
                        chunk_id=doc["chunk_id"],
                        text=doc["text"],
                        parent_text=doc.get("parent_text"),
                        metadata=metadata
                    )
                    score = float(scores[0][idx])
                    retrieved.append(RetrievedChunk(chunk=chunk, sparse_score=score))
                return retrieved
            except Exception as e:
                logger.error(f"BM25 retrieval error: {e}")

        return retrieved
