import os
import json
import logging
from typing import List, Dict, Any
from shared.config import settings
from shared.models.schemas import Chunk

logger = logging.getLogger(__name__)

class DualIndexer:
    def __init__(self, qdrant_host: str = settings.QDRANT_HOST, qdrant_port: int = settings.QDRANT_PORT):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_client = None
        self.bm25_index = None
        self.bm25_documents: List[Dict[str, Any]] = []
        self._initialize_qdrant()

    def _initialize_qdrant(self):
        try:
            from qdrant_client import QdrantClient
            self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port, timeout=5.0)
            logger.info(f"Connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
        except Exception as e:
            logger.warning(f"Qdrant connection not available ({e}). DualIndexer will operate in local storage mode.")

    def index_chunks(self, chunks: List[Chunk], embedder, collection_name: str = "clinical_reference"):
        """
        Indexes chunks into Qdrant vector database and BM25 sparse search.
        """
        if not chunks:
            logger.warning("No chunks to index.")
            return

        logger.info(f"Indexing {len(chunks)} chunks into collection '{collection_name}'...")
        embeddings = embedder.embed_batch([c.text for c in chunks])

        # 1. Qdrant indexing
        if self.qdrant_client:
            try:
                from qdrant_client.models import VectorParams, Distance, PointStruct
                
                # Ensure collection exists
                collections = [c.name for c in self.qdrant_client.get_collections().collections]
                if collection_name not in collections:
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=embedder.vector_dim, distance=Distance.COSINE)
                    )
                
                points = []
                for idx, chunk in enumerate(chunks):
                    payload = chunk.metadata.model_dump()
                    payload["text"] = chunk.text
                    payload["parent_text"] = chunk.parent_text
                    
                    points.append(PointStruct(
                        id=idx,  # Integer ID or UUID string
                        vector=embeddings[idx],
                        payload=payload
                    ))

                self.qdrant_client.upsert(collection_name=collection_name, points=points)
                logger.info(f"Upserted {len(points)} points to Qdrant collection '{collection_name}'.")
            except Exception as e:
                logger.error(f"Failed to upsert to Qdrant: {e}")

        # 2. BM25s indexing
        try:
            import bm25s
            corpus = [c.text for c in chunks]
            tokens = bm25s.tokenize(corpus)
            self.bm25_index = bm25s.BM25()
            self.bm25_index.index(tokens)

            # Store BM25 documents registry locally
            self.bm25_documents = [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "parent_text": c.parent_text,
                    "metadata": c.metadata.model_dump()
                }
                for c in chunks
            ]
            
            # Save local backup index
            os.makedirs(".bm25_store", exist_ok=True)
            self.bm25_index.save(f".bm25_store/{collection_name}_index")
            with open(f".bm25_store/{collection_name}_docs.json", "w") as f:
                json.dump(self.bm25_documents, f, default=str)
                
            logger.info(f"Successfully built BM25s sparse index for {len(chunks)} chunks.")
        except Exception as e:
            logger.warning(f"BM25s indexing failed ({e}). Storing raw documents in memory.")
            self.bm25_documents = [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "parent_text": c.parent_text,
                    "metadata": c.metadata.model_dump()
                }
                for c in chunks
            ]
