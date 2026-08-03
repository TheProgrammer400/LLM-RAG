import numpy as np
import logging
from typing import List
from shared.config import settings

logger = logging.getLogger(__name__)

class MedicalEmbedder:
    def __init__(self, model_name: str = settings.PRIMARY_EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.vector_dim = 768
        self._initialize_model()

    def _initialize_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            if hasattr(self.model, "get_embedding_dimension"):
                self.vector_dim = self.model.get_embedding_dimension()
            else:
                self.vector_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded SentenceTransformer embedding model: {self.model_name} (dim: {self.vector_dim})")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer model {self.model_name} ({e}). Using deterministic fallback mock embedder.")

    def embed_text(self, text: str) -> List[float]:
        """Embeds a single string into a dense vector."""
        if self.model:
            emb = self.model.encode(text, convert_to_numpy=True)
            return emb.tolist()
        else:
            return self._mock_embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embeds a list of strings."""
        if self.model:
            embs = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
            return [e.tolist() for e in embs]
        else:
            return [self._mock_embed(t) for t in texts]

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculates cosine similarity between two vectors."""
        v1 = np.array(vec1, dtype=float)
        v2 = np.array(vec2, dtype=float)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _mock_embed(self, text: str) -> List[float]:
        """Generates a pseudo-embedding vector for offline unit tests without model weights."""
        import hashlib
        vec = [0.0] * self.vector_dim
        words = text.lower().split()
        for i, word in enumerate(words):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % self.vector_dim
            vec[idx] += 1.0 / (i + 1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = (np.array(vec) / norm).tolist()
        return vec
