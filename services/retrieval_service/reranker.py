import logging
from typing import List
from shared.config import settings
from shared.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    def __init__(self, model_name: str = settings.RERANKER_MODEL):
        self.model_name = model_name
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded CrossEncoder model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load CrossEncoder model {self.model_name} ({e}). Using normalized RRF fallback scores.")

    def rerank(self, primary_query: str, candidates: List[RetrievedChunk], top_n: int = 20) -> List[RetrievedChunk]:
        """
        Reranks top candidate pool using CrossEncoder against the primary presentation facet query.
        """
        if not candidates:
            return []

        if self.model:
            try:
                pairs = [(primary_query, c.chunk.text) for c in candidates]
                scores = self.model.predict(pairs)
                for idx, score in enumerate(scores):
                    # Sigmoid or linear normalization if raw logits
                    import math
                    s_float = float(score)
                    norm_score = 1.0 / (1.0 + math.exp(-s_float)) if s_float < 0 or s_float > 1 else s_float
                    candidates[idx].rerank_score = norm_score
                
                reranked = sorted(candidates, key=lambda x: x.rerank_score, reverse=True)
                return reranked[:top_n]
            except Exception as e:
                logger.warning(f"CrossEncoder prediction failed: {e}. Falling back to RRF score scaling.")

        # Fallback using normalized RRF rank scores
        max_rrf = max(c.rrf_score for c in candidates) if candidates else 1.0
        for c in candidates:
            c.rerank_score = min(0.95, (c.rrf_score / max(max_rrf, 1e-6)) * 0.88)
        
        reranked = sorted(candidates, key=lambda x: x.rerank_score, reverse=True)
        return reranked[:top_n]
