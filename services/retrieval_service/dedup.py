import logging
from typing import List
from shared.models.schemas import RetrievedChunk
from ingestion.embedder import MedicalEmbedder

logger = logging.getLogger(__name__)

class CandidateDeduplicator:
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.embedder = MedicalEmbedder()

    def deduplicate_candidates(self, candidates: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Deduplicates candidate hits BEFORE spending cross-encoder reranking compute.
        Target candidate pool duplicate ratio < 15%.
        """
        if len(candidates) <= 1:
            return candidates

        texts = [c.chunk.text for c in candidates]
        embs = self.embedder.embed_batch(texts)

        unique_candidates: List[RetrievedChunk] = []
        unique_embs: List[List[float]] = []

        for i, cand in enumerate(candidates):
            emb = embs[i]
            is_dup = False

            for u_emb in unique_embs:
                sim = self.embedder.cosine_similarity(emb, u_emb)
                if sim >= self.similarity_threshold:
                    is_dup = True
                    break

            if not is_dup:
                unique_candidates.append(cand)
                unique_embs.append(emb)

        dup_ratio = 1.0 - (len(unique_candidates) / len(candidates))
        logger.info(f"Retrieved candidate pool deduplicated: {len(candidates)} -> {len(unique_candidates)} (Duplicate ratio: {dup_ratio:.2%})")
        return unique_candidates
