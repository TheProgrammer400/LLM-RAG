import logging
from typing import List
from shared.models.schemas import Chunk

logger = logging.getLogger(__name__)

class CoherenceChecker:
    def __init__(self, min_similarity_threshold: float = 0.25):
        self.min_similarity_threshold = min_similarity_threshold

    def check_heading_coherence(self, chunks: List[Chunk], embedder) -> List[Chunk]:
        """
        Embeds heading vs chunk body and checks cosine similarity.
        Flags chunks that fall below the threshold.
        """
        valid_chunks = []
        for chunk in chunks:
            heading = chunk.metadata.heading
            body_snippet = chunk.text[:300]

            if len(heading) < 3 or "General Overview" in heading:
                valid_chunks.append(chunk)
                continue

            heading_emb = embedder.embed_text(heading)
            body_emb = embedder.embed_text(body_snippet)
            sim = embedder.cosine_similarity(heading_emb, body_emb)

            if sim < self.min_similarity_threshold:
                logger.warning(
                    f"Low heading-body coherence ({sim:.2f}) for chunk {chunk.chunk_id} "
                    f"under heading '{heading}'. Flagged for review."
                )
                chunk.metadata.ocr_derived = True  # Flagged metadata tag
            
            valid_chunks.append(chunk)

        return valid_chunks
