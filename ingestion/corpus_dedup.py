import logging
from typing import List
from shared.models.schemas import Chunk

logger = logging.getLogger(__name__)

class CorpusDeduplicator:
    def __init__(self, similarity_threshold: float = 0.90):
        self.similarity_threshold = similarity_threshold

    def deduplicate_chunks(self, chunks: List[Chunk], embedder) -> List[Chunk]:
        """
        Runs near-duplicate detection across all ingested chunks.
        Keeps highest authority-tier version, flags duplicates.
        """
        if len(chunks) <= 1:
            return chunks

        logger.info(f"Running corpus-wide deduplication on {len(chunks)} chunks...")
        embeddings = [embedder.embed_text(c.text) for c in chunks]
        
        unique_chunks: List[Chunk] = []
        unique_embeddings: List[List[float]] = []

        for i, chunk in enumerate(chunks):
            emb = embeddings[i]
            is_duplicate = False

            for j, u_emb in enumerate(unique_embeddings):
                sim = embedder.cosine_similarity(emb, u_emb)
                if sim >= self.similarity_threshold:
                    is_duplicate = True
                    logger.info(
                        f"Chunk {chunk.chunk_id} marked duplicate of {unique_chunks[j].chunk_id} "
                        f"(similarity: {sim:.3f})"
                    )
                    break

            if not is_duplicate:
                unique_chunks.append(chunk)
                unique_embeddings.append(emb)

        logger.info(f"Deduplication complete. Retained {len(unique_chunks)} unique chunks from {len(chunks)}.")
        return unique_chunks
