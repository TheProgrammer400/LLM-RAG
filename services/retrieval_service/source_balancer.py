import logging
from typing import List, Dict
from shared.config import settings
from shared.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

class SourceDiversityBalancer:
    def __init__(self, max_percentage_per_source: float = settings.MAX_SOURCE_PERCENTAGE):
        self.max_pct = max_percentage_per_source

    def balance_sources(self, ranked_chunks: List[RetrievedChunk], target_count: int = 15) -> List[RetrievedChunk]:
        """
        Enforces max-per-source cap (e.g. max 40% of final set from any single source)
        as a HARD CONSTRAINT applied last after reranking and weighting.
        Iterates down the ranked list and skips any chunk that would violate the cap.
        """
        if not ranked_chunks:
            return []

        max_allowed_per_source = max(1, int(target_count * self.max_pct))
        source_counts: Dict[str, int] = {}
        balanced_list: List[RetrievedChunk] = []

        for item in ranked_chunks:
            src_id = item.chunk.metadata.source_id
            current_count = source_counts.get(src_id, 0)

            if current_count < max_allowed_per_source:
                balanced_list.append(item)
                source_counts[src_id] = current_count + 1
            else:
                logger.info(f"Source cap reached for '{src_id}' ({current_count}/{max_allowed_per_source}). Skipping chunk {item.chunk.chunk_id}.")

            if len(balanced_list) >= target_count:
                break

        return balanced_list
