import logging
from typing import List, Tuple
from shared.config import settings
from shared.models.schemas import RetrievedChunk, ConfidenceGateOutcome

logger = logging.getLogger(__name__)

class RetrievalConfidenceGate:
    def __init__(self, score_floor: float = settings.RERANK_SCORE_FLOOR):
        self.score_floor = score_floor

    def evaluate_confidence(self, final_chunks: List[RetrievedChunk], is_retry: bool = False) -> Tuple[ConfidenceGateOutcome, float, int]:
        """
        Hard architectural confidence gate.
        Computes top_rerank_score and n_chunks_above_floor.
        Returns (SUFFICIENT / RETRY / INSUFFICIENT, top_score, count_above_floor).
        """
        if not final_chunks:
            return ConfidenceGateOutcome.INSUFFICIENT, 0.0, 0

        top_score = max(c.rerank_score for c in final_chunks)
        chunks_above_floor = sum(1 for c in final_chunks if c.rerank_score >= self.score_floor)

        if top_score >= self.score_floor and chunks_above_floor >= 2:
            outcome = ConfidenceGateOutcome.SUFFICIENT
        elif not is_retry:
            outcome = ConfidenceGateOutcome.RETRY
        else:
            outcome = ConfidenceGateOutcome.INSUFFICIENT

        logger.info(
            f"Retrieval Confidence Gate: outcome={outcome.value}, "
            f"top_score={top_score:.3f}, chunks_above_floor={chunks_above_floor} (floor={self.score_floor})"
        )
        return outcome, top_score, chunks_above_floor
