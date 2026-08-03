from typing import List
from shared.models.schemas import CitationClaim, EntailmentStatus

class FaithfulnessEvaluator:
    def evaluate_faithfulness(self, citations: List[CitationClaim]) -> float:
        """Computes percentage of generated claims passing the NLI entailment check."""
        if not citations:
            return 1.0  # No uncited claims
        verified_count = sum(1 for c in citations if c.entailment_status == EntailmentStatus.VERIFIED)
        return verified_count / len(citations)
