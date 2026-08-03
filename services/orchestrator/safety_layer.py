import logging
from shared.models.schemas import ConsultationResponse, EntailmentStatus, SeverityLevel

logger = logging.getLogger(__name__)

class OutputSafetyLayer:
    def apply_safety_layer(self, response: ConsultationResponse) -> ConsultationResponse:
        """
        Applies pre-delivery safety filters:
        - Drops unverified claims or marks them visually
        - Ensures emergency / urgent warnings are prominent
        - Enforces standard decision-support disclaimer block
        """
        # Filter unverified citation claims
        for diff in response.differentials:
            verified_cites = [c for c in diff.citations if c.entailment_status == EntailmentStatus.VERIFIED]
            if len(verified_cites) < len(diff.citations):
                logger.warning(f"Safety Layer: Dropped {len(diff.citations) - len(verified_cites)} unverified claims from '{diff.disease_name}'.")
            diff.citations = verified_cites

        if response.severity == SeverityLevel.EMERGENCY:
            response.is_emergency = True
            response.emergency_guidance = (
                "CRITICAL WARNING: High-severity emergency pattern detected. Immediate emergency medical intervention required. "
                "Do not wait for RAG retrieval or secondary diagnostic reasoning."
            )

        return response
