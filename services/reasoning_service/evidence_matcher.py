from typing import List, Dict, Any
from shared.models.schemas import PatientState, CandidateEvidenceMatch, CompatibilityTier, ProvenanceCitation

class EvidenceMatcher:
    def match_patient_state(
        self,
        candidate_info: Dict[str, Any],
        state: PatientState,
        evidence_text: str
    ) -> CandidateEvidenceMatch:
        """
        Compares PatientState (present findings, negations) against evidence text.
        Produces matching_present_findings, conflicting_prerequisites, and provenance citations.
        """
        disease_name = candidate_info["disease_name"]
        facts = state.extracted_facts
        ev_lower = evidence_text.lower()

        matching = []
        conflicts = []
        missing = []

        # Check matched symptoms
        for s in facts.symptoms:
            if s.lower() in ev_lower or any(w in ev_lower for w in s.lower().split()):
                matching.append(s)

        # Check negated conflicts
        for neg in facts.negative_findings:
            if neg.lower() in ev_lower:
                conflicts.append(f"Patient negated finding: '{neg}'")

        # Provenance
        provenance = [
            ProvenanceCitation(
                source_title=candidate_info["source_title"],
                page=candidate_info["page"],
                chunk_id=candidate_info["chunk_id"],
                excerpt=evidence_text[:200] + "..."
            )
        ]

        # Determine evidence-overlap compatibility tier
        if len(matching) >= 3 and not conflicts:
            tier = CompatibilityTier.MOST_COMPATIBLE
            rationale = f"Strong evidence overlap: Matched {len(matching)} key clinical features ({', '.join(matching)}) with 0 negation conflicts."
        elif len(matching) >= 1 and not conflicts:
            tier = CompatibilityTier.COMPATIBLE
            rationale = f"Moderate evidence overlap: Matched features ({', '.join(matching)})."
        elif conflicts:
            tier = CompatibilityTier.LESS_COMPATIBLE
            rationale = f"Conflicting prerequisites detected: {', '.join(conflicts)}."
        else:
            tier = CompatibilityTier.POSSIBLE
            rationale = "Possible candidate with partial feature matching."

        return CandidateEvidenceMatch(
            disease_name=disease_name,
            cui=candidate_info.get("cui"),
            compatibility_tier=tier,
            matching_present_findings=matching,
            conflicting_prerequisites=conflicts,
            missing_high_yield_features=missing,
            provenance=provenance,
            tier_rationale=rationale
        )
