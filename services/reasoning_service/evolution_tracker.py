from typing import List, Dict, Any
from shared.models.schemas import CandidateEvidenceMatch, EvolutionDelta, CompatibilityTier

class EvolutionTracker:
    def track_ranking_evolution(
        self,
        current_candidates: List[CandidateEvidenceMatch],
        previous_ranked: List[Dict[str, Any]]
    ) -> List[EvolutionDelta]:
        """Tracks changes in candidate compatibility tiers across multi-turn consultation."""
        prev_map = {p["disease_name"]: p.get("compatibility_tier") for p in previous_ranked}
        deltas = []

        for cand in current_candidates:
            prev_tier_str = prev_map.get(cand.disease_name)
            prev_tier = CompatibilityTier(prev_tier_str) if prev_tier_str else None

            if prev_tier != cand.compatibility_tier:
                reason = f"Rank evolved from {prev_tier.value if prev_tier else 'Unranked'} -> {cand.compatibility_tier.value}"
                deltas.append(EvolutionDelta(
                    disease_name=cand.disease_name,
                    previous_tier=prev_tier,
                    new_tier=cand.compatibility_tier,
                    change_reason=reason
                ))

        return deltas
