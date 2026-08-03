from typing import List
from shared.models.schemas import DifferentialItem

class ReasoningEvaluator:
    def evaluate_differential_recall(self, differentials: List[DifferentialItem], expected_top_disease: str) -> bool:
        """Checks if expected disease appears in top-3 differentials."""
        if not expected_top_disease or not differentials:
            return False
        top_3 = [d.disease_name.lower() for d in differentials[:3]]
        return any(expected_top_disease.lower() in d_name for d_name in top_3)
