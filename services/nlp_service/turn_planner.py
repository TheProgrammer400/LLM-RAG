from shared.models.schemas import PatientState, TurnStrategy

class TurnPlanner:
    def plan_turn(self, state: PatientState) -> TurnStrategy:
        """Determines turn strategy (Information Gathering vs Diagnostic Reasoning)."""
        facts = state.extracted_facts
        if not facts.symptoms and not facts.lab_findings:
            return TurnStrategy.INFORMATION_GATHERING
        if len(facts.symptoms) < 2 and not facts.lab_findings:
            return TurnStrategy.SYMPTOM_CLARIFICATION
        return TurnStrategy.DIAGNOSTIC_REASONING
