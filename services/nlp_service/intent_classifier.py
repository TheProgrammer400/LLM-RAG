from shared.models.schemas import IntentType

class IntentClassifier:
    def classify_intent(self, text: str) -> IntentType:
        """Classifies incoming user intent."""
        t = text.strip().lower()
        if t in ["exit", "quit", "bye", "stop"]:
            return IntentType.EXIT
        if t in ["hello", "hi", "hey", "greetings"]:
            return IntentType.GREETING
        if any(kw in t for kw in ["lab", "esr", "crp", "mri", "ct", "ultrasound", "biopsy"]):
            return IntentType.INVESTIGATION_RESULTS_UPDATE
        if any(kw in t for kw in ["what is", "explain", "how does", "guideline for"]):
            return IntentType.GENERAL_MEDICAL_QUESTION
        if any(kw in t for kw in ["symptom", "pain", "headache", "fever", "loss"]):
            return IntentType.SYMPTOM_DISCUSSION
        
        return IntentType.CLINICAL_UPDATE
