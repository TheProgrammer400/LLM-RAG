from enum import Enum

class ConsultationStage(Enum):
    GREETING = "GREETING"
    SYMPTOM_GATHERING = "SYMPTOM_GATHERING"
    DIFFERENTIAL_DIAGNOSIS = "DIFFERENTIAL_DIAGNOSIS"
    TREATMENT_PLAN = "TREATMENT_PLAN"


class ConsultationState:
    def __init__(self):
        # Using camelCase for variable names as per guidelines
        self.currentSymptoms = []
        self.symptomMetadata = {}
        self.negativeFindings = []
        self.givenRecommendations = []
        self.currentStage = ConsultationStage.GREETING

    @property
    def current_symptoms(self):
        # Expose property to match chatbot.py expectation
        return self.currentSymptoms

    def add_symptom(self, symptomName, metadata = None):
        # Maintain spacing around = and operators
        cleanSymptom = symptomName.strip().lower()
        if cleanSymptom not in self.currentSymptoms:
            self.currentSymptoms.append(cleanSymptom)
        
        if metadata:
            if cleanSymptom not in self.symptomMetadata:
                self.symptomMetadata[cleanSymptom] = {}
            self.symptomMetadata[cleanSymptom].update(metadata)

    def add_negative_finding(self, findingName):
        cleanFinding = findingName.strip().lower()
        if cleanFinding not in self.negativeFindings:
            self.negativeFindings.append(cleanFinding)

    def add_recommendation(self, recommendation):
        cleanRec = recommendation.strip().lower()
        if cleanRec not in self.givenRecommendations:
            self.givenRecommendations.append(cleanRec)

    def reset_session(self):
        self.currentSymptoms = []
        self.symptomMetadata = {}
        self.negativeFindings = []
        self.givenRecommendations = []
        self.currentStage = ConsultationStage.GREETING
