from router import Intent

class ClinicalPlan:
    def __init__(self, requiresRag = False, ragCollection = "medical_knowledge", instructions = ""):
        self.requiresRag = requiresRag
        self.ragCollection = ragCollection
        self.instructions = instructions

    @property
    def requires_rag(self):
        # Expose properties to match chatbot.py expectation
        return self.requiresRag

    @property
    def rag_collection(self):
        # Expose properties to match chatbot.py expectation
        return self.ragCollection


class ClinicalPlanner:
    def __init__(self):
        pass

    def plan_turn(self, question, intent, consultationState, profile):
        # Using camelCase variables and spacing as requested
        requiresRag = False
        if intent == Intent.MEDICAL:
            requiresRag = True

        ragCollection = "medical_knowledge"
        instructionsList = []

        numSymptoms = len(consultationState.current_symptoms)
        if numSymptoms == 0:
            instructionsList.append("Prompt the patient politely to describe their symptoms or health concerns.")
        elif numSymptoms == 1:
            symptom = consultationState.current_symptoms[0]
            instructionsList.append(
                f"The patient has reported one symptom: '{symptom}'. "
                "Do NOT jump to listing diseases yet. Focus on gathering more details: ask about its duration, "
                "severity, onset, and any potential triggers or associated symptoms."
            )
        else:
            symptomsStr = ", ".join(consultationState.current_symptoms)
            instructionsList.append(
                f"The patient has reported multiple symptoms: {symptomsStr}. "
                "Begin discussing broad, common categories of mild illness that might explain these symptoms. "
                "Use cautious uncertainty language (e.g., 'One possible explanation is...', 'At this stage, it's too early to confirm')."
            )

        if consultationState.negativeFindings:
            negStr = ", ".join(consultationState.negativeFindings)
            instructionsList.append(f"Note that the patient has explicitly denied experiencing: {negStr}. Do not suggest these symptoms.")

        if consultationState.givenRecommendations:
            recStr = ", ".join(consultationState.givenRecommendations)
            instructionsList.append(f"Avoid repeating these previously discussed recommendations: {recStr}.")

        instructions = "\n".join(f"- {inst}" for inst in instructionsList)

        plan = ClinicalPlan(requiresRag = requiresRag, ragCollection = ragCollection, instructions = instructions)
        return plan
