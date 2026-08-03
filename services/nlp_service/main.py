from fastapi import FastAPI
from pydantic import BaseModel
from shared.models.schemas import PatientState, IntentType, SeverityLevel, TurnStrategy, FactExtraction
from services.nlp_service.intent_classifier import IntentClassifier
from services.nlp_service.clinical_extractor import ClinicalFactExtractor
from services.nlp_service.entity_normalizer import EntityNormalizer
from services.nlp_service.red_flag_detector import RedFlagDetector
from services.nlp_service.turn_planner import TurnPlanner

app = FastAPI(title="CDSS NLP Service", version="1.0.0")

intent_classifier = IntentClassifier()
fact_extractor = ClinicalFactExtractor()
entity_normalizer = EntityNormalizer()
red_flag_detector = RedFlagDetector()
turn_planner = TurnPlanner()

class NLPProcessRequest(BaseModel):
    session_id: str
    physician_input: str
    existing_state: PatientState

class NLPProcessResponse(BaseModel):
    intent: IntentType
    extracted_facts: FactExtraction
    normalized_entities: list
    severity: SeverityLevel
    red_flags: list
    turn_strategy: TurnStrategy

@app.post("/process", response_model=NLPProcessResponse)
async def process_input(request: NLPProcessRequest):
    intent = intent_classifier.classify_intent(request.physician_input)
    extracted = fact_extractor.extract_facts(request.physician_input)
    normalized = entity_normalizer.normalize_extracted_facts(extracted)

    # Merge into state for red flag assessment
    merged_state = request.existing_state.model_copy()
    merged_state.extracted_facts = extracted
    merged_state.normalized_entities = normalized

    severity, red_flags = red_flag_detector.assess_severity_and_red_flags(merged_state)
    strategy = turn_planner.plan_turn(merged_state)

    return NLPProcessResponse(
        intent=intent,
        extracted_facts=extracted,
        normalized_entities=normalized,
        severity=severity,
        red_flags=red_flags,
        turn_strategy=strategy
    )
