import pytest
from shared.models.schemas import PatientState, SeverityLevel
from services.nlp_service.clinical_extractor import ClinicalFactExtractor
from services.nlp_service.red_flag_detector import RedFlagDetector

def test_negation_prevents_red_flag_trigger():
    extractor = ClinicalFactExtractor()
    detector = RedFlagDetector()

    # Case: "patient denies chest pain"
    facts = extractor.extract_facts("72F, temporal headache, patient denies chest pain and denies shortness of breath.")
    
    assert "chest pain" in facts.negative_findings
    assert "chest pain" not in facts.symptoms

    state = PatientState(session_id="test_negation", extracted_facts=facts)
    severity, red_flags = detector.assess_severity_and_red_flags(state)

    # Cardiac emergency pattern must NOT trigger!
    assert severity != SeverityLevel.EMERGENCY
    assert not any("STEMI" in rf or "Coronary" in rf for rf in red_flags)
