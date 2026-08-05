import pytest
from services.orchestrator.main import ConsultationTurnRequest

def test_request_with_full_payload():
    req = ConsultationTurnRequest(session_id="sess_123", physician_input="Patient has fever")
    assert req.session_id == "sess_123"
    assert req.physician_input == "Patient has fever"

def test_request_without_session_id():
    req = ConsultationTurnRequest.model_validate({"physician_input": "Patient has headache"})
    assert req.session_id is not None
    assert req.session_id.startswith("session_")
    assert req.physician_input == "Patient has headache"

def test_request_with_query_alias():
    req = ConsultationTurnRequest.model_validate({"query": "Patient has chest pain"})
    assert req.session_id is not None
    assert req.session_id.startswith("session_")
    assert req.physician_input == "Patient has chest pain"

def test_request_with_input_alias():
    req = ConsultationTurnRequest.model_validate({"input": "Patient has cough", "session_id": "custom_sess"})
    assert req.session_id == "custom_sess"
    assert req.physician_input == "Patient has cough"

def test_request_with_raw_string():
    req = ConsultationTurnRequest.model_validate("Patient has sore throat")
    assert req.session_id.startswith("session_")
    assert req.physician_input == "Patient has sore throat"
