import pytest
from services.orchestrator.flow import OrchestratorFlow
from shared.models.schemas import SeverityLevel, ConfidenceGateOutcome, CompatibilityTier
from ingestion.run_ingestion import create_sample_pdf_and_ingest

@pytest.mark.asyncio
async def test_section_13_end_to_end_gca_case():
    # 1. Ingest sample GCA reference document
    create_sample_pdf_and_ingest()

    flow = OrchestratorFlow()
    session_id = "test_gca_session_001"
    physician_input = "72F, sudden painless vision loss right eye, temporal headache, jaw claudication, 6kg weight loss over 2 months, ESR 110, CRP elevated, MRI normal, no diabetes."

    # 2. Run end-to-end orchestrator turn
    response = await flow.handle_consultation_turn(session_id, physician_input)

    # 3. Assertions matching Section 13 expected trace
    assert response.session_id == session_id
    assert response.severity == SeverityLevel.URGENT
    assert response.retrieval_confidence in [ConfidenceGateOutcome.SUFFICIENT, ConfidenceGateOutcome.RETRY]
    assert len(response.differentials) > 0

    top_diff = response.differentials[0]
    assert "Giant Cell Arteritis" in top_diff.disease_name
    assert top_diff.compatibility_tier in [CompatibilityTier.MOST_COMPATIBLE, CompatibilityTier.COMPATIBLE]
    assert len(top_diff.citations) > 0

    # Ensure latency metrics tracked
    assert "total_ms" in response.latency_ms
    assert response.latency_ms["total_ms"] > 0
