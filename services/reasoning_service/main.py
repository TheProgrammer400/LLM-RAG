from fastapi import FastAPI
from pydantic import BaseModel
from shared.models.schemas import PatientState, RetrievalResult, ReasoningResult
from services.reasoning_service.candidate_generator import CandidateGenerator
from services.reasoning_service.evidence_matcher import EvidenceMatcher
from services.reasoning_service.evolution_tracker import EvolutionTracker

app = FastAPI(title="CDSS Reasoning Service", version="1.0.0")

candidate_generator = CandidateGenerator()
evidence_matcher = EvidenceMatcher()
evolution_tracker = EvolutionTracker()

class ReasonRequest(BaseModel):
    state: PatientState
    retrieval_result: RetrievalResult

@app.post("/reason", response_model=ReasoningResult)
async def reason_candidates(request: ReasonRequest):
    # 1. Candidate Generation
    raw_candidates = candidate_generator.extract_disease_candidates(request.retrieval_result)

    matched_candidates = []
    for cand in raw_candidates:
        # Match evidence text
        ev_text = " ".join([c.chunk.text for c in request.retrieval_result.retrieved_chunks if c.chunk.chunk_id == cand["chunk_id"]])
        if not ev_text and request.retrieval_result.retrieved_chunks:
            ev_text = request.retrieval_result.retrieved_chunks[0].chunk.text

        match_res = evidence_matcher.match_patient_state(cand, request.state, ev_text)
        matched_candidates.append(match_res)

    # 2. Sort by tier rank
    tier_order = {
        "Most Compatible": 1,
        "Compatible": 2,
        "Possible": 3,
        "Less Compatible": 4,
        "Currently Unlikely": 5
    }
    sorted_candidates = sorted(matched_candidates, key=lambda x: tier_order.get(x.compatibility_tier.value, 99))

    # 3. Track Evolution
    deltas = evolution_tracker.track_ranking_evolution(sorted_candidates, request.state.previous_ranked_candidates)

    return ReasoningResult(
        candidates=sorted_candidates,
        grouped_evidence={},
        evolution_deltas=deltas
    )
