from fastapi import FastAPI
from pydantic import BaseModel
from shared.models.schemas import PatientState, ReasoningResult, RetrievalResult, DifferentialItem, CitationClaim, CompatibilityTier, EntailmentStatus
from services.inference_gateway.prompt_builder import PromptBuilder
from services.inference_gateway.llm_client import LLMClient
from services.inference_gateway.self_verifier import NLIEntailmentVerifier
from services.inference_gateway.structured_output import OutputParser

app = FastAPI(title="CDSS Inference Gateway", version="1.0.0")

prompt_builder = PromptBuilder()
llm_client = LLMClient()
verifier = NLIEntailmentVerifier()
parser = OutputParser()

class GenerateRequest(BaseModel):
    state: PatientState
    reasoning_result: ReasoningResult
    retrieval_result: RetrievalResult
    escalate: bool = False

class GenerateResponse(BaseModel):
    differentials: list
    recommended_investigations: list
    red_flags: list
    missing_critical_info: list

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    sys_prompt = prompt_builder.build_system_prompt(request.state, request.reasoning_result, request.retrieval_result)
    raw_llm_out = await llm_client.generate_response(sys_prompt, "Generate clinical reasoning response.", escalate=request.escalate)
    parsed = parser.parse_llm_json(raw_llm_out)

    # Self-verification pass on citations
    raw_diffs = parsed.get("differentials", [])
    processed_diffs = []

    for d in raw_diffs:
        raw_cites = d.get("citations", [])
        citations_obj = [CitationClaim(**c) for c in raw_cites]
        verified_cites = verifier.verify_citations(citations_obj, request.retrieval_result)

        processed_diffs.append(DifferentialItem(
            rank=d.get("rank", 1),
            disease_name=d.get("disease_name", "Unknown"),
            compatibility_tier=CompatibilityTier(d.get("compatibility_tier", "Possible")),
            clinical_rationale=d.get("clinical_rationale", ""),
            citations=verified_cites
        ))

    return GenerateResponse(
        differentials=processed_diffs,
        recommended_investigations=parsed.get("recommended_investigations", []),
        red_flags=parsed.get("red_flags", []),
        missing_critical_info=parsed.get("missing_critical_info", [])
    )
