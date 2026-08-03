from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from shared.models.schemas import ConsultationResponse
from services.orchestrator.flow import OrchestratorFlow

app = FastAPI(
    title="Clinical Decision Support System (CDSS) Orchestrator API",
    version="1.0.0",
    description="Physician-facing clinical decision support API with evidence confidence gating and 100% citation traceability."
)

orchestrator_flow = OrchestratorFlow()

class ConsultationTurnRequest(BaseModel):
    session_id: str
    physician_input: str

@app.get("/")
async def root():
    return {"system": "CDSS Orchestrator", "status": "online"}

@app.post("/api/v1/consultation/turn", response_model=ConsultationResponse)
async def consultation_turn(request: ConsultationTurnRequest):
    if not request.physician_input.strip():
        raise HTTPException(status_code=400, detail="Physician input cannot be empty.")
    
    try:
        response = await orchestrator_flow.handle_consultation_turn(request.session_id, request.physician_input)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CDSS turn execution error: {str(e)}")
