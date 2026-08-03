import json
import logging
from typing import Dict, Optional
from shared.models.schemas import PatientState

logger = logging.getLogger(__name__)

# In-memory session store fallback (for single-node / dev) backed by Redis/Postgres
SESSION_STORE: Dict[str, PatientState] = {}

class ConsultationStateStore:
    async def get_state(self, session_id: str) -> PatientState:
        if session_id in SESSION_STORE:
            return SESSION_STORE[session_id]
        new_state = PatientState(session_id=session_id)
        SESSION_STORE[session_id] = new_state
        return new_state

    async def save_state(self, session_id: str, state: PatientState):
        SESSION_STORE[session_id] = state
        logger.info(f"Saved consultation state for session {session_id} (Turn {state.turn_number}).")
