import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AuditLogger:
    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    async def log_turn(self, session_id: str, physician_input: str, state: Any, response: Any, short_circuited: bool = False):
        """Logs structured turn trace for governance and auditability."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "physician_input": physician_input,
            "turn_number": state.turn_number if hasattr(state, "turn_number") else 1,
            "short_circuited": short_circuited,
            "response": response.model_dump() if hasattr(response, "model_dump") else str(response)
        }

        filename = os.path.join(self.log_dir, f"{session_id}.jsonl")
        with open(filename, "a") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")

        logger.info(f"Audit log recorded for session {session_id}.")
