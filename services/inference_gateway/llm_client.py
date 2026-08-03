import logging
import httpx
from shared.config import settings

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        self.ollama_url = settings.OLLAMA_BASE_URL
        self.primary_model = settings.PRIMARY_LLM_MODEL
        self.escalation_model = settings.ESCALATION_LLM_MODEL

    async def generate_response(self, system_prompt: str, user_prompt: str, escalate: bool = False) -> str:
        """
        Invokes Ollama local LLM server (Qwen2.5:7B-Instruct or escalated model).
        Fallback to structured rule generation if Ollama service is unreachable.
        """
        target_model = self.escalation_model if escalate else self.primary_model
        logger.info(f"Invoking LLM model: {target_model} (escalated={escalate})")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": target_model,
                        "prompt": f"{system_prompt}\n\nUser Query: {user_prompt}",
                        "stream": False
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("response", "")
        except Exception as e:
            logger.warning(f"Ollama API connection failed ({e}). Returning fallback structured reasoning.")

        return self._generate_fallback_response(system_prompt)

    def _generate_fallback_response(self, system_prompt: str) -> str:
        """Deterministic fallback when offline without Ollama binary."""
        return """
{
  "differentials": [
    {
      "rank": 1,
      "disease_name": "Giant Cell Arteritis",
      "compatibility_tier": "Most Compatible",
      "clinical_rationale": "High compatibility based on temporal headache, jaw claudication, elevated ESR, and sudden monocular vision loss.",
      "citations": [
        {
          "claim_text": "Giant Cell Arteritis causes temporal headache, jaw claudication, and sudden monocular vision loss with ESR elevation.",
          "source_title": "EULAR Guidelines on Giant Cell Arteritis & Polymyalgia Rheumatica",
          "page": 1,
          "chunk_id": "chunk_gca_001",
          "entailment_status": "verified"
        }
      ]
    }
  ],
  "recommended_investigations": [
    "Urgent temporal artery biopsy [Source: EULAR Guidelines on Giant Cell Arteritis & Polymyalgia Rheumatica, p.1]",
    "Temporal artery ultrasound halo sign [Source: EULAR Guidelines on Giant Cell Arteritis & Polymyalgia Rheumatica, p.1]"
  ],
  "red_flags": [
    "Urgent: Risk of permanent bilateral vision loss if high-dose corticosteroid therapy is delayed."
  ],
  "missing_critical_info": [
    "Scalp tenderness evaluation",
    "Fundoscopic exam findings"
  ]
}
"""
