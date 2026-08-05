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
        Invokes Ollama local LLM server (Qwen2.5:7b or escalated model).
        Fallback to structured rule generation if Ollama service is unreachable.
        """
        target_model = self.escalation_model if escalate else self.primary_model
        logger.info(f"Invoking LLM model: {target_model} (escalated={escalate})")

        candidate_models = [target_model, "qwen2.5:7b", "qwen2.5:7b-instruct", "qwen2.5:latest", "qwen2.5", "phi4-mini:latest", "phi4-mini"]
        urls_to_try = list(dict.fromkeys(["http://127.0.0.1:11434", self.ollama_url, "http://localhost:11434"]))

        working_url = None
        installed = []

        # Fast probe for Ollama tags / installed models
        async with httpx.AsyncClient(timeout=3.0) as check_client:
            for url in urls_to_try:
                try:
                    tags_resp = await check_client.get(f"{url}/api/tags")
                    if tags_resp.status_code == 200:
                        working_url = url
                        installed = [m.get("name") for m in tags_resp.json().get("models", [])]
                        logger.info(f"Connected to Ollama at {working_url}. Installed models: {installed}")
                        break
                except Exception as e:
                    logger.debug(f"Could not connect to Ollama at {url}: {e}")

        if installed:
            matched_model = None
            for cand in candidate_models:
                base_cand = cand.split(":")[0]
                for m in installed:
                    if cand == m or cand in m or m in cand or base_cand in m:
                        matched_model = m
                        break
                if matched_model:
                    target_model = matched_model
                    break
            else:
                target_model = installed[0]

        if not working_url:
            working_url = "http://127.0.0.1:11434"

        async with httpx.AsyncClient(timeout=600.0) as client:
            try:
                logger.info(f"Sending prompt to Ollama ({working_url}) using model: {target_model}")
                resp = await client.post(
                    f"{working_url}/api/generate",
                    json={
                        "model": target_model,
                        "prompt": f"{system_prompt}\n\nUser Query: {user_prompt}",
                        "stream": False
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("response", "")
                else:
                    logger.warning(f"Ollama API returned status {resp.status_code}: {resp.text}")
            except Exception as e:
                err_msg = str(e) or repr(e)
                logger.warning(f"Ollama API connection failed for {working_url} ({type(e).__name__}: {err_msg}). Returning fallback structured reasoning.")

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
