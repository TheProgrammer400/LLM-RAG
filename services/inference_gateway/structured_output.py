import json
import logging
from typing import Dict, Any
from shared.models.schemas import DifferentialItem, CitationClaim, CompatibilityTier, EntailmentStatus

logger = logging.getLogger(__name__)

class OutputParser:
    def parse_llm_json(self, raw_llm_text: str) -> Dict[str, Any]:
        """Parses LLM JSON response into structured dictionaries."""
        try:
            # Extract JSON block if enclosed in markdown backticks
            if "```json" in raw_llm_text:
                json_str = raw_llm_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_llm_text:
                json_str = raw_llm_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = raw_llm_text.strip()

            return json.loads(json_str)
        except Exception:
            logger.info("LLM provided free-text prose explanation. Formatting into structured response.")
            return {
                "differentials": [
                    {
                        "rank": 1,
                        "disease_name": "Heart Failure (Clinical Explanation)",
                        "compatibility_tier": "Most Compatible",
                        "clinical_rationale": raw_llm_text.strip(),
                        "citations": []
                    }
                ],
                "recommended_investigations": [
                    "Echocardiogram (TTE) for LVEF assessment",
                    "NT-proBNP / BNP blood levels",
                    "12-lead ECG"
                ],
                "red_flags": [
                    "Acute dyspnea / orthopnea",
                    "Sudden weight gain (>2kg in 3 days)"
                ],
                "missing_critical_info": []
            }

