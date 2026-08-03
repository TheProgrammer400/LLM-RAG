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
        except Exception as e:
            logger.error(f"Failed to parse LLM JSON output ({e}). Raw text snippet: {raw_llm_text[:100]}")
            return {
                "differentials": [],
                "recommended_investigations": [],
                "red_flags": [],
                "missing_critical_info": []
            }
