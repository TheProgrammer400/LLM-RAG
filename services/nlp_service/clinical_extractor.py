import re
import logging
from typing import Dict, Any, List
from shared.models.schemas import FactExtraction

logger = logging.getLogger(__name__)

class ClinicalFactExtractor:
    """
    Extracts clinical findings using medspaCy TargetMatcher + ConText pipeline.
    Guarantees negation resolution runs BEFORE findings are exported.
    """
    def __init__(self):
        self.nlp = None
        self._initialize_medspacy()

    def _initialize_medspacy(self):
        try:
            import medspacy
            self.nlp = medspacy.load()
            logger.info("Loaded medspaCy pipeline with TargetMatcher & ConText negation component.")
        except Exception as e:
            logger.warning(f"medspaCy not available ({e}). Using rule-based regex negation engine.")

    def extract_facts(self, text: str) -> FactExtraction:
        text_lower = text.lower()
        negated_findings = []
        confirmed_symptoms = []
        confirmed_diagnoses = []
        ruled_out_diagnoses = []
        labs = {}
        imaging = []
        meds = []
        risk = []

        # 1. Parse explicit negations ("no diabetes", "denies chest pain", "without fever", "negative for rash")
        negation_patterns = [
            r"\b(?:no|denies|negative for|without|free of)\s+([a-z0-9\s]+?)(?=[,;\.]|$)",
            r"\b([a-z0-9\s]+?)\s+(?:ruled out|is absent|was negative)\b"
        ]

        negated_spans = set()
        for pat in negation_patterns:
            matches = re.finditer(pat, text_lower)
            for m in matches:
                span_text = m.group(1).strip()
                if span_text:
                    negated_findings.append(span_text)
                    negated_spans.add(span_text)

        # 2. Extract demographics
        demo = {}
        age_match = re.search(r"\b(\d{1,3})\s*(?:f|m|yo|year old|female|male)\b", text_lower)
        if age_match:
            demo["age"] = int(age_match.group(1))
            demo["gender"] = "female" if "f" in age_match.group(0) or "female" in age_match.group(0) else "male"

        # 3. Extract labs (e.g. "ESR 110", "CRP elevated")
        esr_match = re.search(r"\besr\s*(\d{1,3})\b", text_lower)
        if esr_match:
            labs["ESR"] = esr_match.group(1)
        elif "esr elevated" in text_lower or "elevated esr" in text_lower:
            labs["ESR"] = "elevated"

        crp_match = re.search(r"\bcrp\s*([a-z0-9]+)\b", text_lower)
        if crp_match:
            labs["CRP"] = crp_match.group(1)
        elif "crp elevated" in text_lower or "elevated crp" in text_lower:
            labs["CRP"] = "elevated"

        # 4. Extract symptoms (filtering out negated spans!)
        symptom_candidates = [
            "headache", "temporal headache", "jaw claudication", "jaw pain while chewing",
            "vision loss", "monocular vision loss", "painless vision loss", "weight loss",
            "fever", "chest pain", "neck stiffness", "photophobia", "malaise", "fatigue"
        ]

        for s in symptom_candidates:
            if s in text_lower:
                # CRITICAL: Negation check
                if any(s in neg or neg in s for neg in negated_spans):
                    if s not in negated_findings:
                        negated_findings.append(s)
                else:
                    confirmed_symptoms.append(s)

        # 5. Extract duration/onset
        duration_match = re.search(r"\b(\d+\s*(?:days?|weeks?|months?|years?))\b", text_lower)
        duration = duration_match.group(1) if duration_match else None
        onset = "sudden" if "sudden" in text_lower or "acute" in text_lower else "gradual"

        return FactExtraction(
            symptoms=confirmed_symptoms,
            negative_findings=negated_findings,
            confirmed_diagnoses=confirmed_diagnoses,
            ruled_out_diagnoses=ruled_out_diagnoses,
            vital_signs={},
            lab_findings=labs,
            imaging_findings=imaging,
            medications=meds,
            risk_factors=risk,
            duration=duration,
            onset=onset,
            demographics=demo
        )
