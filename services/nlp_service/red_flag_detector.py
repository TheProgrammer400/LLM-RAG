import logging
from typing import Tuple, List
from shared.models.schemas import PatientState, SeverityLevel

logger = logging.getLogger(__name__)

class RedFlagDetector:
    def assess_severity_and_red_flags(self, state: PatientState) -> Tuple[SeverityLevel, List[str]]:
        """
        Evaluates red flags against negation-resolved PatientState.
        CRITICAL: Negated findings (e.g. 'denies chest pain') must NEVER trigger a red flag!
        """
        facts = state.extracted_facts
        symptoms = [s.lower() for s in facts.symptoms]
        negatives = [n.lower() for n in facts.negative_findings]
        labs = facts.lab_findings

        red_flags = []
        severity = SeverityLevel.ROUTINE

        # 1. Anaphylaxis emergency pattern
        if "anaphylaxis" in symptoms or ("stridor" in symptoms and "urticaria" in symptoms):
            if "anaphylaxis" not in negatives:
                red_flags.append("Emergency: Acute Anaphylaxis / Airway Compromise")
                return SeverityLevel.EMERGENCY, red_flags

        # 2. Acute neuro deficit / Stroke pattern
        if ("hemiparesis" in symptoms or "slurred speech" in symptoms or "facial droop" in symptoms):
            if not any(s in negatives for s in ["hemiparesis", "slurred speech", "stroke"]):
                red_flags.append("Emergency: Acute Focal Neurological Deficit (Stroke Protocol)")
                return SeverityLevel.EMERGENCY, red_flags

        # 3. Severe cardiac risk / STEMI pattern
        if ("chest pain" in symptoms and ("diaphoresis" in symptoms or "st elevation" in symptoms)):
            if "chest pain" not in negatives:
                red_flags.append("Emergency: Acute Coronary Syndrome / STEMI pattern")
                return SeverityLevel.EMERGENCY, red_flags

        # 4. Urgent Ophthalmologic / Vasculitis pattern (GCA vision loss)
        if ("vision loss" in symptoms or "monocular vision loss" in symptoms or "painless vision loss" in symptoms):
            if not any("vision loss" in neg for neg in negatives):
                if "jaw claudication" in symptoms or "temporal headache" in symptoms or "ESR" in labs:
                    red_flags.append("Urgent: High risk Giant Cell Arteritis with acute visual loss threat")
                    severity = SeverityLevel.URGENT
                else:
                    red_flags.append("Urgent: Acute Monocular Vision Loss requiring immediate ophthalmology evaluation")
                    severity = SeverityLevel.URGENT

        # 5. Meningitis pattern
        if ("neck stiffness" in symptoms and "fever" in symptoms):
            if "neck stiffness" not in negatives:
                red_flags.append("Urgent: Meningeal signs with fever")
                severity = SeverityLevel.URGENT

        return severity, red_flags
