import re
from shared.models.schemas import ContentType

class ContentClassifier:
    def classify_section(self, heading: str, body_text: str) -> ContentType:
        """
        Classifies section content into clinical_reference, patient_education, or administrative.
        Uses rule-based heuristics and medical density indicators.
        """
        combined = f"{heading} {body_text}".lower()

        # Administrative / Copyright indicators
        admin_keywords = ["isbn", "all rights reserved", "printed in", "cataloging-in-publication", "disclaimer:", "copyright ©"]
        if any(kw in combined for kw in admin_keywords):
            return ContentType.ADMINISTRATIVE

        # Patient Education indicators
        patient_keywords = ["patient guide", "what you should know", "questions to ask your doctor", "caring for yourself at home"]
        if any(kw in combined for kw in patient_keywords):
            return ContentType.PATIENT_EDUCATION

        # Case report indicators
        case_report_keywords = ["case report", "a 45-year-old male presented with", "we report a rare case"]
        if any(kw in combined for kw in case_report_keywords):
            return ContentType.CASE_REPORTS

        # Default clinical reference
        return ContentType.CLINICAL_REFERENCE
