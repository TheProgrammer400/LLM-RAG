from typing import List
from shared.models.schemas import FactExtraction, EntityCUI
from ingestion.entity_extractor import MedicalEntityExtractor

class EntityNormalizer:
    def __init__(self):
        self.extractor = MedicalEntityExtractor()

    def normalize_extracted_facts(self, facts: FactExtraction) -> List[EntityCUI]:
        """Normalizes extracted symptoms and findings into canonical UMLS CUIs."""
        combined_text = " ".join(facts.symptoms + list(facts.lab_findings.keys()) + facts.negative_findings)
        extracted = self.extractor.extract_entities(combined_text)
        
        normalized: List[EntityCUI] = []
        seen_cuis = set()

        for category in ["diseases", "symptoms", "drugs"]:
            for entity in extracted[category]:
                if entity.cui not in seen_cuis:
                    seen_cuis.add(entity.cui)
                    normalized.append(entity)

        return normalized
