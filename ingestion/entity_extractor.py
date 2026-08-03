import logging
from typing import List, Dict, Tuple
from shared.models.schemas import EntityCUI

logger = logging.getLogger(__name__)

# Essential pre-mapped UMLS dictionary for core clinical concepts
CORE_UMLS_DICTIONARY: Dict[str, Tuple[str, str, str]] = {
    "giant cell arteritis": ("C0017571", "Giant Cell Arteritis", "Disease"),
    "temporal arteritis": ("C0017571", "Giant Cell Arteritis", "Disease"),
    "gca": ("C0017571", "Giant Cell Arteritis", "Disease"),
    "jaw claudication": ("C0236018", "Jaw Claudication", "Symptom"),
    "jaw pain while chewing": ("C0236018", "Jaw Claudication", "Symptom"),
    "temporal headache": ("C0239849", "Temporal Headache", "Symptom"),
    "headache": ("C0018681", "Headache", "Symptom"),
    "vision loss": ("C0042798", "Vision Loss", "Symptom"),
    "painless vision loss": ("C0042798", "Vision Loss", "Symptom"),
    "monocular vision loss": ("C0234674", "Amaurosis Fugax", "Symptom"),
    "amaurosis fugax": ("C0234674", "Amaurosis Fugax", "Symptom"),
    "polymyalgia rheumatica": ("C0032547", "Polymyalgia Rheumatica", "Disease"),
    "pmr": ("C0032547", "Polymyalgia Rheumatica", "Disease"),
    "esr": ("C0015134", "Erythrocyte Sedimentation Rate", "Lab"),
    "erythrocyte sedimentation rate": ("C0015134", "Erythrocyte Sedimentation Rate", "Lab"),
    "crp": ("C0009801", "C-Reactive Protein", "Lab"),
    "c-reactive protein": ("C0009801", "C-Reactive Protein", "Lab"),
    "prednisone": ("C0033116", "Prednisone", "Drug"),
    "corticosteroids": ("C0001617", "Adrenal Cortex Hormones", "Drug"),
    "temporal artery biopsy": ("C0162590", "Biopsy of Temporal Artery", "Investigation"),
    "central retinal artery occlusion": ("C0154832", "Central Retinal Artery Occlusion", "Disease"),
    "crao": ("C0154832", "Central Retinal Artery Occlusion", "Disease"),
}

class MedicalEntityExtractor:
    def __init__(self):
        self.nlp = None
        self._initialize_spacy()

    def _initialize_spacy(self):
        """Attempts to load scispaCy / medspaCy pipeline if installed."""
        try:
            import spacy
            self.nlp = spacy.load("en_core_sci_sm")
            logger.info("Loaded scispaCy model: en_core_sci_sm")
        except Exception as e:
            logger.warning(f"scispaCy model not available ({e}). Using dictionary-based UMLS extractor.")

    def extract_entities(self, text: str) -> Dict[str, List[EntityCUI]]:
        """
        Extracts diseases, symptoms, and drug entities linked to UMLS CUIs.
        Returns a dict keyed by category: 'diseases', 'symptoms', 'drugs'.
        """
        text_lower = text.lower()
        diseases: List[EntityCUI] = []
        symptoms: List[EntityCUI] = []
        drugs: List[EntityCUI] = []

        seen_cuis = set()

        # Dictionary-based lookup for guaranteed core matching
        for term, (cui, canonical, category) in CORE_UMLS_DICTIONARY.items():
            if term in text_lower and cui not in seen_cuis:
                seen_cuis.add(cui)
                entity = EntityCUI(text=term, cui=cui, canonical_name=canonical, semantic_type=category)
                if category == "Disease":
                    diseases.append(entity)
                elif category == "Symptom":
                    symptoms.append(entity)
                elif category == "Drug":
                    drugs.append(entity)

        # spaCy extraction pass if model is loaded
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                ent_text_lower = ent.text.lower()
                if ent_text_lower in CORE_UMLS_DICTIONARY:
                    cui, canonical, category = CORE_UMLS_DICTIONARY[ent_text_lower]
                    if cui not in seen_cuis:
                        seen_cuis.add(cui)
                        entity = EntityCUI(text=ent.text, cui=cui, canonical_name=canonical, semantic_type=category)
                        if category == "Disease":
                            diseases.append(entity)
                        elif category == "Symptom":
                            symptoms.append(entity)
                        elif category == "Drug":
                            drugs.append(entity)

        return {
            "diseases": diseases,
            "symptoms": symptoms,
            "drugs": drugs
        }
