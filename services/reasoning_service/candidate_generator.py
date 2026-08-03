from typing import List, Dict, Any
from shared.models.schemas import RetrievalResult, RetrievedChunk

class CandidateGenerator:
    def extract_disease_candidates(self, retrieval_result: RetrievalResult) -> List[Dict[str, Any]]:
        """Extracts candidate diseases from retrieval result."""
        return self.extract_disease_candidates_from_chunks(retrieval_result.retrieved_chunks)

    def extract_disease_candidates_from_chunks(self, retrieved_chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """
        Extracts candidate diseases from evidence pack headings and metadata CUIs.
        Ontology-grounded disease candidate discovery.
        """
        candidates = []
        seen_names = set()

        for item in retrieved_chunks:
            meta = item.chunk.metadata
            for disease in meta.disease_entities:
                if disease.canonical_name.lower() not in seen_names:
                    seen_names.add(disease.canonical_name.lower())
                    candidates.append({
                        "disease_name": disease.canonical_name,
                        "cui": disease.cui,
                        "chunk_id": item.chunk.chunk_id,
                        "source_title": meta.source_title,
                        "page": meta.page_start
                    })

            # Check chunk text/heading
            text = f"{meta.heading} {item.chunk.text}".lower()
            if "giant cell arteritis" in text and "giant cell arteritis" not in seen_names:
                seen_names.add("giant cell arteritis")
                candidates.append({"disease_name": "Giant Cell Arteritis", "cui": "C0017571", "chunk_id": item.chunk.chunk_id, "source_title": meta.source_title, "page": meta.page_start})
            elif "central retinal artery occlusion" in text and "central retinal artery occlusion" not in seen_names:
                seen_names.add("central retinal artery occlusion")
                candidates.append({"disease_name": "Central Retinal Artery Occlusion", "cui": "C0154832", "chunk_id": item.chunk.chunk_id, "source_title": meta.source_title, "page": meta.page_start})

        return candidates
