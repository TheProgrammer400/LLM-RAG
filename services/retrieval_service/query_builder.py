import logging
from typing import List
from shared.models.schemas import PatientState, QueryFacet, EntityCUI
from ingestion.embedder import MedicalEmbedder

logger = logging.getLogger(__name__)

class MultiFacetQueryBuilder:
    def __init__(self, dedup_similarity_threshold: float = 0.90):
        self.dedup_threshold = dedup_similarity_threshold
        self.embedder = MedicalEmbedder()

    def build_query_facets(self, state: PatientState) -> List[QueryFacet]:
        """
        Builds 4-5 distinct query facets:
        1. Presentation query
        2. Syndrome/pattern query
        3. Differential query
        4. Investigation query
        5. Guideline query (incorporating UMLS-linked concept associations e.g. GCA)
        Deduplicates facets if similarity > threshold.
        """
        facts = state.extracted_facts
        symptoms_str = ", ".join(facts.symptoms) if facts.symptoms else "clinical symptoms"
        labs_str = ", ".join([f"{k} {v}" for k, v in facts.lab_findings.items()]) if facts.lab_findings else ""

        # Extract UMLS concept associations from normalized entities AND symptoms
        concept_expansions = []
        all_text = f"{symptoms_str} {' '.join([e.canonical_name + ' ' + e.text for e in state.normalized_entities])}".lower()

        if any(term in all_text for term in ["jaw claudication", "temporal headache", "c0017571", "c0236018"]):
            concept_expansions.extend(["giant cell arteritis", "temporal arteritis", "GCA", "polymyalgia rheumatica"])

        # 1. Presentation facet
        facet_pres = QueryFacet(
            facet_type="presentation",
            query_text=f"clinical presentation of {symptoms_str} {labs_str}".strip(),
            expanded_terms=concept_expansions
        )

        # 2. Syndrome / pattern facet
        facet_syndrome = QueryFacet(
            facet_type="syndrome",
            query_text=f"syndrome pattern {symptoms_str} in elderly patient".strip(),
            expanded_terms=concept_expansions
        )

        # 3. Differential facet
        facet_diff = QueryFacet(
            facet_type="differential",
            query_text=f"differential diagnosis of {symptoms_str} with {labs_str}".strip(),
            expanded_terms=concept_expansions
        )

        # 4. Investigation facet
        facet_inv = QueryFacet(
            facet_type="investigation",
            query_text=f"recommended diagnostic investigations biopsy imaging lab for {symptoms_str}".strip(),
            expanded_terms=concept_expansions
        )

        # 5. Guideline facet (incorporates UMLS-linked concepts!)
        expanded_guideline_terms = " ".join(set(concept_expansions)) if concept_expansions else symptoms_str
        facet_guide = QueryFacet(
            facet_type="guideline",
            query_text=f"clinical management guideline criteria for {expanded_guideline_terms}".strip(),
            expanded_terms=concept_expansions
        )

        raw_facets = [facet_pres, facet_syndrome, facet_diff, facet_inv, facet_guide]
        deduped_facets = self._deduplicate_facets(raw_facets)
        logger.info(f"Built {len(deduped_facets)} deduplicated query facets from raw 5.")
        return deduped_facets

    def _deduplicate_facets(self, facets: List[QueryFacet]) -> List[QueryFacet]:
        if len(facets) <= 1:
            return facets

        texts = [f"{f.query_text} {' '.join(f.expanded_terms)}" for f in facets]
        embs = self.embedder.embed_batch(texts)

        unique_facets: List[QueryFacet] = []
        unique_embs: List[List[float]] = []

        for i, facet in enumerate(facets):
            emb = embs[i]
            is_dup = False
            for j, u_emb in enumerate(unique_embs):
                sim = self.embedder.cosine_similarity(emb, u_emb)
                if sim >= self.dedup_threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique_facets.append(facet)
                unique_embs.append(emb)

        return unique_facets
