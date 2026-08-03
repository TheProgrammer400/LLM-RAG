import pytest
from shared.models.schemas import PatientState, FactExtraction, EntityCUI
from services.retrieval_service.query_builder import MultiFacetQueryBuilder

def test_query_builder_facets_and_umls_expansion():
    builder = MultiFacetQueryBuilder(dedup_similarity_threshold=0.98)
    
    facts = FactExtraction(symptoms=["jaw claudication", "temporal headache"], lab_findings={"ESR": "110"})
    cuis = [EntityCUI(text="jaw claudication", cui="C0236018", canonical_name="Jaw Claudication")]
    
    state = PatientState(session_id="test_qb", extracted_facts=facts, normalized_entities=cuis)
    facets = builder.build_query_facets(state)

    assert len(facets) >= 2
    # Ensure UMLS concept expansion ('giant cell arteritis') is attached to facets
    all_expanded_terms = [term for f in facets for term in f.expanded_terms]
    assert any("giant cell arteritis" in t.lower() for t in all_expanded_terms)
