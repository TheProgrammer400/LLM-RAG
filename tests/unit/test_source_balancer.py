import pytest
from shared.models.schemas import RetrievedChunk, Chunk, ChunkMetadata, AuthorityTier, ContentType
from services.retrieval_service.source_balancer import SourceDiversityBalancer

def test_source_balancer_hard_cap_adherence():
    balancer = SourceDiversityBalancer(max_percentage_per_source=0.40)  # max 40%

    chunks = []
    # Create 10 chunks from source_A and 10 from source_B
    for i in range(10):
        meta = ChunkMetadata(
            chunk_id=f"c_a_{i}", parent_id="p1", source_id="SRC_A", source_title="Source A",
            page_start=1, page_end=1, heading="Heading", section_type="general",
            authority_tier=AuthorityTier.GUIDELINE, publication_year=2023,
            content_type=ContentType.CLINICAL_REFERENCE, embedding_model_version="v1"
        )
        chunks.append(RetrievedChunk(chunk=Chunk(chunk_id=f"c_a_{i}", text="Text A", metadata=meta), rerank_score=0.9 - (i*0.01)))

    for i in range(10):
        meta = ChunkMetadata(
            chunk_id=f"c_b_{i}", parent_id="p2", source_id="SRC_B", source_title="Source B",
            page_start=1, page_end=1, heading="Heading", section_type="general",
            authority_tier=AuthorityTier.SPECIALTY_TEXTBOOK, publication_year=2023,
            content_type=ContentType.CLINICAL_REFERENCE, embedding_model_version="v1"
        )
        chunks.append(RetrievedChunk(chunk=Chunk(chunk_id=f"c_b_{i}", text="Text B", metadata=meta), rerank_score=0.85 - (i*0.01)))

    balanced = balancer.balance_sources(chunks, target_count=10)

    src_a_count = sum(1 for c in balanced if c.chunk.metadata.source_id == "SRC_A")
    src_b_count = sum(1 for c in balanced if c.chunk.metadata.source_id == "SRC_B")

    # Max allowed per source out of 10 at 40% cap is 4
    assert src_a_count <= 4
    assert src_b_count <= 4
