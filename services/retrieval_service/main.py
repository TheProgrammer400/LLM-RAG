import asyncio
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from shared.models.schemas import PatientState, RetrievalResult, ConfidenceGateOutcome
from services.retrieval_service.query_builder import MultiFacetQueryBuilder
from services.retrieval_service.dense_retriever import DenseRetriever
from services.retrieval_service.sparse_retriever import SparseRetriever
from services.retrieval_service.fusion import ReciprocalRankFusion
from services.retrieval_service.dedup import CandidateDeduplicator
from services.retrieval_service.reranker import CrossEncoderReranker
from services.retrieval_service.source_balancer import SourceDiversityBalancer
from services.retrieval_service.context_expander import ContextExpander
from services.retrieval_service.confidence_gate import RetrievalConfidenceGate
from services.retrieval_service.semantic_cache import SemanticCache
from ingestion.embedder import MedicalEmbedder

logger = logging.getLogger(__name__)
app = FastAPI(title="CDSS Retrieval Service", version="1.0.0")

query_builder = MultiFacetQueryBuilder()
embedder = MedicalEmbedder()
dense_retriever = DenseRetriever()
sparse_retriever = SparseRetriever()
rrf_fusion = ReciprocalRankFusion()
candidate_deduper = CandidateDeduplicator()
reranker = CrossEncoderReranker()
source_balancer = SourceDiversityBalancer()
context_expander = ContextExpander()
confidence_gate = RetrievalConfidenceGate()
semantic_cache = SemanticCache()

class RetrieveRequest(BaseModel):
    state: PatientState

@app.post("/retrieve", response_model=RetrievalResult)
async def retrieve_evidence(request: RetrieveRequest):
    # 1. Build deduplicated query facets
    facets = query_builder.build_query_facets(request.state)
    primary_query = facets[0].query_text if facets else "clinical presentation"

    # 2. Parallel Dense + Sparse Retrieval across all facets
    async def retrieve_facet_hits(facet):
        dense_hits = await dense_retriever.search_facet(facet.query_text, embedder, top_k=50)
        sparse_hits = await sparse_retriever.search_facet(facet.query_text, top_k=50)
        return dense_hits + sparse_hits

    facet_hit_lists = await asyncio.gather(*[retrieve_facet_hits(f) for f in facets])

    # 3. Reciprocal Rank Fusion (RRF)
    fused_candidates = rrf_fusion.fuse_results(facet_hit_lists)

    # 4. Pre-reranking Candidate Deduplication
    deduped_candidates = candidate_deduper.deduplicate_candidates(fused_candidates)

    # 5. Cross-Encoder Reranking
    reranked_chunks = reranker.rerank(primary_query, deduped_candidates, top_n=20)

    # 6. Source Diversity Balancing (Hard Cap Max 40% per source applied last)
    balanced_chunks = source_balancer.balance_sources(reranked_chunks, target_count=15)

    # 7. Parent-Section Context Expansion
    expanded_chunks = context_expander.expand_chunks_with_parent_context(balanced_chunks)

    # 8. Confidence Gate
    outcome, top_score, count_above_floor = confidence_gate.evaluate_confidence(expanded_chunks, is_retry=False)

    if outcome == ConfidenceGateOutcome.RETRY:
        logger.info("Confidence gate returned RETRY. Running query broadening retry pass...")
        broadened_facet = facets[0].query_text + " " + " ".join(facets[0].expanded_terms)
        retry_dense = await dense_retriever.search_facet(broadened_facet, embedder, top_k=50)
        retry_sparse = await sparse_retriever.search_facet(broadened_facet, top_k=50)
        retry_fused = rrf_fusion.fuse_results([fused_candidates + retry_dense + retry_sparse])
        retry_reranked = reranker.rerank(primary_query, retry_fused, top_n=20)
        expanded_chunks = source_balancer.balance_sources(retry_reranked, target_count=15)
        outcome, top_score, count_above_floor = confidence_gate.evaluate_confidence(expanded_chunks, is_retry=True)

    return RetrievalResult(
        facets_used=facets,
        retrieved_chunks=expanded_chunks,
        confidence=outcome,
        top_rerank_score=top_score,
        n_chunks_above_floor=count_above_floor,
        cache_hit=False
    )
