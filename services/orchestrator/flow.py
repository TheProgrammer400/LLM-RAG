import time
import logging
from typing import Dict
from shared.models.schemas import (
    ConsultationResponse, IntentType, SeverityLevel, TurnStrategy,
    ConfidenceGateOutcome, DifferentialItem, CitationClaim, CompatibilityTier, EntailmentStatus
)
from services.orchestrator.state_store import ConsultationStateStore
from services.orchestrator.safety_layer import OutputSafetyLayer
from services.orchestrator.audit_logger import AuditLogger

from services.nlp_service.intent_classifier import IntentClassifier
from services.nlp_service.clinical_extractor import ClinicalFactExtractor
from services.nlp_service.entity_normalizer import EntityNormalizer
from services.nlp_service.red_flag_detector import RedFlagDetector
from services.nlp_service.turn_planner import TurnPlanner

from services.retrieval_service.query_builder import MultiFacetQueryBuilder
from services.retrieval_service.dense_retriever import DenseRetriever
from services.retrieval_service.sparse_retriever import SparseRetriever
from services.retrieval_service.fusion import ReciprocalRankFusion
from services.retrieval_service.dedup import CandidateDeduplicator
from services.retrieval_service.reranker import CrossEncoderReranker
from services.retrieval_service.source_balancer import SourceDiversityBalancer
from services.retrieval_service.confidence_gate import RetrievalConfidenceGate

from services.reasoning_service.candidate_generator import CandidateGenerator
from services.reasoning_service.evidence_matcher import EvidenceMatcher
from services.reasoning_service.evolution_tracker import EvolutionTracker

from services.inference_gateway.prompt_builder import PromptBuilder
from services.inference_gateway.llm_client import LLMClient
from services.inference_gateway.self_verifier import NLIEntailmentVerifier
from services.inference_gateway.structured_output import OutputParser
from ingestion.embedder import MedicalEmbedder

logger = logging.getLogger("OrchestratorFlow")

class OrchestratorFlow:
    def __init__(self):
        self.state_store = ConsultationStateStore()
        self.safety_layer = OutputSafetyLayer()
        self.audit_logger = AuditLogger()

        self.intent_classifier = IntentClassifier()
        self.fact_extractor = ClinicalFactExtractor()
        self.entity_normalizer = EntityNormalizer()
        self.red_flag_detector = RedFlagDetector()
        self.turn_planner = TurnPlanner()

        self.query_builder = MultiFacetQueryBuilder()
        self.embedder = MedicalEmbedder()
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
        self.rrf_fusion = ReciprocalRankFusion()
        self.candidate_deduper = CandidateDeduplicator()
        self.reranker = CrossEncoderReranker()
        self.source_balancer = SourceDiversityBalancer()
        self.confidence_gate = RetrievalConfidenceGate()

        self.candidate_generator = CandidateGenerator()
        self.evidence_matcher = EvidenceMatcher()
        self.evolution_tracker = EvolutionTracker()

        self.prompt_builder = PromptBuilder()
        self.llm_client = LLMClient()
        self.verifier = NLIEntailmentVerifier()
        self.output_parser = OutputParser()

    async def handle_consultation_turn(self, session_id: str, physician_input: str) -> ConsultationResponse:
        t0 = time.time()
        latencies: Dict[str, float] = {}

        # 0. Load State
        state = await self.state_store.get_state(session_id)

        # 1. NLP Pipeline
        t_nlp = time.time()
        intent = self.intent_classifier.classify_intent(physician_input)
        if intent == IntentType.EXIT:
            return ConsultationResponse(
                session_id=session_id,
                turn_number=state.turn_number,
                intent=intent,
                severity=SeverityLevel.ROUTINE,
                turn_strategy=TurnStrategy.INFORMATION_GATHERING,
                retrieval_confidence=ConfidenceGateOutcome.SUFFICIENT
            )

        extracted = self.fact_extractor.extract_facts(physician_input)
        normalized = self.entity_normalizer.normalize_extracted_facts(extracted)

        # Update State
        state.extracted_facts = extracted
        state.normalized_entities = normalized

        severity, red_flags = self.red_flag_detector.assess_severity_and_red_flags(state)
        strategy = self.turn_planner.plan_turn(state)
        latencies["nlp_ms"] = round((time.time() - t_nlp) * 1000, 2)

        # EMERGENCY Short-Circuit Fast Path (<1s)
        if severity == SeverityLevel.EMERGENCY:
            emergency_resp = ConsultationResponse(
                session_id=session_id,
                turn_number=state.turn_number,
                intent=intent,
                severity=severity,
                turn_strategy=strategy,
                is_emergency=True,
                emergency_guidance="EMERGENCY ALERT: Immediate resuscitation / emergency intervention protocol triggered.",
                red_flags=red_flags,
                retrieval_confidence=ConfidenceGateOutcome.SUFFICIENT,
                latency_ms={"total_ms": round((time.time() - t0) * 1000, 2)}
            )
            await self.audit_logger.log_turn(session_id, physician_input, state, emergency_resp, short_circuited=True)
            return emergency_resp

        # 2. Retrieval Pipeline
        t_ret = time.time()
        facets = self.query_builder.build_query_facets(state)
        primary_query = facets[0].query_text if facets else physician_input

        dense_hits = await self.dense_retriever.search_facet(primary_query, self.embedder, top_k=50)
        sparse_hits = await self.sparse_retriever.search_facet(primary_query, top_k=50)

        fused = self.rrf_fusion.fuse_results([dense_hits + sparse_hits])
        deduped = self.candidate_deduper.deduplicate_candidates(fused)
        reranked = self.reranker.rerank(primary_query, deduped, top_n=20)
        balanced = self.source_balancer.balance_sources(reranked, target_count=15)

        outcome, top_score, count_above_floor = self.confidence_gate.evaluate_confidence(balanced)
        latencies["retrieval_ms"] = round((time.time() - t_ret) * 1000, 2)

        # Hard Confidence Gate Abstention Check
        if outcome == ConfidenceGateOutcome.INSUFFICIENT:
            gated_resp = ConsultationResponse(
                session_id=session_id,
                turn_number=state.turn_number,
                intent=intent,
                severity=severity,
                turn_strategy=strategy,
                red_flags=red_flags,
                retrieval_confidence=ConfidenceGateOutcome.INSUFFICIENT,
                missing_critical_info=["Insufficient evidence found in knowledge base to support high-confidence clinical reasoning."],
                latency_ms=latencies
            )
            await self.audit_logger.log_turn(session_id, physician_input, state, gated_resp)
            return gated_resp

        # 3. Reasoning Pipeline
        t_reason = time.time()
        raw_candidates = self.candidate_generator.extract_disease_candidates_from_chunks(balanced)
        matched_candidates = []
        for cand in raw_candidates:
            ev_text = " ".join([c.chunk.text for c in balanced if c.chunk.chunk_id == cand["chunk_id"]])
            match_res = self.evidence_matcher.match_patient_state(cand, state, ev_text or primary_query)
            matched_candidates.append(match_res)

        tier_order = {"Most Compatible": 1, "Compatible": 2, "Possible": 3, "Less Compatible": 4, "Currently Unlikely": 5}
        sorted_candidates = sorted(matched_candidates, key=lambda x: tier_order.get(x.compatibility_tier.value, 99))
        latencies["reasoning_ms"] = round((time.time() - t_reason) * 1000, 2)

        # 4. Inference Gateway & Self-Verification Pass
        t_inf = time.time()
        sys_prompt = self.prompt_builder.build_system_prompt(state, sorted_candidates, balanced)
        escalate = (outcome == ConfidenceGateOutcome.RETRY or severity == SeverityLevel.URGENT)
        raw_llm_out = await self.llm_client.generate_response(sys_prompt, physician_input, escalate=escalate)
        parsed_out = self.output_parser.parse_llm_json(raw_llm_out)

        differentials = []
        for d in parsed_out.get("differentials", []):
            raw_cites = d.get("citations", [])
            cite_objs = [CitationClaim(**c) for c in raw_cites]
            verified_cites = self.verifier.verify_citations(cite_objs, balanced)
            differentials.append(DifferentialItem(
                rank=d.get("rank", 1),
                disease_name=d.get("disease_name", "Unknown"),
                compatibility_tier=CompatibilityTier(d.get("compatibility_tier", "Possible")),
                clinical_rationale=d.get("clinical_rationale", ""),
                citations=verified_cites
            ))

        latencies["inference_ms"] = round((time.time() - t_inf) * 1000, 2)
        latencies["total_ms"] = round((time.time() - t0) * 1000, 2)

        response = ConsultationResponse(
            session_id=session_id,
            turn_number=state.turn_number,
            intent=intent,
            severity=severity,
            turn_strategy=strategy,
            differentials=differentials,
            recommended_investigations=parsed_out.get("recommended_investigations", []),
            red_flags=red_flags + parsed_out.get("red_flags", []),
            missing_critical_info=parsed_out.get("missing_critical_info", []),
            retrieval_confidence=outcome,
            latency_ms=latencies
        )

        # 5. Safety Layer & Persistence
        final_response = self.safety_layer.apply_safety_layer(response)
        state.turn_number += 1
        await self.state_store.save_state(session_id, state)
        await self.audit_logger.log_turn(session_id, physician_input, state, final_response)

        return final_response
