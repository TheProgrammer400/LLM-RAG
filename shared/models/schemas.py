from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

# --- Enums ---

class IntentType(str, Enum):
    EXIT = "EXIT"
    GREETING = "GREETING"
    CLINICAL_UPDATE = "CLINICAL_UPDATE"
    SYMPTOM_DISCUSSION = "SYMPTOM_DISCUSSION"
    INVESTIGATION_RESULTS_UPDATE = "INVESTIGATION_RESULTS_UPDATE"
    GENERAL_MEDICAL_QUESTION = "GENERAL_MEDICAL_QUESTION"

class SeverityLevel(str, Enum):
    HOME_CARE_APPROPRIATE = "HOME_CARE_APPROPRIATE"
    ROUTINE = "ROUTINE"
    URGENT = "URGENT"
    EMERGENCY = "EMERGENCY"

class TurnStrategy(str, Enum):
    INFORMATION_GATHERING = "INFORMATION_GATHERING"
    SYMPTOM_CLARIFICATION = "SYMPTOM_CLARIFICATION"
    DIAGNOSTIC_REASONING = "DIAGNOSTIC_REASONING"

class ContentType(str, Enum):
    CLINICAL_REFERENCE = "clinical_reference"
    PATIENT_EDUCATION = "patient_education"
    CASE_REPORTS = "case_reports"
    ADMINISTRATIVE = "administrative"

class AuthorityTier(str, Enum):
    GUIDELINE = "guideline"
    SPECIALTY_TEXTBOOK = "specialty_textbook"
    GENERAL_TEXTBOOK = "general_textbook"
    PATIENT_EDUCATION = "patient_education"

class ConfidenceGateOutcome(str, Enum):
    SUFFICIENT = "sufficient"
    RETRY = "retry"
    INSUFFICIENT = "insufficient"

class CompatibilityTier(str, Enum):
    MOST_COMPATIBLE = "Most Compatible"
    COMPATIBLE = "Compatible"
    POSSIBLE = "Possible"
    LESS_COMPATIBLE = "Less Compatible"
    CURRENTLY_UNLIKELY = "Currently Unlikely"

class EntailmentStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    FAILED = "failed"


# --- Entity & Fact Extraction Models ---

class EntityCUI(BaseModel):
    text: str
    cui: str
    canonical_name: str
    semantic_type: Optional[str] = None
    synonyms: List[str] = Field(default_factory=list)

class FactExtraction(BaseModel):
    symptoms: List[str] = Field(default_factory=list)
    negative_findings: List[str] = Field(default_factory=list)
    confirmed_diagnoses: List[str] = Field(default_factory=list)
    ruled_out_diagnoses: List[str] = Field(default_factory=list)
    vital_signs: Dict[str, str] = Field(default_factory=dict)
    lab_findings: Dict[str, str] = Field(default_factory=dict)
    imaging_findings: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    duration: Optional[str] = None
    onset: Optional[str] = None
    demographics: Dict[str, Any] = Field(default_factory=dict)

class PatientState(BaseModel):
    session_id: str
    turn_number: int = 1
    extracted_facts: FactExtraction = Field(default_factory=FactExtraction)
    normalized_entities: List[EntityCUI] = Field(default_factory=list)
    fact_deltas: FactExtraction = Field(default_factory=FactExtraction)
    previous_ranked_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- Ingestion & Chunking Models ---

class ChunkMetadata(BaseModel):
    chunk_id: str
    parent_id: str
    source_id: str
    source_title: str
    page_start: int
    page_end: int
    heading: str
    section_type: str
    specialty_tags: List[str] = Field(default_factory=list)
    disease_entities: List[EntityCUI] = Field(default_factory=list)
    symptom_entities: List[EntityCUI] = Field(default_factory=list)
    drug_entities: List[EntityCUI] = Field(default_factory=list)
    authority_tier: AuthorityTier
    publication_year: int
    content_type: ContentType
    embedding_model_version: str
    ocr_derived: bool = False
    ingestion_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Chunk(BaseModel):
    chunk_id: str
    text: str
    parent_text: Optional[str] = None
    metadata: ChunkMetadata


# --- Retrieval Models ---

class QueryFacet(BaseModel):
    facet_type: str
    query_text: str
    expanded_terms: List[str] = Field(default_factory=list)

class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_weighted_score: float = 0.0

class RetrievalResult(BaseModel):
    facets_used: List[QueryFacet]
    retrieved_chunks: List[RetrievedChunk]
    confidence: ConfidenceGateOutcome
    top_rerank_score: float
    n_chunks_above_floor: int
    cache_hit: bool = False


# --- Reasoning Models ---

class ProvenanceCitation(BaseModel):
    source_title: str
    page: int
    chunk_id: str
    excerpt: str

class CandidateEvidenceMatch(BaseModel):
    disease_name: str
    cui: Optional[str] = None
    compatibility_tier: CompatibilityTier
    matching_present_findings: List[str] = Field(default_factory=list)
    conflicting_prerequisites: List[str] = Field(default_factory=list)
    missing_high_yield_features: List[str] = Field(default_factory=list)
    provenance: List[ProvenanceCitation] = Field(default_factory=list)
    tier_rationale: str

class EvolutionDelta(BaseModel):
    disease_name: str
    previous_tier: Optional[CompatibilityTier] = None
    new_tier: CompatibilityTier
    change_reason: str

class ReasoningResult(BaseModel):
    candidates: List[CandidateEvidenceMatch]
    grouped_evidence: Dict[str, List[RetrievedChunk]] = Field(default_factory=dict)
    evolution_deltas: List[EvolutionDelta] = Field(default_factory=list)


# --- Inference Gateway & Final Response Models ---

class CitationClaim(BaseModel):
    claim_text: str
    source_title: str
    page: int
    chunk_id: str
    entailment_status: EntailmentStatus = EntailmentStatus.UNVERIFIED

class DifferentialItem(BaseModel):
    rank: int
    disease_name: str
    compatibility_tier: CompatibilityTier
    clinical_rationale: str
    citations: List[CitationClaim]

class ConsultationResponse(BaseModel):
    session_id: str
    turn_number: int
    intent: IntentType
    severity: SeverityLevel
    turn_strategy: TurnStrategy
    is_emergency: bool = False
    emergency_guidance: Optional[str] = None
    differentials: List[DifferentialItem] = Field(default_factory=list)
    recommended_investigations: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    missing_critical_info: List[str] = Field(default_factory=list)
    retrieval_confidence: ConfidenceGateOutcome
    disclaimer: str = (
        "Notice: This system is a Clinical Decision Support System (CDSS) for physician reasoning assistance. "
        "It does not replace clinical judgment. All dosing, interactions, and diagnoses must be independently verified."
    )
    latency_ms: Dict[str, float] = Field(default_factory=dict)
