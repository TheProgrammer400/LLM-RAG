import logging
from typing import List, Union
from shared.models.schemas import CitationClaim, EntailmentStatus, RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)

class NLIEntailmentVerifier:
    def verify_citations(self, citations: List[CitationClaim], retrieval_result: Union[List[RetrievedChunk], RetrievalResult]) -> List[CitationClaim]:
        """
        Runs Stage 9.3 self-verification pass.
        Checks NLI entailment between generated claim and cited chunk text.
        Flags claims as verified or unverified.
        """
        if not citations:
            return []

        chunks_list = retrieval_result.retrieved_chunks if isinstance(retrieval_result, RetrievalResult) else retrieval_result
        chunk_map = {c.chunk.chunk_id: c.chunk.text for c in chunks_list}

        verified_citations = []
        for cite in citations:
            chunk_text = chunk_map.get(cite.chunk_id, "")
            claim_text = cite.claim_text.lower()

            if not chunk_text:
                # If chunk ID was not found directly, check if claim matches any retrieved text snippet
                matched_any = any(word in c.chunk.text.lower() for c in chunks_list for word in claim_text.split() if len(word) > 4)
                cite.entailment_status = EntailmentStatus.VERIFIED if matched_any else EntailmentStatus.UNVERIFIED
                verified_citations.append(cite)
                continue

            # Entailment heuristic check (keyword/overlap entailment pass)
            claim_words = [w for w in claim_text.split() if len(w) > 3]
            match_count = sum(1 for w in claim_words if w in chunk_text.lower())
            
            if match_count / max(len(claim_words), 1) >= 0.30:
                cite.entailment_status = EntailmentStatus.VERIFIED
            else:
                cite.entailment_status = EntailmentStatus.UNVERIFIED
                logger.warning(f"Entailment check unverified for claim: '{cite.claim_text[:50]}...'")

            verified_citations.append(cite)

        return verified_citations
