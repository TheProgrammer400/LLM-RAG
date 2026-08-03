from typing import List, Union
from shared.models.schemas import PatientState, CandidateEvidenceMatch, RetrievedChunk, RetrievalResult

class PromptBuilder:
    def build_system_prompt(
        self,
        state: PatientState,
        candidates: List[CandidateEvidenceMatch],
        retrieved_chunks: Union[List[RetrievedChunk], RetrievalResult]
    ) -> str:
        """
        Assembles strict system prompt:
        - Mandatory inline source+page tags: [Source: filename, p.X]
        - Explicit state representation: Confirmed vs Negated findings
        - Explicit disclaimer that compatibility tiers are evidence-overlap heuristics, not diagnostic probabilities
        - Prohibits introducing uncited diagnoses or drugs
        """
        facts = state.extracted_facts
        symptoms_str = ", ".join(facts.symptoms) if facts.symptoms else "None reported"
        negatives_str = ", ".join(facts.negative_findings) if facts.negative_findings else "None reported"
        labs_str = ", ".join([f"{k}: {v}" for k, v in facts.lab_findings.items()]) if facts.lab_findings else "None"

        chunks_list = retrieved_chunks.retrieved_chunks if isinstance(retrieved_chunks, RetrievalResult) else retrieved_chunks

        # Format Evidence Packet
        evidence_lines = []
        for r_chunk in chunks_list:
            meta = r_chunk.chunk.metadata
            evidence_lines.append(
                f"- [Source: {meta.source_title}, p.{meta.page_start}] Heading: {meta.heading}\n  Excerpt: {r_chunk.chunk.text}"
            )
        evidence_packet = "\n".join(evidence_lines) if evidence_lines else "No retrieved evidence."

        # Format Candidates
        candidate_lines = []
        for cand in candidates:
            candidate_lines.append(
                f"- Candidate: {cand.disease_name} (Evidence-Overlap Tier: {cand.compatibility_tier.value})\n"
                f"  Matched Features: {', '.join(cand.matching_present_findings)}\n"
                f"  Conflicting Negations: {', '.join(cand.conflicting_prerequisites)}\n"
                f"  Tier Rationale: {cand.tier_rationale}"
            )
        candidates_str = "\n".join(candidate_lines)

        prompt = f"""You are a physician-facing clinical decision support reasoning assistant.

=== PATIENT CLINICAL STATE ===
Demographics: {facts.demographics}
Confirmed Present Symptoms: {symptoms_str}
EXPLICITLY NEGATED FINDINGS: {negatives_str}
Laboratory & Imaging Findings: {labs_str}

=== RETRIEVED MEDICAL EVIDENCE PACKET ===
{evidence_packet}

=== PRE-RANKED DISEASE CANDIDATES (Evidence-Overlap Heuristic Tiers) ===
{candidates_str}

=== MANDATORY GENERATION INSTRUCTIONS ===
1. Every claim, recommendation, or investigation must carry an explicit citation: `[Source: filename, p.X]`.
2. Do NOT introduce any diagnosis, drug, or clinical recommendation that is not present in the supplied evidence packet.
3. Treat the pre-ranked candidates as evidence-overlap heuristics, NOT diagnostic probabilities.
4. Highlight any negated findings (e.g., '{negatives_str}') that rule out or reduce likelihood of specific diagnoses.
5. If evidence is ambiguous, state uncertainty explicitly.
"""
        return prompt
