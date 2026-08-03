import uuid
import logging
from typing import List, Tuple
from shared.models.schemas import Chunk, ChunkMetadata, AuthorityTier, ContentType
from ingestion.pdf_parser import LayoutBlock

logger = logging.getLogger(__name__)

class SectionBoundaryChunker:
    def __init__(self, target_min_words: int = 80, target_max_words: int = 180, overlap_pct: float = 0.12):
        self.target_min_words = target_min_words
        self.target_max_words = target_max_words
        self.overlap_pct = overlap_pct

    def create_chunks_from_blocks(
        self,
        blocks: List[LayoutBlock],
        source_id: str,
        source_title: str,
        authority_tier: AuthorityTier,
        publication_year: int,
        content_type: ContentType,
        ocr_derived: bool = False
    ) -> Tuple[List[Chunk], List[dict]]:
        """
        Groups blocks into sections, produces parent sections and child chunks with overlap.
        Enforces 80-180 word boundaries.
        """
        sections = self._group_into_sections(blocks)
        chunks: List[Chunk] = []
        parents: List[dict] = []

        for current_heading, section_blocks, start_page, end_page in sections:
            section_id = f"parent_{uuid.uuid4().hex[:12]}"
            section_text = " ".join([b.text for b in section_blocks])

            parents.append({
                "parent_id": section_id,
                "heading": current_heading,
                "text": section_text,
                "source_id": source_id
            })

            # Create child chunks
            words = section_text.split()
            if not words:
                continue

            chunk_word_lists = self._split_words_into_windows(words)

            for i, chunk_words in enumerate(chunk_word_lists):
                chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
                chunk_text = " ".join(chunk_words)

                # Classify section_type
                section_type = self._determine_section_type(current_heading, chunk_text)

                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    parent_id=section_id,
                    source_id=source_id,
                    source_title=source_title,
                    page_start=start_page,
                    page_end=end_page,
                    heading=current_heading,
                    section_type=section_type,
                    specialty_tags=[],  # Populated downstream
                    disease_entities=[],
                    symptom_entities=[],
                    drug_entities=[],
                    authority_tier=authority_tier,
                    publication_year=publication_year,
                    content_type=content_type,
                    embedding_model_version="ncbi/MedCPT-Query-Encoder",
                    ocr_derived=ocr_derived
                )

                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    parent_text=section_text,
                    metadata=metadata
                ))

        logger.info(f"Chunker created {len(chunks)} child chunks across {len(parents)} parent sections for {source_id}.")
        return chunks, parents

    def _group_into_sections(self, blocks: List[LayoutBlock]) -> List[Tuple[str, List[LayoutBlock], int, int]]:
        """Groups blocks into (heading, list_of_blocks, start_page, end_page)."""
        sections = []
        current_heading = "General Overview"
        current_blocks: List[LayoutBlock] = []
        start_page = blocks[0].page_num if blocks else 1

        for b in blocks:
            if b.block_type == "heading" or self._looks_like_heading(b.text):
                if current_blocks:
                    end_page = current_blocks[-1].page_num
                    sections.append((current_heading, current_blocks, start_page, end_page))
                current_heading = b.text
                current_blocks = [b]
                start_page = b.page_num
            else:
                current_blocks.append(b)

        if current_blocks:
            end_page = current_blocks[-1].page_num
            sections.append((current_heading, current_blocks, start_page, end_page))

        return sections

    def _looks_like_heading(self, text: str) -> bool:
        """Heuristic for section heading detection."""
        t = text.strip()
        if len(t) < 80 and not t.endswith(".") and (t.isupper() or t.istitle()):
            return True
        return False

    def _split_words_into_windows(self, words: List[str]) -> List[List[str]]:
        """Splits word list into target_max_words windows with overlap_pct overlap."""
        if len(words) <= self.target_max_words:
            return [words]

        chunk_size = self.target_max_words
        overlap_size = int(chunk_size * self.overlap_pct)
        step = chunk_size - overlap_size

        windows = []
        for start in range(0, len(words), step):
            end = start + chunk_size
            window = words[start:end]
            if len(window) >= 30:  # Avoid tiny trailing fragments
                windows.append(window)
            elif windows:
                windows[-1].extend(window)
            if end >= len(words):
                break

        return windows

    def _determine_section_type(self, heading: str, text: str) -> str:
        """Determines clinical section type (differential, investigations, management, etc.)."""
        h_lower = heading.lower()
        t_lower = text.lower()

        if "differential" in h_lower or "diagnosis" in h_lower:
            return "differential"
        elif "investigation" in h_lower or "laboratory" in h_lower or "imaging" in h_lower or "biopsy" in h_lower:
            return "investigations"
        elif "treatment" in h_lower or "management" in h_lower or "therapy" in h_lower:
            return "management"
        elif "symptom" in h_lower or "clinical presentation" in h_lower or "features" in h_lower:
            return "symptoms"
        elif "risk factor" in h_lower or "etiology" in h_lower or "epidemiology" in h_lower:
            return "risk_factors"
        elif "sign" in h_lower or "physical exam" in h_lower:
            return "signs"
        
        return "general"
