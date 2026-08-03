import os
import sys
import argparse
import logging
from typing import List

from shared.models.schemas import AuthorityTier, ContentType
from ingestion.pdf_parser import PDFParser
from ingestion.text_cleaner import TextCleaner
from ingestion.content_classifier import ContentClassifier
from ingestion.entity_extractor import MedicalEntityExtractor
from ingestion.chunker import SectionBoundaryChunker
from ingestion.coherence_checker import CoherenceChecker
from ingestion.corpus_dedup import CorpusDeduplicator
from ingestion.embedder import MedicalEmbedder
from ingestion.indexer import DualIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("IngestionPipeline")

def run_ingestion_pipeline(
    file_path: str,
    source_id: str,
    source_title: str,
    authority_tier: AuthorityTier,
    publication_year: int,
    specialty_tags: List[str]
):
    logger.info(f"--- Starting CDSS Ingestion Pipeline for: {source_title} ({source_id}) ---")

    # 1. Parse PDF
    parser = PDFParser()
    blocks, pdf_meta = parser.parse_pdf(file_path)

    # 2. Text Cleaning
    cleaner = TextCleaner()
    cleaned_blocks = cleaner.clean_blocks(blocks)

    # 3. Content Classification & Chunking
    chunker = SectionBoundaryChunker(target_min_words=80, target_max_words=180)
    chunks, parent_sections = chunker.create_chunks_from_blocks(
        blocks=cleaned_blocks,
        source_id=source_id,
        source_title=source_title,
        authority_tier=authority_tier,
        publication_year=publication_year,
        content_type=ContentType.CLINICAL_REFERENCE,
        ocr_derived=pdf_meta.get("ocr_derived", False)
    )

    # 4. Medical Entity Extraction & Metadata Tagging
    entity_extractor = MedicalEntityExtractor()
    for chunk in chunks:
        chunk.metadata.specialty_tags = specialty_tags
        entities = entity_extractor.extract_entities(f"{chunk.metadata.heading} {chunk.text}")
        chunk.metadata.disease_entities = entities["diseases"]
        chunk.metadata.symptom_entities = entities["symptoms"]
        chunk.metadata.drug_entities = entities["drugs"]

    # 5. Coherence Check
    embedder = MedicalEmbedder()
    coherence_checker = CoherenceChecker()
    coherent_chunks = coherence_checker.check_heading_coherence(chunks, embedder)

    # 6. Corpus Deduplication
    deduper = CorpusDeduplicator()
    deduped_chunks = deduper.deduplicate_chunks(coherent_chunks, embedder)

    # 7. Dual Indexing into Qdrant + BM25s
    indexer = DualIndexer()
    indexer.index_chunks(deduped_chunks, embedder, collection_name="clinical_reference")

    logger.info(f"--- Successfully Ingested {len(deduped_chunks)} chunks for {source_id} ---")
    return deduped_chunks

def create_sample_pdf_and_ingest():
    """Generates a synthetic PDF with clinical content to test ingestion end-to-end."""
    sample_pdf = "sample_gca_guideline.pdf"
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "EULAR Guidelines on Giant Cell Arteritis (GCA)\n\n"
        "1. CLINICAL PRESENTATION AND DIAGNOSIS\n"
        "Giant cell arteritis (GCA), also known as temporal arteritis, is a systemic necrotizing vasculitis affecting large and medium-sized arteries. "
        "It predominantly affects patients over 50 years of age. Classical symptoms include temporal headache, jaw claudication (jaw pain while chewing), "
        "scalp tenderness, constitutional symptoms like fever and 6kg weight loss over 2 months, and sudden painless monocular vision loss (amaurosis fugax or CRAO).\n\n"
        "2. LABORATORY AND IMAGING INVESTIGATIONS\n"
        "Laboratory evaluation demonstrates markedly elevated inflammatory markers including Erythrocyte Sedimentation Rate (ESR > 50 mm/h, often > 100 mm/h) and C-Reactive Protein (CRP). "
        "Golden standard diagnostic investigation remains temporal artery biopsy showing granulomatous inflammation with multinucleated giant cells. Temporal artery ultrasound showing halo sign is also high yield.\n\n"
        "3. MANAGEMENT AND TREATMENT\n"
        "Immediate high-dose systemic corticosteroid therapy (oral prednisone 40-60 mg daily or IV methylprednisolone pulse for visual loss) must be initiated immediately upon strong clinical suspicion. "
        "Treatment should never be delayed pending temporal artery biopsy results due to imminent risk of permanent bilateral vision loss.\n",
        fontsize=11
    )
    doc.save(sample_pdf)
    doc.close()

    run_ingestion_pipeline(
        file_path=sample_pdf,
        source_id="SRC_RHEUM_001",
        source_title="EULAR Guidelines on Giant Cell Arteritis & Polymyalgia Rheumatica",
        authority_tier=AuthorityTier.GUIDELINE,
        publication_year=2023,
        specialty_tags=["rheumatology", "ophthalmology"]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CDSS Offline Ingestion Pipeline CLI")
    parser.add_argument("--file", type=str, help="Path to PDF file to ingest")
    parser.add_argument("--source_id", type=str, help="Source ID tag")
    parser.add_argument("--title", type=str, help="Source Title")
    parser.add_argument("--sample", action="store_true", help="Run ingestion on a synthetic sample GCA document")
    args = parser.parse_args()

    if args.sample:
        create_sample_pdf_and_ingest()
    elif args.file:
        run_ingestion_pipeline(
            file_path=args.file,
            source_id=args.source_id or "SRC_GEN_001",
            source_title=args.title or "Clinical Reference Doc",
            authority_tier=AuthorityTier.SPECIALTY_TEXTBOOK,
            publication_year=2023,
            specialty_tags=["general"]
        )
    else:
        parser.print_help()
