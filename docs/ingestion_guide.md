# CDSS RAG PDF Ingestion Guide

This guide provides step-by-step instructions for ingesting clinical documents, guidelines, and textbooks into the Clinical Decision Support System (CDSS) RAG knowledge base.

---

## 1. Prerequisites & Infrastructure Setup

Before ingesting documents, ensure that the Qdrant vector database, PostgreSQL, Redis, and Ollama services are running via Docker Compose:

```bash
docker compose -f infra/docker-compose.yml up -d
``` 

Verify that Qdrant is accessible on `http://localhost:6333`.

---

## 2. Ingesting a Single PDF Document

To ingest an individual PDF file into the knowledge base, use the offline ingestion CLI script `ingestion/run_ingestion.py`:

```bash
python3 ingestion/run_ingestion.py \
  --file "/path/to/clinical_guideline.pdf" \
  --source_id "SRC_RHEUM_001" \
  --title "EULAR Guidelines on Giant Cell Arteritis"
```

### Command Arguments:
- `--file`: Absolute or relative path to the target PDF document.
- `--source_id`: Unique string tag for tracking source provenance (e.g. `SRC_CARDIO_001`, `SRC_NEURO_002`).
- `--title`: Official title of the publication or chapter.

---

## 3. Batch Ingesting a Directory of PDFs

To ingest an entire folder of PDF documents at once, execute the following bash loop in your terminal:

```bash
#!/bin/bash
PDF_DIR="/path/to/your/pdf_folder"

for pdf_file in "$PDF_DIR"/*.pdf; do
    if [ -f "$pdf_file" ]; then
        filename=$(basename "$pdf_file" .pdf)
        source_id="SRC_${filename// /_}"
        
        echo "=================================================="
        echo "Ingesting document: $filename ($source_id)"
        echo "=================================================="
        
        python3 ingestion/run_ingestion.py \
          --file "$pdf_file" \
          --source_id "$source_id" \
          --title "$filename"
    fi
done
```

---

## 4. What Happens During Ingestion Pipeline Execution?

Each document processed through `ingestion/run_ingestion.py` passes through 10 automated pipeline stages:

1. **PDF Layout Parsing (`pdf_parser.py`)**: PyMuPDF extraction preserving headings, paragraphs, and page bounding boxes with OCR fallback for scanned pages.
2. **Text Cleaning (`text_cleaner.py`)**: Running header/footer stripping, unicode normalization (NFKC), and line-break de-hyphenation.
3. **Content Classification (`content_classifier.py`)**: Automatic section routing (`clinical_reference`, `patient_education`, `administrative`).
4. **Pre-computed Entity Extraction (`entity_extractor.py`)**: spaCy/medspaCy pass tagging diseases, symptoms, and drugs with canonical UMLS CUIs.
5. **Parent-Child Section Chunker (`chunker.py`)**: Hard-enforced 80–180 word windows with 10–15% overlap. Stores full parent section text for context expansion.
6. **Heading-Content Coherence (`coherence_checker.py`)**: Embeds heading vs body text to flag low-similarity misaligned chunks.
7. **Corpus Deduplication (`corpus_dedup.py`)**: Detects and links near-duplicate chunks across the corpus.
8. **Dense Embedding (`embedder.py`)**: Generates MedCPT query/article vectors.
9. **Dual Indexing (`indexer.py`)**: Upserts dense vectors into Qdrant collections and exact-term indices into the BM25s store.
10. **Source Registry Update (`docs/source_registry.md`)**: Records document provenance metadata.

---

## 5. Verifying Ingestion Success

After ingesting documents, run the CDSS evaluation harness to verify retrieval metrics:

```bash
python3 -m evaluation.run_eval_suite
```
