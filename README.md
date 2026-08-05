# Clinical Decision Support System (CDSS)

> **An Evidence-Grounded Medical RAG & Multi-Agent Reasoning Architecture for Physician Consultation**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-green.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-red.svg)](https://qdrant.tech/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black.svg)](https://ollama.ai/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)

---

## 📌 Executive Summary

The **Clinical Decision Support System (CDSS)** is a production-ready, multi-service microservice platform designed to assist healthcare professionals during clinical consultations. Built upon an advanced **Hybrid Retrieval-Augmented Generation (RAG)** pipeline and deterministic **Hard Safety Gates**, CDSS converts raw patient presentation data into evidence-grounded differential diagnosis recommendations, missing clinical feature highlights, and actionable diagnostic strategies—all strictly linked to verified medical literature with full provenance tracking.

---

## ⚙️ Architecture & Microservices Overview

The platform uses a decoupled microservices architecture coordinated by an Orchestrator Gateway.

```
                   +-----------------------------+
                   |  Physician Client / API     |
                   +--------------+--------------+
                                  |
                                  v
                   +--------------+--------------+
                   | Orchestrator Service (8000) |
                   +--------------+--------------+
                                  |
    +-----------------------------+-----------------------------+
    |                             |                             |
    v                             v                             v
+---+-------------------+   +-----+-------------------+   +-----+-------------------+
|  NLP Service (8001)   |   |Retrieval Service (8002) |   |Reasoning Service (8003) |
| • MedSpaCy / LLM      |   | • Qdrant Dense Vector   |   | • Evidence Matcher      |
| • Entity Extraction   |   | • BM25s Sparse Lexical  |   | • Compatibility Tiers   |
| • Red Flag Detector   |   | • RRF & Reranking       |   | • Evolution Tracker     |
+-----------------------+   +-------------------------+   +-------------------------+
                                                                |
                                                                v
                                                  +-------------+-----------+
                                                  | Inference Gateway (8004)|
                                                  | • Structured Output     |
                                                  | • Entailment Verifier   |
                                                  +-------------------------+
```

### Microservices Breakdown

| Service Name | Port | Primary Responsibilities | Key Technologies |
| :--- | :---: | :--- | :--- |
| **Orchestrator Gateway** | `8000` | Turn pipeline execution, session state management, hard safety checking, audit logging | FastAPI, Redis, Pydantic v2 |
| **NLP Service** | `8001` | Clinical fact extraction, medical entity CUI normalization, red-flag emergency detection | MedSpaCy, spaCy, Instructor |
| **Retrieval Service** | `8002` | Multi-faceted query generation, hybrid retrieval (dense + sparse), cross-encoder reranking, semantic caching | Qdrant, BM25s, SentenceTransformers, Redis |
| **Reasoning Service** | `8003` | Differential candidate generation, clinical prerequisite matching, tier assignment, multi-turn delta tracking | Pydantic, Custom Evidence Engine |
| **Inference Gateway** | `8004` | Prompt construction, local/external LLM execution, response structuring, citation entailment verification | Instructor, Ollama (`qwen2.5:7b-instruct`) |

---

## 🚀 Key Technical Features

### 1. 🛡️ Deterministic Hard Safety Gates & Emergency Triage
- Bypasses generative LLM pathways when acute medical red flags (e.g., suspected temporal arteritis with visual symptoms, acute coronary syndrome, severe anaphylaxis) are identified.
- Triggers immediate clinical safety guidance and protocol escalation.

### 2. 🔍 Hybrid RAG & Confidence-Gated Retrieval
- **Dual Engine Retrieval**: Combines dense semantic search via **Qdrant** (`nomic-embed-text`) and lexical keyword matching via **BM25s**.
- **Reciprocal Rank Fusion (RRF)** & **Cross-Encoder Reranking**: Merges and ranks retrieved passages based on relevancy scores.
- **Authority Tier Weighting**: Prefers clinical practice guidelines over general textbooks.
- **Semantic Caching**: Utilizes Redis for sub-millisecond retrieval of cached clinical queries.

### 3. 🧠 Clinical Reasoning & Compatibility Tiering
Classifies candidate differential diagnoses into structured compatibility tiers:
- **Most Compatible**: Meets all core prerequisites with high evidence overlap.
- **Compatible**: Strong match with minor missing non-critical features.
- **Possible**: Moderate presentation fit requiring further diagnostic workup.
- **Less Compatible**: Missing high-yield key features.
- **Currently Unlikely**: Conflicting clinical prerequisites or ruled out by lab/imaging.

### 4. 🔗 Self-Verifying Inference & Citation Entailment
- Validates generated claims against exact retrieved passage chunks (`EntailmentStatus`: `VERIFIED`, `UNVERIFIED`, `FAILED`).
- Enforces strict structured output via Pydantic schema validation to ensure response consistency.

### 5. 📄 Hierarchical PDF Ingestion Pipeline
- Processes medical guidelines via PyMuPDF, Unstructured, and Tesseract OCR.
- Performs parent-child chunking to retain structural context (chapters, sections, sub-headings).
- Automatically enriches chunks with CUI tags and metadata authority tiers.

---

## 📁 Repository Structure

```
LLM & RAG/
├── docs/                      # Execution, ingestion, coverage, and source registry docs
│   ├── coverage_matrix.csv
│   ├── how_to_start.md
│   ├── ingestion_guide.md
│   ├── pdfs/                  # Clinical PDF documents repository
│   └── source_registry.md
├── evaluation/                # Automated RAG & LLM Evaluation Framework
│   ├── faithfulness_eval.py
│   ├── gold_query_set.jsonl
│   ├── reasoning_eval.py
│   ├── retrieval_eval.py
│   └── run_eval_suite.py
├── infra/                     # Infrastructure configuration & containerization
│   ├── Dockerfile
│   └── docker-compose.yml
├── ingestion/                 # Document Parsing & Vector Ingestion Pipeline
│   ├── chunker.py
│   ├── coherence_checker.py
│   ├── content_classifier.py
│   ├── corpus_dedup.py
│   ├── embedder.py
│   ├── entity_extractor.py
│   ├── indexer.py
│   ├── pdf_parser.py
│   ├── run_ingestion.py
│   └── text_cleaner.py
├── services/                  # Core Microservices Architecture
│   ├── inference_gateway/     # LLM inference, prompts & citation self-verification
│   ├── nlp_service/           # Clinical NLP, entity extraction & red-flag detection
│   ├── orchestrator/          # API Gateway, pipeline flow & safety layer
│   ├── reasoning_service/     # Differential generator & evidence matcher
│   └── retrieval_service/     # Dense/Sparse hybrid retrieval & reranking
├── shared/                    # Shared data models, schemas, and configurations
│   ├── config.py
│   └── models/
│       └── schemas.py
├── tests/                     # Automated Test Suites
│   ├── integration/           # End-to-end integration tests
│   └── unit/                  # Unit tests for core algorithms
├── pyproject.toml             # Python dependencies and build metadata
├── start_services.py          # Microservices single-command python launcher
└── README.md                  # System Documentation
```

---

## 📊 Current Progress & Implementation Status

| Component | Status | Description |
| :--- | :---: | :--- |
| **Microservice Core (5/5 Services)** | `100% Complete` | All 5 microservices fully implemented with FastAPI & Uvicorn |
| **Hybrid Retrieval Engine** | `100% Complete` | Qdrant + BM25s + RRF + Cross-Encoder Reranker operational |
| **Clinical Safety & Audit Layer** | `100% Complete` | Hard safety gates, red flag rule engine, and structured logging |
| **Reasoning & Compatibility Engine** | `100% Complete` | Prerequisite matching, differential delta tracking fully operational |
| **Inference Gateway & Self-Verifier** | `100% Complete` | Citation verification & structured Pydantic response enforcement |
| **PDF Ingestion Pipeline** | `100% Complete` | Parent-child chunking, metadata extraction & vector indexing ready |
| **Evaluation Suite** | `100% Complete` | Benchmark tools for retrieval (NDCG/Recall), faithfulness, and reasoning |
| **Docker & Deployment Infra** | `100% Complete` | Full Docker Compose orchestration and native Python launcher provided |

---

## 🚀 Quick Start Guide

### Prerequisites

- **Python**: `>= 3.11`
- **Docker & Docker Compose**: Installed and active
- **Ollama**: Installed locally with the following pulled models:
  ```bash
  ollama pull nomic-embed-text
  ollama pull qwen2.5:7b-instruct
  ollama pull phi4-mini:latest
  ```

---

### Running the Application

#### Option A: Docker Compose Deployment (Recommended 🐳)

Boot infrastructure databases (Qdrant, Redis, PostgreSQL, Ollama) alongside all 5 microservices:

```bash
docker compose -f infra/docker-compose.yml up -d --build
```

Verify service containers:
```bash
docker compose -f infra/docker-compose.yml ps
```

#### Option B: Python Local Launcher (Development Mode 🐍)

1. Start database containers:
   ```bash
   docker compose -f infra/docker-compose.yml up -d qdrant postgres redis ollama
   ```

2. Launch all 5 microservices concurrently:
   ```bash
   python start_services.py
   ```

---

### Ingesting Clinical Knowledge Base

To populate the vector database with medical literature:

```bash
# Ingest single guideline document
python -m ingestion.run_ingestion --file "docs/pdfs/sample_gca_guideline.pdf" --source_id "GCA_001" --title "Giant Cell Arteritis Guideline"

# Ingest entire PDF directory
for pdf in docs/pdfs/*.pdf; do
    fname=$(basename "$pdf" .pdf)
    python -m ingestion.run_ingestion --file "$pdf" --source_id "$fname" --title "$fname"
done
```

---

## 🌐 API Endpoints & Usage

Once started, the Orchestrator API Gateway runs on `http://localhost:8000`.

- **Interactive API Documentation (Swagger)**: `http://localhost:8000/docs`
- **System Health Check**: `GET http://localhost:8000/`

### Example Consultation Turn Request

```bash
curl -X POST "http://localhost:8000/api/v1/consultation/turn" \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "session_test_001",
       "physician_input": "72-year-old female presents with 3-week history of severe right temporal headache, jaw claudication while chewing, scalp tenderness, and brief episode of vision loss in right eye. No fever. ESR is 88 mm/h."
     }'
```

---

## 🧪 Testing & Evaluation

### Run Test Suite
```bash
# Run unit & integration test suite
pytest
```

### Run Benchmark Evaluation Suite
```bash
# Execute retrieval, faithfulness, and reasoning benchmarks
python -m evaluation.run_eval_suite
```

---

## ⚖️ License & Medical Disclaimer

This software is developed strictly for research, educational, and clinical decision support demonstration purposes. 

**Clinical Notice**: This system is a Clinical Decision Support System (CDSS) intended solely for physician reasoning assistance. It does not replace professional clinical judgment. All medication dosages, drug-drug interactions, and clinical diagnoses must be independently verified by a qualified physician prior to clinical intervention.
