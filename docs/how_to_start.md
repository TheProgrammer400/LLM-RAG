# Clinical Decision Support System (CDSS) - Step-by-Step Execution Guide

This guide provides minor details on how to set up, start infrastructure, run services, and interact with the CDSS system once PDF knowledge base ingestion is complete.

---

## Prerequisites & Checklist

1. **Python**: Python >= 3.11 installed.
2. **Docker & Docker Compose**: Installed and running (for Qdrant, PostgreSQL, Redis, Ollama).
3. **Ollama Models**: Ensure required LLM & embedding models are downloaded locally:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen2.5:7b-instruct
   ollama pull phi4-mini:latest
   ```

---

## Option A: Single-Command Startup (Docker Compose - Recommended 🐳)

Spin up **EVERYTHING** (Databases + Qdrant + Redis + Ollama + All 5 Microservices) concurrently in the background with a single command:

```bash
docker compose -f infra/docker-compose.yml up -d --build
```

### Verify Container Status:
```bash
docker compose -f infra/docker-compose.yml ps
```
Ensure ports `8000` (Main API), `8001` (NLP), `8002` (Retrieval), `8003` (Reasoning), `8004` (Inference), `6333` (Qdrant), `5432` (PostgreSQL), `6379` (Redis), and `11434` (Ollama) are active.

---

## Option B: Single-Command Startup (Python Launcher 🐍)

If you prefer running databases in Docker and Python services locally:

### 1. Start All Infrastructure & Ollama Containers:
```bash
docker compose -f infra/docker-compose.yml up -d qdrant postgres redis ollama
```

### 2. Run All 5 Microservices with 1 Command:
```bash
python start_services.py
```
*(Press `Ctrl+C` anytime to shut down all 5 services cleanly).*

---

## Step 3: Knowledge Base Ingestion (Verification)

If you haven't ingested your PDFs yet, ingest them now:

```bash
# Ingest single PDF file
python -m ingestion.run_ingestion --file "docs/pdfs/your_guideline.pdf" --source_id "SRC_001" --title "Clinical Reference"

# OR batch ingest all PDFs in docs/pdfs/
for pdf in docs/pdfs/*.pdf; do
    fname=$(basename "$pdf" .pdf)
    python -m ingestion.run_ingestion --file "$pdf" --source_id "$fname" --title "$fname"
done
```

---

## Step 4: Interacting with the CDSS API

Once services are running, interact with the system via HTTP REST calls or Swagger UI.

### 1. System Health Check
```bash
curl http://localhost:8000/
# Returns: {"system": "CDSS Orchestrator", "status": "online"}
```

### 2. Interactive Swagger UI
Open your web browser and navigate to:
`http://localhost:8000/docs`

### 3. Submit a Physician Consultation Turn (cURL Request)

```bash
curl -X POST "http://localhost:8000/api/v1/consultation/turn" \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "session_test_001",
       "physician_input": "72-year-old female presents with 3-week history of severe right temporal headache, jaw claudication while chewing, scalp tenderness, and brief episode of vision loss in right eye. No fever. ESR is 88 mm/h."
     }'
```

---

## Step 5: Running Automated Test Suites

To verify end-to-end pipeline functionality, run pytest:

```bash
pytest
```
