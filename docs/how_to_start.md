# Clinical Decision Support System (CDSS) - Step-by-Step Execution Guide

This guide provides minor details on how to set up, start infrastructure, run services, and interact with the CDSS system once PDF knowledge base ingestion is complete.

---

## Prerequisites & Checklist

1. **Python**: Python >= 3.11 installed.
2. **Docker & Docker Compose**: Installed and running (for Qdrant, PostgreSQL, Redis, Ollama).
3. **Ollama Models**: Ensure required LLM & embedding models are downloaded locally:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen2.5:7b
   ollama pull phi4-mini:latest
   ```

---

## Step 1: Virtual Environment & Project Installation

Activate your Python virtual environment and install project dependencies:

```bash
# Navigate to project root directory
cd "/home/programmer/LLM & RAG"

# Activate existing virtual environment
source .venv/bin/activate

# Install CDSS package in editable mode
pip install -e .
```

---

## Step 2: Start Infrastructure Dependencies (Docker Compose)

Start vector database (Qdrant), session store (PostgreSQL), semantic cache (Redis), and Ollama server:

```bash
# Spin up infrastructure containers in detached mode
docker-compose -f infra/docker-compose.yml up -d
```

### Verify Container Status:
```bash
docker-compose -f infra/docker-compose.yml ps
```
Ensure ports `6333` (Qdrant), `5432` (PostgreSQL), `6379` (Redis), and `11434` (Ollama) are active.

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

## Step 4: Launch Microservices Architecture

The system consists of 5 microservices. You can launch them in separate terminal tabs or run them in background processes.

### Service Port Mapping:
| Service Name | Port | Entrypoint |
| :--- | :--- | :--- |
| **NLP Extraction Service** | `8001` | `services.nlp_service.main:app` |
| **Retrieval Service** | `8002` | `services.retrieval_service.main:app` |
| **Reasoning Service** | `8003` | `services.reasoning_service.main:app` |
| **Inference Gateway** | `8004` | `services.inference_gateway.main:app` |
| **Orchestrator Main API** | `8000` | `services.orchestrator.main:app` |

### Terminal Commands to Start Services:

```bash
# Terminal 1: NLP Service
uvicorn services.nlp_service.main:app --port 8001 --reload

# Terminal 2: Retrieval Service
uvicorn services.retrieval_service.main:app --port 8002 --reload

# Terminal 3: Reasoning Service
uvicorn services.reasoning_service.main:app --port 8003 --reload

# Terminal 4: Inference Gateway
uvicorn services.inference_gateway.main:app --port 8004 --reload

# Terminal 5: Main Orchestrator API Gateway
uvicorn services.orchestrator.main:app --port 8000 --reload
```

---

## Step 5: Interacting with the CDSS API

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

## Step 6: Running Automated Test Suites

To verify end-to-end pipeline functionality, run pytest:

```bash
pytest
```
