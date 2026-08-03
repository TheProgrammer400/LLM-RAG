FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (PyMuPDF, pytesseract, tesseract-ocr, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and source code
COPY pyproject.toml .
COPY shared/ shared/
COPY ingestion/ ingestion/
COPY services/ services/

# Install python package and dependencies
RUN pip install --no-cache-dir -e .

EXPOSE 8000 8001 8002 8003 8004
