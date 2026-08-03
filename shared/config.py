from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # App General Settings
    APP_NAME: str = "Clinical Decision Support System (CDSS)"
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = Field(default=True)
    LOG_LEVEL: str = Field(default="INFO")

    # Service URLs / Ports
    NLP_SERVICE_URL: str = Field(default="http://localhost:8001")
    RETRIEVAL_SERVICE_URL: str = Field(default="http://localhost:8002")
    REASONING_SERVICE_URL: str = Field(default="http://localhost:8003")
    INFERENCE_GATEWAY_URL: str = Field(default="http://localhost:8004")
    ORCHESTRATOR_URL: str = Field(default="http://localhost:8000")

    # Database & Storage Configurations
    POSTGRES_URI: str = Field(default="postgresql+asyncpg://cdss_user:cdss_password@localhost:5432/cdss_db")
    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # Embedding & NLP Models
    PRIMARY_EMBEDDING_MODEL: str = Field(default="ncbi/MedCPT-Query-Encoder")
    FALLBACK_EMBEDDING_MODEL: str = Field(default="nomic-ai/nomic-embed-text-v1.5")
    RERANKER_MODEL: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    NLI_VERIFIER_MODEL: str = Field(default="cross-encoder/nli-deberta-v3-small")

    # LLM Settings
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    PRIMARY_LLM_MODEL: str = Field(default="qwen2.5:7b-instruct")
    ESCALATION_LLM_MODEL: str = Field(default="qwen2.5:14b-instruct")

    # Retrieval Thresholds & Constraints
    RERANK_SCORE_FLOOR: float = Field(default=0.45)
    MAX_SOURCE_PERCENTAGE: float = Field(default=0.40)  # max 40% per source
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = Field(default=0.95)
    DEDUP_SIMILARITY_THRESHOLD: float = Field(default=0.90)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
