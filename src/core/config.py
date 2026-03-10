from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application core settings loaded from environment variables or .env file.
    """
    # Environment variables
    project_name: str = "MediFlow Multi-Agent Medical AI"
    environment: str = "dev"
    log_level: str = "INFO"
    rag_embedding_provider: str = "nvidia_api"
    rag_embedding_model_name: str = "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"
    rag_embedding_fallback_dimension: int = 384
    rag_embedding_local_files_only: bool = True
    rag_embedding_nvidia_api_url: str = "https://integrate.api.nvidia.com/v1/embeddings"
    rag_embedding_nvidia_api_key: str = Field(default="", validation_alias="NVIDIA_EMBED_API_KEY")
    rag_embedding_nvidia_truncate: str = "NONE"
    rag_embedding_request_timeout_seconds: float = 60.0
    rag_embedding_nvidia_max_batch_size: int = 32
    rag_global_store_dir: str = "src/rag/global_store"
    rag_patient_data_root: str = "data/User"
    pgvector_database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/mediflow",
        validation_alias="PGVECTOR_DATABASE_URL",
    )
    
    # Phase 5 LLM Settings/Cerebras
    cerebras_api_key: str = Field(default="", validation_alias="CEREBRAS_API_KEY")
    diagnostic_model_name: str = "llama3.1-8b"
    
    # Phase 1.5 Vision Model Settings / NVIDIA API
    nvidia_api_key: str = Field(default="", validation_alias="NVIDIA_API_KEY")
    vision_model_name: str = "google/gemma-3-27b-it"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
