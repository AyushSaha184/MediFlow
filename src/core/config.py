from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application core settings loaded from environment variables or .env file.
    """
    # Environment variables
    project_name: str = "MediFlow Multi-Agent Medical AI"
    environment: str = "dev"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
