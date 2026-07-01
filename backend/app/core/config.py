import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Enterprise GenAI Phishing Intelligence Platform"
    API_V1_STR: str = "/api/v1"
    
    # Database Settings
    # Defaulting to an empty string so the system knows to fallback to SQLite if not provided
    DATABASE_URL: Optional[str] = None
    
    # LLM Settings
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    # MLflow Settings
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    
    # OCR settings
    TESSERACT_CMD: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
