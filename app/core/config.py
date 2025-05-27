"""
Application configuration
"""
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
import yaml
from pathlib import Path

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Local Call Center AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # Database
    DATABASE_URL: str = "sqlite:///./data/db/call_center.db"
    
    # Redis
    REDIS_URL: Optional[str] = None
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # LLM
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4-turbo-preview"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 500
    
    # Speech
    STT_MODEL: str = "base"
    TTS_CACHE_ENABLED: bool = True
    VAD_THRESHOLD: float = 0.5
    
    # Paths
    DATA_DIR: Path = Path("./data")
    CACHE_DIR: Path = Path("./data/cache")
    AUDIO_DIR: Path = Path("./data/audio")
    
    class Config:
        env_file = ".env"
        
    def load_yaml_config(self, config_path: str = "config/settings.yaml"):
        """Load additional settings from YAML"""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                return yaml_config
        return {}

# Create settings instance
settings = Settings()

# Load YAML configuration
yaml_config = settings.load_yaml_config()