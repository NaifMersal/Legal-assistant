"""Configuration management for the Legal Assistant API"""
import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    app_name: str = "Legal Assistant API"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    debug: bool = False
    
    # CORS Settings
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000", "http://localhost:5174", "http://localhost:8080", "http://localhost:8081"]
    
    # Google Gemini API
    google_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.1
    
    # RAG Configuration
    rag_k: int = 12
    rag_dense_weight: float = 0.7
    rag_sparse_weight: float = 0.3
    
    # File Paths
    base_dir: Path = Path(__file__).parent.parent
    faiss_index_path: str = "data/m3_legal_faiss.index"
    documents_path: str = "data/saudi_laws_scraped.json"
    
    # Model Settings
    embeddings_model_name: str = "BAAI/bge-m3"
    use_fp16: bool = False  # Disable fp16 for stability on Mac
    metric_type: str = "ip"
    faiss_threads: int = 4  # Reduce threads to avoid crashes
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
