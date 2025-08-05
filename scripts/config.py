import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "rag_chunks"
    
    # Embedding model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_size: int = 384
    
    # Chunking settings
    chunk_size: int = 300
    chunk_overlap: int = 50
    
    # Query settings
    top_k: int = 5  # Reduced from 10 to prevent context overflow
    similarity_threshold: float = 0.3
    
    # LLM settings
    llm_model: str = "qwen/qwen3-8b"
    openai_api_base: str = "http://localhost:1234/v1"
    openai_api_key: str = "sk-local"
    temperature: float = 0.3
    max_tokens: int = 1000
    
    # Context management settings
    max_context_tokens: int = 3000  # Reserve space for prompt template and response
    max_total_tokens: int = 4096    # Model's context window
    
    # Language settings
    ocr_languages: str = "eng+swe"  # Tesseract OCR languages
    system_language: str = "sv"     # System language (sv, en, etc.)
    fallback_language: str = "eng"  # Fallback language for OCR
    
    # File processing
    data_dir: str = "data"
    supported_extensions: set = None
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = {'.pdf', '.txt', '.png', '.jpg', '.jpeg', '.xml', '.docx'}

# Global config instance
config = Config() 