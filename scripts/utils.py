import logging
import os
import time
from typing import List, Optional
from pathlib import Path

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rag_system.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_file_extension(file_path: str) -> str:
    """Get file extension in lowercase"""
    return Path(file_path).suffix.lower()

def is_supported_file(file_path: str, supported_extensions: set) -> bool:
    """Check if file extension is supported"""
    return get_file_extension(file_path) in supported_extensions

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    import re
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    return text.strip()

def measure_time(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def batch_process(items: List, batch_size: int = 100):
    """Process items in batches"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size] 