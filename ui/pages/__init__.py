# UI Pages package for RAG System

from .upload_process import upload_and_process_page
from .query_system import query_system_page
from .statistics import statistics_page
from .settings import settings_page

__all__ = [
    'upload_and_process_page',
    'query_system_page',
    'statistics_page',
    'settings_page'
] 