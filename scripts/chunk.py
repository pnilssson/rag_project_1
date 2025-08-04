import re
from typing import List
from scripts.config import config
from scripts.utils import clean_text

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Improved text chunking with semantic boundaries and better overlap handling.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum chunk size in words
        overlap: Overlap size in words
    
    Returns:
        List of text chunks
    """
    if chunk_size is None:
        chunk_size = config.chunk_size
    if overlap is None:
        overlap = config.chunk_overlap
    
    # Clean the text first
    text = clean_text(text)
    
    # Split into sentences first for better semantic boundaries
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_length = len(sentence_words)
        
        # If adding this sentence would exceed chunk size
        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))
            
            # Start new chunk with overlap
            if overlap > 0:
                # Keep last few sentences for overlap
                overlap_words = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    sent_words = sent.split()
                    if overlap_length + len(sent_words) <= overlap:
                        overlap_words.insert(0, sent)
                        overlap_length += len(sent_words)
                    else:
                        break
                current_chunk = overlap_words
                current_length = overlap_length
            else:
                current_chunk = []
                current_length = 0
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Filter out very short chunks
    chunks = [chunk for chunk in chunks if len(chunk.split()) >= 10]
    
    return chunks

def chunk_by_paragraphs(text: str, max_chunk_size: int = None) -> List[str]:
    """
    Alternative chunking method that respects paragraph boundaries.
    """
    if max_chunk_size is None:
        max_chunk_size = config.chunk_size
    
    # Split by paragraphs
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        paragraph_words = paragraph.split()
        paragraph_length = len(paragraph_words)
        
        if current_length + paragraph_length > max_chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraph]
            current_length = paragraph_length
        else:
            current_chunk.append(paragraph)
            current_length += paragraph_length
    
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks
