import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add scripts to path for LangChain chunker
sys.path.append('scripts')

def handle_file_upload() -> List[dict]:
    """Handle file upload and return temporary files with original names"""
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=['pdf', 'txt', 'png', 'jpg', 'jpeg', 'xml', 'docx', 'md'],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, PNG, JPG, XML, DOCX, MD"
    )
    
    temp_files = []
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files selected")
        
        # Show file details and chunking preview
        with st.expander("📋 File Details & Chunking Preview", expanded=False):
            for uploaded_file in uploaded_files:
                file_size = len(uploaded_file.getvalue())
                file_ext = Path(uploaded_file.name).suffix.lower()
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{uploaded_file.name}**")
                with col2:
                    st.write(f"{file_size:,} bytes")
                with col3:
                    st.write(f"`.{file_ext[1:]}`")
                
                # Show chunking strategy info
                chunking_info = get_chunking_info(file_ext)
                st.info(f"Chunking: {chunking_info}")
        
        # Create temporary files with original names preserved
        for uploaded_file in uploaded_files:
            # Create temp file with original extension
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_file.name).suffix
            )
            temp_file.write(uploaded_file.getvalue())
            temp_file.close()
            
            # Store both temp file and original name
            temp_files.append({
                'temp_file': temp_file,
                'original_name': uploaded_file.name,
                'file_path': temp_file.name,
                'file_size': len(uploaded_file.getvalue()),
                'file_type': Path(uploaded_file.name).suffix.lower()
            })
    
    return temp_files

def get_chunking_info(file_ext: str) -> str:
    """Get chunking strategy information for file type"""
    try:
        from scripts.config import config
        
        if file_ext == '.pdf':
            return f"PyMuPDF loader + RecursiveCharacterTextSplitter (chunk_size={config.chunk_size})"
        elif file_ext in ['.docx', '.doc']:
            return f"UnstructuredWordDocumentLoader + RecursiveCharacterTextSplitter (chunk_size={config.chunk_size})"
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            return f"UnstructuredImageLoader + RecursiveCharacterTextSplitter (chunk_size={config.chunk_size})"
        elif file_ext in ['.txt', '.md']:
            return f"TextLoader + RecursiveCharacterTextSplitter (chunk_size={config.chunk_size})"
        elif file_ext == '.xml':
            return f"TextLoader + RecursiveCharacterTextSplitter (chunk_size={config.chunk_size})"
        else:
            return f"TextLoader + RecursiveCharacterTextSplitter (chunk_size={config.chunk_size})"
    except ImportError:
        return "LangChain RecursiveCharacterTextSplitter"

def preview_chunking(file_path: str, max_preview_chunks: int = 3) -> List[str]:
    """Preview chunking for a file (for demonstration purposes)"""
    try:
        from scripts.chunk import LangChainChunker
        
        chunker = LangChainChunker()
        chunks = chunker.chunk_file(file_path)
        
        # Return first few chunks for preview
        preview_chunks = []
        for i, chunk in enumerate(chunks[:max_preview_chunks]):
            content = chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content
            preview_chunks.append(f"Chunk {i+1}: {content}")
        
        if len(chunks) > max_preview_chunks:
            preview_chunks.append(f"... and {len(chunks) - max_preview_chunks} more chunks")
        
        return preview_chunks
    except Exception as e:
        return [f"Error previewing chunks: {str(e)}"]

def cleanup_temp_files(temp_files: List[dict]):
    """Clean up temporary files"""
    import logging
    
    for file_info in temp_files:
        try:
            os.unlink(file_info['file_path'])
        except Exception as e:
            logging.warning(f"Failed to delete temp file {file_info['file_path']}: {e}") 