import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import List

def handle_file_upload() -> List[tempfile.NamedTemporaryFile]:
    """Handle file upload and return temporary files"""
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=['pdf', 'txt', 'png', 'jpg', 'jpeg', 'xml', 'docx'],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, PNG, JPG, XML, DOCX"
    )
    
    temp_files = []
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files selected")
        
        # Create temporary files
        for uploaded_file in uploaded_files:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_file.name).suffix
            )
            temp_file.write(uploaded_file.getvalue())
            temp_file.close()
            temp_files.append(temp_file)
    
    return temp_files

def cleanup_temp_files(temp_files: List[tempfile.NamedTemporaryFile]):
    """Clean up temporary files"""
    import logging
    
    for temp_file in temp_files:
        try:
            os.unlink(temp_file.name)
        except Exception as e:
            logging.warning(f"Failed to delete temp file {temp_file.name}: {e}") 