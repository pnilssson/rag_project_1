import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import List

def handle_file_upload() -> List[dict]:
    """Handle file upload and return temporary files with original names"""
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=['pdf', 'txt', 'png', 'jpg', 'jpeg', 'xml', 'docx'],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, PNG, JPG, XML, DOCX"
    )
    
    temp_files = []
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files selected")
        
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
                'file_path': temp_file.name
            })
    
    return temp_files

def cleanup_temp_files(temp_files: List[dict]):
    """Clean up temporary files"""
    import logging
    
    for file_info in temp_files:
        try:
            os.unlink(file_info['file_path'])
        except Exception as e:
            logging.warning(f"Failed to delete temp file {file_info['file_path']}: {e}") 