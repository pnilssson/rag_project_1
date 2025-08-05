import streamlit as st
import tempfile
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add scripts to path
sys.path.append('scripts')

from ui.components import (
    handle_file_upload, 
    display_processing_results, 
    cleanup_temp_files,
    show_error_with_details
)

def upload_and_process_page():
    """Upload and Process page functionality"""
    st.title("📤 Upload & Process Documents")
    
    st.markdown("""
    Upload your documents to process them through the RAG pipeline. 
    Supported formats: PDF, TXT, PNG, JPG, XML, DOCX
    """)
    
    # File upload
    temp_files = handle_file_upload()
    
    if temp_files:
        st.markdown("---")
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            recreate_collection = st.checkbox(
                "Recreate Collection", 
                help="Delete existing data and create new collection"
            )
        
        with col2:
            if st.button("🚀 Process Documents", type="primary"):
                try:
                    from scripts.pipeline import RAGPipeline
                    from scripts.embeddings import embedding_manager
                    
                    # Initialize pipeline
                    pipeline = RAGPipeline()
                    
                    # Recreate collection if requested
                    if recreate_collection:
                        with st.spinner("Recreating collection..."):
                            embedding_manager.init_collection(recreate=True)
                        st.success("Collection recreated!")
                    
                    # Process files
                    with st.spinner("Processing documents..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        results = []
                        for i, file_info in enumerate(temp_files):
                            original_name = file_info['original_name']
                            file_path = file_info['file_path']
                            file_type = file_info['file_type']
                            status_text.text(f"Processing {original_name} ({file_type})...")
                            
                            try:
                                result = pipeline.process_file(file_path, source_name=original_name)
                                results.append(result)
                                
                                # Show chunking info for successful processing
                                if result["status"] == "success":
                                    st.success(f"✅ {original_name}: {result['chunks_count']} chunks created")
                                else:
                                    st.warning(f"⚠️ {original_name}: {result['status']}")
                                    
                            except Exception as e:
                                results.append({
                                    "status": "error",
                                    "file": original_name,
                                    "error": str(e)
                                })
                                st.error(f"❌ {original_name}: {str(e)}")
                            
                            progress_bar.progress((i + 1) / len(temp_files))
                        
                        status_text.text("Processing complete!")
                        
                        # Display results
                        display_processing_results({
                            "total_files": len(temp_files),
                            "processed_files": len([r for r in results if r["status"] == "success"]),
                            "failed_files": len([r for r in results if r["status"] == "error"]),
                            "success_rate": len([r for r in results if r["status"] == "success"]) / len(temp_files),
                            "processed": [r for r in results if r["status"] == "success"],
                            "failed": [r for r in results if r["status"] == "error"]
                        })
                        
                except Exception as e:
                    show_error_with_details(e, "Processing failed")
                
                finally:
                    # Cleanup temporary files
                    cleanup_temp_files(temp_files) 