import streamlit as st
import os
from typing import Dict, Any

def display_processing_results(results: Dict[str, Any]):
    """Display processing results in a nice format"""
    if not results:
        return
    
    st.subheader("📊 Processing Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Files", results.get('total_files', 0))
    col2.metric("Processed", results.get('processed_files', 0))
    col3.metric("Failed", results.get('failed_files', 0))
    col4.metric("Success Rate", f"{results.get('success_rate', 0):.1%}")
    
    # Detailed results
    if results.get('processed'):
        st.subheader("✅ Successfully Processed")
        for result in results['processed']:
            with st.expander(f"📄 {os.path.basename(result['file'])}"):
                st.write(f"**Status:** {result['status']}")
                st.write(f"**Text Length:** {result['text_length']:,} characters")
                st.write(f"**Chunks Created:** {result['chunks_count']}")
                st.write(f"**File Size:** {result['file_size']:,} bytes")
    
    if results.get('failed'):
        st.subheader("❌ Failed Files")
        for result in results['failed']:
            with st.expander(f"📄 {os.path.basename(result['file'])}"):
                st.error(f"**Error:** {result['error']}")

def display_query_result(result: Dict[str, Any]):
    """Display query result in a formatted way"""
    if not result:
        return
    
    st.subheader("🤖 Answer")
    st.write(result.get('answer', 'No answer generated'))
    
    # Context management info
    if result.get('context_management'):
        st.subheader("🔧 Context Management")
        context_info = result['context_management']
        col1, col2, col3 = st.columns(3)
        col1.metric("Prompt Tokens", result.get('prompt_tokens', 0))
        col2.metric("Max Context Tokens", context_info.get('max_context_tokens', 0))
        col3.metric("Truncated Chunks", context_info.get('truncated_chunks', 0))
        
        if context_info.get('truncated_chunks', 0) > 0:
            st.warning("⚠️ Some chunks were truncated to fit within the model's context window.")
    
    # Sources and metadata
    if result.get('sources'):
        st.subheader("📚 Sources")
        sources = list(set(result['sources']))
        for source in sources:
            st.write(f"• {source}")
    
    # Detailed chunks
    if result.get('chunks'):
        st.subheader("📖 Retrieved Chunks")
        for i, chunk in enumerate(result['chunks'], 1):
            truncated_note = " (truncated)" if chunk.get('truncated', False) else ""
            with st.expander(f"Chunk {i} (Score: {chunk.get('score', 0):.3f}){truncated_note}"):
                st.write(f"**Source:** {chunk.get('source', 'Unknown')}")
                st.write(f"**Text:** {chunk.get('text', '')}")
                if chunk.get('truncated', False):
                    st.info("This chunk was truncated to fit within the context window.")

def display_statistics():
    """Display system statistics"""
    try:
        from scripts.embeddings import embedding_manager
        
        stats = embedding_manager.get_collection_info()
        
        st.subheader("📊 Vector Database Statistics")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Chunks", stats['points_count'])
        col2.metric("Vector Dimension", stats['vector_size'])
        col3.metric("Distance Metric", stats['distance'])
        
        # Collection details
        st.subheader("📋 Collection Details")
        st.json(stats)
        
    except Exception as e:
        st.error(f"Failed to retrieve statistics: {e}")

def display_settings():
    """Display and manage runtime settings"""
    from scripts.config import config
    
    st.subheader("⚙️ System Configuration")
    
    # Read-only settings
    st.subheader("📋 Current Settings (Read-only)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Qdrant Settings:**")
        st.code(f"Host: {config.qdrant_host}")
        st.code(f"Port: {config.qdrant_port}")
        st.code(f"Collection: {config.collection_name}")
        
        st.write("**Embedding Settings:**")
        st.code(f"Model: {config.embedding_model}")
        st.code(f"Vector Size: {config.vector_size}")
        
        st.write("**Chunking Settings:**")
        st.code(f"Chunk Size: {config.chunk_size}")
        st.code(f"Overlap: {config.chunk_overlap}")
    
    with col2:
        st.write("**Query Settings:**")
        st.code(f"Top K: {config.top_k}")
        st.code(f"Similarity Threshold: {config.similarity_threshold}")
        
        st.write("**LLM Settings:**")
        st.code(f"Model: {config.llm_model}")
        st.code(f"Temperature: {config.temperature}")
        st.code(f"Max Tokens: {config.max_tokens}")
        
        st.write("**Context Management:**")
        st.code(f"Max Context Tokens: {config.max_context_tokens}")
        st.code(f"Max Total Tokens: {config.max_total_tokens}")
        
        st.write("**Language Settings:**")
        st.code(f"OCR Languages: {config.ocr_languages}")
        st.code(f"System Language: {config.system_language}")
    
    st.info("💡 These settings are configured in `scripts/config.py`. Restart the application after changing settings.") 