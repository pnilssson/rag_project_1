#!/usr/bin/env python3
"""
RAG System Web UI - Streamlit Application
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add scripts directory to path
sys.path.append('scripts')

# Import UI components
from ui.components import setup_page_config, create_sidebar
from ui.pages import (
    upload_and_process_page,
    query_system_page,
    statistics_page,
    settings_page
)

def main():
    """Main application function"""
    # Setup page configuration
    setup_page_config()
    
    # Create sidebar and get current page
    current_page = create_sidebar()
    
    # Initialize session state
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    if 'runtime_settings' not in st.session_state:
        st.session_state.runtime_settings = {}
    
    # Route to appropriate page
    if current_page == "📤 Upload & Process":
        upload_and_process_page()
    elif current_page == "🔍 Query System":
        query_system_page()
    elif current_page == "📊 Statistics":
        statistics_page()
    elif current_page == "⚙️ Settings":
        settings_page()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>RAG System with LM Studio + Qdrant | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 