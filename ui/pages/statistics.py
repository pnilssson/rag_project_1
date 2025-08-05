import streamlit as st
import sys

# Add scripts to path
sys.path.append('scripts')

from ui.components import display_statistics

def statistics_page():
    """Statistics page functionality"""
    st.title("📊 System Statistics")
    
    st.markdown("""
    View system statistics and monitor the health of your RAG system.
    """)
    
    # Display statistics
    display_statistics()
    
    # Additional system info
    st.markdown("---")
    st.subheader("🔧 System Information")
    
    try:
        from scripts.config import config
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Directory:**")
            st.code(config.data_dir)
            
            st.write("**Supported Extensions:**")
            st.code(", ".join(config.supported_extensions))
        
        with col2:
            st.write("**API Endpoints:**")
            st.code(f"Qdrant: {config.qdrant_host}:{config.qdrant_port}")
            st.code(f"LM Studio: {config.openai_api_base}")
        
    except Exception as e:
        st.error(f"Failed to retrieve system information: {e}") 