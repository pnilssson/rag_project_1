import streamlit as st

def setup_page_config():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="RAG System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    ) 