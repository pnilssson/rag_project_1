import streamlit as st

def show_error_with_details(error: Exception, title: str = "An error occurred"):
    """Display error with expandable details"""
    st.error(title)
    with st.expander("🔍 Error Details"):
        st.exception(error) 