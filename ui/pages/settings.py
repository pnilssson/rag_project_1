import streamlit as st
import sys

# Add scripts to path
sys.path.append('scripts')

from ui.components import (
    display_settings,
    show_error_with_details
)

def settings_page():
    """Settings page functionality"""
    st.title("⚙️ Settings")
    
    st.markdown("""
    Configure runtime settings and view system configuration.
    """)
    
    # Display settings
    display_settings()
    
    # Reset options
    st.markdown("---")
    st.subheader("🗑️ System Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Reset Vector Database", type="secondary"):
            try:
                from scripts.embeddings import embedding_manager
                
                if st.session_state.get('confirm_reset', False):
                    with st.spinner("Resetting vector database..."):
                        embedding_manager.delete_collection()
                        embedding_manager.init_collection()
                    st.success("Vector database reset successfully!")
                    st.session_state.confirm_reset = False
                else:
                    st.session_state.confirm_reset = True
                    st.warning("Click the button again to confirm reset")
                    
            except Exception as e:
                show_error_with_details(e, "Reset failed")
    
    with col2:
        if st.button("🧹 Clear Session Data", type="secondary"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Session data cleared!")
            st.rerun()
    
    # Confirmation message
    if st.session_state.get('confirm_reset', False):
        st.warning("⚠️ Click 'Reset Vector Database' again to confirm. This will delete all processed data!") 