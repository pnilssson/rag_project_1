import streamlit as st
import sys

# Add scripts to path
sys.path.append('scripts')

from ui.components import (
    display_query_result,
    show_error_with_details
)

def query_system_page():
    """Query System page functionality"""
    st.title("🔍 Query System")
    
    st.markdown("""
    Ask questions about your documents. The system will search through your processed documents 
    and provide answers based on the relevant content.
    """)
    
    # Query input
    query = st.text_area(
        "Ask a question:",
        placeholder="Enter your question here...",
        height=100
    )
    
    # Query parameters
    with st.expander("🔧 Query Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            from scripts.config import config
            top_k = st.slider("Top K Results", 1, 50, config.top_k)
            similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, config.similarity_threshold, 0.05)
        
        with col2:
            temperature = st.slider("Temperature", 0.0, 1.0, config.temperature, 0.1)
            max_tokens = st.slider("Max Tokens", 100, 2000, config.max_tokens, 100)
    
    # Submit query
    col1, col2 = st.columns([1, 4])
    with col1:
        submit_button = st.button("🔍 Submit Query", type="primary")
    
    with col2:
        if st.button("🗑️ Clear"):
            st.session_state.query_history = []
            st.rerun()
    
    # Process query
    if submit_button and query.strip():
        try:
            from scripts.query import RAGQueryEngine
            
            # Create query engine with custom parameters
            query_engine = RAGQueryEngine()
            
            # Override parameters with UI values
            query_engine.temperature = temperature
            query_engine.max_tokens = max_tokens
            
            # Execute query
            with st.spinner("Searching and generating answer..."):
                result = query_engine.query(
                    query, 
                    top_k=top_k, 
                    score_threshold=similarity_threshold
                )
            
            # Display result
            display_query_result(result)
            
            # Store in query history
            if 'query_history' not in st.session_state:
                st.session_state.query_history = []
            
            st.session_state.query_history.append({
                'query': query,
                'result': result,
                'timestamp': st.session_state.get('timestamp', 0)
            })
            
        except Exception as e:
            show_error_with_details(e, "Query failed")
    
    # Display query history
    if 'query_history' in st.session_state and st.session_state.query_history:
        st.markdown("---")
        st.subheader("📚 Query History")
        
        for i, history_item in enumerate(reversed(st.session_state.query_history)):
            with st.expander(f"Q: {history_item['query'][:50]}..."):
                st.write(f"**Question:** {history_item['query']}")
                st.write(f"**Answer:** {history_item['result'].get('answer', 'No answer')}")
                if history_item['result'].get('sources'):
                    st.write(f"**Sources:** {', '.join(set(history_item['result']['sources']))}") 