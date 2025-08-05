import streamlit as st

def create_sidebar():
    """Create the sidebar navigation"""
    st.sidebar.title("🤖 RAG System")
    
    # Navigation
    st.sidebar.subheader("Navigation")
    
    # Use buttons for navigation
    if st.sidebar.button("📤 Upload & Process", use_container_width=True):
        st.session_state.current_page = "📤 Upload & Process"
    
    if st.sidebar.button("🔍 Query System", use_container_width=True):
        st.session_state.current_page = "🔍 Query System"
    
    if st.sidebar.button("📊 Statistics", use_container_width=True):
        st.session_state.current_page = "📊 Statistics"
    
    if st.sidebar.button("⚙️ Settings", use_container_width=True):
        st.session_state.current_page = "⚙️ Settings"
    
    # Get current page from session state, default to first page
    page = st.session_state.get('current_page', "📤 Upload & Process")
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    
    # Check Qdrant connection
    try:
        from scripts.embeddings import embedding_manager
        stats = embedding_manager.get_collection_info()
        st.sidebar.success("✅ Qdrant Connected")
        st.sidebar.info(f"Chunks: {stats['points_count']}")
    except Exception as e:
        st.sidebar.error("❌ Qdrant Disconnected")
    
    # Check LM Studio connection
    try:
        from scripts.query import RAGQueryEngine
        query_engine = RAGQueryEngine()
        # Test with a simple query
        test_result = query_engine.query("test", top_k=1)
        st.sidebar.success("✅ LM Studio Connected")
    except Exception as e:
        st.sidebar.error("❌ LM Studio Disconnected")
    
    return page 