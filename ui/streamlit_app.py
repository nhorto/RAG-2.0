"""Streamlit UI for RAG system."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src.retrieval.query_processor import QueryProcessor
from src.retrieval.hybrid_searcher import HybridSearcher
from src.generation.llm_interface import LLMInterface
from src.database.qdrant_client import QdrantManager, build_metadata_filter


# Page config
st.set_page_config(
    page_title="Tekla PowerFab Consulting Assistant",
    page_icon="🏗️",
    layout="wide",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "components_initialized" not in st.session_state:
    try:
        st.session_state.query_processor = QueryProcessor()
        st.session_state.searcher = HybridSearcher()
        st.session_state.llm = LLMInterface()
        st.session_state.qdrant = QdrantManager()
        st.session_state.components_initialized = True
        st.session_state.init_error = None
    except Exception as e:
        st.session_state.components_initialized = False
        st.session_state.init_error = str(e)


# Title
st.title("🏗️ Tekla PowerFab Consulting Assistant")
st.markdown("*Search your consulting session transcripts and summaries*")

# Check initialization
if not st.session_state.components_initialized:
    st.error(f"Failed to initialize components: {st.session_state.init_error}")
    st.info(
        "Make sure Qdrant is running and your .env file has the required API keys."
    )
    st.stop()

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")

    # Collection info
    try:
        collection_info = st.session_state.qdrant.get_collection_info()
        if collection_info.get("exists"):
            st.info(
                f"📊 Collection: {st.session_state.qdrant.collection_name}\n\n"
                f"Documents: {collection_info.get('points_count', 0):,}"
            )
        else:
            st.warning("Collection not found. Run ingestion first.")
    except Exception:
        st.warning("Could not connect to Qdrant")

    st.markdown("---")

    # Metadata filters
    st.subheader("Metadata Filters")

    client_filter = st.text_input(
        "Client Name",
        help="Filter by specific client",
        placeholder="e.g., ClientA",
    )

    # Date range
    st.write("Date Range")
    col1, col2 = st.columns(2)
    with col1:
        date_start = st.date_input(
            "From",
            value=None,
            help="Start date (leave empty for no filter)",
        )
    with col2:
        date_end = st.date_input(
            "To",
            value=None,
            help="End date (leave empty for no filter)",
        )

    # Document type
    doc_type = st.selectbox(
        "Document Type",
        ["All", "Transcript", "Daily Summary", "Master Summary"],
        help="Filter by document type",
    )

    # Quick date filters
    st.write("Quick Filters")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Last Week"):
            date_end = datetime.now().date()
            date_start = (datetime.now() - timedelta(days=7)).date()
            st.experimental_rerun()
    with col2:
        if st.button("Last Month"):
            date_end = datetime.now().date()
            date_start = (datetime.now() - timedelta(days=30)).date()
            st.experimental_rerun()

    st.markdown("---")

    # Search settings
    st.subheader("⚙️ Search Settings")

    search_mode = st.radio(
        "Search Mode",
        ["Hybrid (Dense + Sparse)", "Dense Only", "Sparse Only"],
        help="Hybrid combines semantic and keyword search",
    )

    num_results = st.slider(
        "Number of Results",
        min_value=3,
        max_value=20,
        value=5,
        help="How many source chunks to retrieve",
    )

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()


# Main chat interface
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"""
                        **Source {source['source_number']}** (Score: {source['score']:.3f})

                        - **Date:** {source.get('date', 'N/A')}
                        - **Client:** {source.get('client', 'N/A')}
                        - **Type:** {source.get('document_type', 'N/A')}
                        - **File:** {source.get('filename', 'N/A')}

                        *Text:* {source['text_preview']}
                        """
                    )
                    st.markdown("---")


# Chat input
if prompt := st.chat_input("Ask a question about your consulting sessions..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating response..."):
            try:
                # Build metadata filter
                filter_kwargs = {}

                if client_filter:
                    filter_kwargs["client_name"] = client_filter

                if date_start:
                    filter_kwargs["date_start"] = str(date_start)

                if date_end:
                    filter_kwargs["date_end"] = str(date_end)

                if doc_type != "All":
                    doc_type_map = {
                        "Transcript": "transcript",
                        "Daily Summary": "daily_summary",
                        "Master Summary": "master_summary",
                    }
                    filter_kwargs["document_type"] = doc_type_map[doc_type]

                metadata_filter = (
                    build_metadata_filter(**filter_kwargs)
                    if filter_kwargs
                    else None
                )

                # Determine search mode
                dense_only = search_mode == "Dense Only"
                sparse_only = search_mode == "Sparse Only"

                # Search
                results = st.session_state.searcher.search(
                    query=prompt,
                    filters=metadata_filter,
                    top_k=num_results,
                    dense_only=dense_only,
                    sparse_only=sparse_only,
                )

                if not results:
                    response_text = (
                        "I couldn't find any relevant information in the knowledge base. "
                        "Try adjusting your filters or rephrasing your question."
                    )
                    sources = []
                else:
                    # Generate response
                    response = st.session_state.llm.generate_response(
                        query=prompt,
                        search_results=results,
                    )

                    response_text = response.answer
                    sources = response.sources

                # Display response
                st.markdown(response_text)

                # Show sources
                if sources:
                    with st.expander("📚 View Sources", expanded=False):
                        for source in sources:
                            st.markdown(
                                f"""
                                **Source {source['source_number']}** (Score: {source['score']:.3f})

                                - **Date:** {source.get('date', 'N/A')}
                                - **Client:** {source.get('client', 'N/A')}
                                - **Type:** {source.get('document_type', 'N/A')}
                                - **File:** {source.get('filename', 'N/A')}

                                *Text:* {source['text_preview']}
                                """
                            )
                            st.markdown("---")

                # Add to message history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "sources": sources,
                    }
                )

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Tekla PowerFab RAG System v2.0 | Powered by OpenAI & Qdrant
    </div>
    """,
    unsafe_allow_html=True,
)
