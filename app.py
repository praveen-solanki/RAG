"""
RAG Dashboard - Streamlit UI
A modular, non-chatbot RAG interface with dynamic model selection and file upload
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

# Import backend modules
from ollama_models import get_available_models
import rag_backend
import Gemini_model as gemini_backend

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="RAG Dashboard",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# CUSTOM CSS STYLING
# ==================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #105666;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .answer-box {
        background-color: #979390;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #81555a;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# SESSION STATE INITIALIZATION
# ==================================================

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'comparison_result' not in st.session_state:
    st.session_state.comparison_result = None

# ==================================================
# HELPER FUNCTIONS
# ==================================================

def save_uploaded_files(uploaded_files) -> List[str]:
    """Save uploaded files to temp directory and return paths"""
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(str(file_path))
    
    return saved_paths


def display_answer_with_sources(result: Dict[str, Any], title: str = "Answer"):
    """Display the answer and its source citations"""
    
    # Display answer
    st.markdown(f"### {title}")
    
    answer = result.get('answer', 'No answer generated')
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
    
    # Display sources
    sources = result.get('sources', [])
    
    if sources:
        st.markdown("#### ?? Sources")
        
        for idx, source in enumerate(sources, 1):
            with st.expander(f"Source {idx}: {source.get('filename', 'Unknown')} (Score: {source.get('score', 0):.4f})"):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown("**Metadata:**")
                    st.write(f"?? File: {source.get('filename', 'N/A')}")
                    st.write(f"?? Score: {source.get('score', 0):.4f}")
                    
                    # Additional metadata if available
                    if 'page' in source:
                        st.write(f"?? Page: {source.get('page', 'N/A')}")
                    if 'sheet' in source:
                        st.write(f"?? Sheet: {source.get('sheet', 'N/A')}")
                    if 'chunk_id' in source:
                        st.write(f"?? Chunk: {source.get('chunk_id', 'N/A')}")
                
                with col2:
                    st.markdown("**Excerpt:**")
                    excerpt = source.get('content', 'No content available')
                    st.markdown(f'<div class="source-box">{excerpt}</div>', unsafe_allow_html=True)
    else:
        st.info("No sources available for this answer")


def display_comparison_view(standard_result: Dict, gemini_result: Dict):
    """Display side-by-side comparison of results"""
    
    st.markdown("## ?? Comparative Analysis")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ?? Selected Model")
        model_name = standard_result.get('model', 'Unknown')
        st.info(f"Model: {model_name}")
        
        answer = standard_result.get('answer', 'No answer')
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
        
        sources = standard_result.get('sources', [])
        st.markdown(f"**Sources:** {len(sources)} documents retrieved")
        
        if sources:
            with st.expander("View Sources"):
                for idx, source in enumerate(sources, 1):
                    st.markdown(f"**{idx}. {source.get('filename', 'Unknown')}** (Score: {source.get('score', 0):.4f})")
                    st.caption(source.get('content', '')[:200] + "...")
    
    with col2:
        st.markdown("### ? Gemini Pro Benchmark")
        st.info("Model: Gemini 2.5 Pro")
        
        answer = gemini_result.get('answer', 'No answer')
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
        
        sources = gemini_result.get('sources', [])
        st.markdown(f"**Sources:** {len(sources)} documents retrieved")
        
        if sources:
            with st.expander("View Sources"):
                for idx, source in enumerate(sources, 1):
                    st.markdown(f"**{idx}. {source.get('filename', 'Unknown')}** (Score: {source.get('score', 0):.4f})")
                    st.caption(source.get('content', '')[:200] + "...")


# ==================================================
# MAIN UI
# ==================================================

# Header
st.markdown('<div class="main-header">?? RAG Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Retrieval Augmented Generation with Multi-Model Support</div>', unsafe_allow_html=True)

# ==================================================
# SIDEBAR - FILE UPLOAD
# ==================================================

with st.sidebar:
    st.markdown("## ?? Document Upload")
    st.markdown("Upload your documents to get started")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'xls', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Supported: PDF, Word, Text, Excel, CSV, Images"
    )
    
    # Update session state
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        
        st.success(f"? {len(uploaded_files)} file(s) uploaded")
        
        # Show uploaded files
        with st.expander("View uploaded files"):
            for file in uploaded_files:
                st.text(f"?? {file.name} ({file.size / 1024:.1f} KB)")
    else:
        st.session_state.uploaded_files = []
    
    st.markdown("---")
    
    # Collection selector (optional)
    st.markdown("## ?? Settings")
    
    # Debug mode
    debug_mode = st.checkbox("?? Debug Mode", value=False, help="Show detailed debug information")
    st.session_state.debug_mode = debug_mode
    
    collection_options = [
        "rag_database_384_new",
        "rag_database_ARAI"
    ]
    
    selected_collection = st.selectbox(
        "Vector Database Collection",
        options=collection_options,
        index=1,  # Default to improved
        help="Select which Qdrant collection to use"
    )
    
    # Store in session state
    st.session_state.collection = selected_collection

# ==================================================
# MAIN AREA - STALE-STATE PROTECTION
# ==================================================

if not st.session_state.uploaded_files:
    # Show warning when no files uploaded
    st.markdown("""
    <div class="warning-box">
        <h3>?? No Documents Uploaded</h3>
        <p>Please upload documents using the sidebar to begin.</p>
        <p><strong>Steps:</strong></p>
        <ol>
            <li>Use the file uploader on the left</li>
            <li>Select one or more documents (PDF, Word, Excel, CSV, Images)</li>
            <li>Ask your question below</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Show example
    st.markdown("### ?? Example Questions")
    st.info("""
    - What is the company name mentioned in the document?
    - Who are the auditors?
    - What are the total assets?
    - Summarize the key findings.
    """)
    
    st.stop()

# ==================================================
# FILES ARE UPLOADED - SHOW FULL INTERFACE
# ==================================================

# Question Input
st.markdown("## ? Ask Your Question")

question = st.text_area(
    "Enter your question",
    height=120,
    placeholder="What would you like to know about your documents?",
    help="Ask specific questions about the uploaded documents"
)

st.markdown("---")

# Model Selection (only shown when files are uploaded)
st.markdown("## ?? Model Selection")

# Load available models
try:
    available_models = get_available_models()
    
    if not available_models:
        st.error("? No Ollama models found. Please ensure Ollama is running and models are pulled.")
        st.stop()
    
    # Model selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Choose AI Model",
            options=available_models,
            help="Select which model to use for answering questions"
        )
    
    with col2:
        st.metric("Available Models", len(available_models))

except Exception as e:
    st.error(f"? Error loading models: {str(e)}")
    st.stop()

st.markdown("---")

# ==================================================
# ACTION BUTTONS
# ==================================================

st.markdown("## ?? Generate Answer")

col1, col2 = st.columns(2)

with col1:
    generate_button = st.button(
        "?? Generate Answer",
        type="primary",
        use_container_width=True,
        help="Generate answer using selected model"
    )

with col2:
    compare_button = st.button(
        "?? Evaluate with Gemini",
        type="secondary",
        use_container_width=True,
        help="Compare selected model with Gemini Pro"
    )

# ==================================================
# EXECUTION MODE 1 - STANDARD GENERATION
# ==================================================

if generate_button:
    if not question.strip():
        st.warning("?? Please enter a question first")
    else:
        with st.spinner(f"?? Generating answer using {selected_model}..."):
            try:
                # Debug info
                st.info(f"Processing with model: {selected_model}")
                st.info(f"Collection: {st.session_state.collection}")
                st.info(f"Files: {len(st.session_state.uploaded_files)}")
                
                # Call RAG backend
                result = rag_backend.generate_answer(
                    question=question,
                    model_name=selected_model,
                    file_objects=st.session_state.uploaded_files,
                    collection=st.session_state.collection
                )
                
                # Debug result
                st.success("? Generation complete!")
                
                # Store result
                st.session_state.last_result = result
                st.session_state.last_question = question
                
                # Force rerun to display result
                st.rerun()
                
            except Exception as e:
                st.error(f"? Error generating answer: {str(e)}")
                st.exception(e)
                
                # Show detailed traceback
                import traceback
                st.code(traceback.format_exc())

# ==================================================
# EXECUTION MODE 2 - COMPARISON WITH GEMINI
# ==================================================

if compare_button:
    if not question.strip():
        st.warning("?? Please enter a question first")
    else:
        # Check for Gemini API key
        if "GEMINI_API_KEY" not in os.environ:
            st.error("? GEMINI_API_KEY environment variable not set. Cannot run comparison.")
        else:
            with st.spinner("?? Running comparison analysis..."):
                try:
                    # Debug info
                    st.info(f"Running comparison: {selected_model} vs Gemini")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Generate with selected model
                    status_text.text(f"Running {selected_model}...")
                    progress_bar.progress(25)
                    
                    standard_result = rag_backend.generate_answer(
                        question=question,
                        model_name=selected_model,
                        file_objects=st.session_state.uploaded_files,
                        collection=st.session_state.collection
                    )
                    
                    progress_bar.progress(50)
                    
                    # Generate with Gemini
                    status_text.text("Running Gemini Pro...")
                    progress_bar.progress(75)
                    
                    gemini_result = gemini_backend.generate_answer(
                        question=question,
                        file_objects=st.session_state.uploaded_files,
                        collection=st.session_state.collection
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                    
                    # Store results
                    st.session_state.comparison_result = {
                        'standard': standard_result,
                        'gemini': gemini_result
                    }
                    st.session_state.last_question = question
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Force rerun to display results
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"? Error during comparison: {str(e)}")
                    st.exception(e)
                    
                    import traceback
                    st.code(traceback.format_exc())

# ==================================================
# DISPLAY PERSISTENT RESULTS
# ==================================================

# Always show results section if they exist
if st.session_state.last_result or st.session_state.comparison_result:
    st.markdown("---")
    st.markdown("## ?? Results")

# Show last single result
if st.session_state.last_result and not st.session_state.comparison_result:
    st.markdown("### ?? Answer")
    st.caption(f"**Question:** {st.session_state.last_question}")
    display_answer_with_sources(st.session_state.last_result)

# Show last comparison
elif st.session_state.comparison_result:
    st.markdown("### ?? Comparative Analysis")
    st.caption(f"**Question:** {st.session_state.last_question}")
    display_comparison_view(
        st.session_state.comparison_result['standard'],
        st.session_state.comparison_result['gemini']
    )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>RAG Dashboard v1.0 | Powered by Ollama & Gemini | Built with Streamlit</small>
</div>
""", unsafe_allow_html=True)