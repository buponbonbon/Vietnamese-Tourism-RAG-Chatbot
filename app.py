import os
import sys
import streamlit as st

# Ensure core backend modules are discoverable within the global system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.run import RAGPipeline

# =========================================================================
# 1. STREAMLIT CONFIGURATION & PERSISTENT SESSION INITIALIZATION
# =========================================================================
st.set_page_config(
    page_title="Vietnam Tourism AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize persistent chat history structures for fluid UI rendering
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_references" not in st.session_state:
    st.session_state.last_references = []

# =========================================================================
# 2. SIDEBAR CONFIGURATION & THEME ENGINE CONTROL
# =========================================================================
with st.sidebar:
    st.markdown('<h2 style="font-weight: 700; font-size: 1.3rem; margin-bottom: 5px;">🤖 AI Assistant Panel</h2>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color: #8b949e; font-size: 0.85rem; margin-bottom: 20px;">Enterprise Hybrid RAG Platform</p>',
        unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 1px solid #30363d; margin-top: 0; margin-bottom: 20px;'>",
                unsafe_allow_html=True)

    st.markdown('<p style="font-weight: 600; font-size: 0.9rem; margin-bottom: 8px;">THEME ENGINE</p>',
                unsafe_allow_html=True)
    theme_selection = st.selectbox(
        "Active Theme Profile:",
        options=["GitHub Dark", "IEEE Light", "Nordic Frost", "Monokai Retro"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    st.markdown('<p style="font-weight: 600; font-size: 0.9rem; margin-bottom: 8px;">PIPELINE PARAMETERS</p>',
                unsafe_allow_html=True)
    st.caption("Top-K Retrieval Source: 10 Chunks")
    st.caption("Top-N Reranker Context: 3 Chunks")
    st.markdown("<hr style='border-top: 1px solid #30363d; margin-top: 20px; margin-bottom: 15px;'>",
                unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size: 0.8rem; color: #8b949e;"><b>Host:</b> Hugging Face Spaces<br><b>LLM:</b> Llama-3.3-70B-Instruct</p>',
        unsafe_allow_html=True)

# Map UI colors dynamically based on selected vector token
theme_palette = {
    "GitHub Dark": {
        "bg": "#0e1117", "card": "#161b22", "border": "#30363d", "text": "#e6edf3",
        "muted": "#8b949e", "accent": "#58a6ff", "badge_bg": "rgba(46, 160, 67, 0.15)", "badge_txt": "#3fb950"
    },
    "IEEE Light": {
        "bg": "#f8f9fa", "card": "#ffffff", "border": "#d1d5db", "text": "#111827",
        "muted": "#4b5563", "accent": "#2563eb", "badge_bg": "rgba(16, 185, 129, 0.15)", "badge_txt": "#059669"
    },
    "Nordic Frost": {
        "bg": "#1a2332", "card": "#222e43", "border": "#3b4d6d", "text": "#ecf0f6",
        "muted": "#94a3b8", "accent": "#38bdf8", "badge_bg": "rgba(14, 165, 233, 0.15)", "badge_txt": "#38bdf8"
    },
    "Monokai Retro": {
        "bg": "#121212", "card": "#1e1e1e", "border": "#333333", "text": "#f8f8f2",
        "muted": "#75715e", "accent": "#a6e22e", "badge_bg": "rgba(230, 219, 116, 0.15)", "badge_txt": "#e6db74"
    }
}

active_cfg = theme_palette[theme_selection]

# =========================================================================
# 3. ADVANCED ACADEMIC CUSTOM CSS INJECTION
# =========================================================================
st.markdown(f"""
    <style>
        .reportview-container, .main {{
            background-color: {active_cfg["bg"]} !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        /* Modern chat bubbles styling modifications */
        .stChatMessage {{
            background-color: {active_cfg["card"]} !important;
            border: 1px solid {active_cfg["border"]} !important;
            border-radius: 8px !important;
            padding: 15px !important;
            margin-bottom: 12px !important;
            color: {active_cfg["text"]} !important;
        }}

        /* Expanders mapping containing retrieved context chunks */
        .streamlit-expanderHeader {{
            background-color: {active_cfg["card"]} !important;
            border: 1px solid {active_cfg["border"]} !important;
            border-radius: 6px !important;
            color: {active_cfg["accent"]} !important;
        }}
        .streamlit-expanderContent {{
            background-color: {active_cfg["bg"]} !important;
            border: 1px solid {active_cfg["border"]} !important;
            color: {active_cfg["muted"]} !important;
            font-size: 0.9rem !important;
        }}

        /* Neural Reranker Metric indicator badge */
        .score-badge {{
            background-color: {active_cfg["badge_bg"]};
            color: {active_cfg["badge_txt"]};
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-family: monospace;
        }}
    </style>
""", unsafe_allow_html=True)


# =========================================================================
# 4. BACKEND CORE HOT-LOAD ROUTINE
# =========================================================================
@st.cache_resource
def load_rag_system():
    return RAGPipeline()


try:
    with st.spinner("🔄 INITIALIZATION: Loading Tourism Knowledge-Base & Vector Indices..."):
        pipeline = load_rag_system()
except Exception as e:
    st.error(f"System Critical Failure: {str(e)}")
    st.stop()

# =========================================================================
# 5. ASYMMETRIC MAIN LAYOUT RENDERING (Chat Interface vs Side References)
# =========================================================================
chat_col, ref_col = st.columns([5, 3], gap="large")

with chat_col:
    st.markdown(
        f'<h2 style="font-weight: 700; color: {active_cfg["text"]}; letter-spacing: -0.5px; margin-bottom: 5px;">💬 Conversational Intelligence Hub</h2>',
        unsafe_allow_html=True)
    st.markdown(
        f'<p style="color: {active_cfg["muted"]}; font-size: 0.9rem; margin-bottom: 20px;">Ask anything about Vietnamese tourism destinations and research data insights.</p>',
        unsafe_allow_html=True)

    # Render persistent chat history logs dynamically on screen re-runs
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Active bottom chat text input container gateway
    if user_input := st.chat_input("Message the RAG Engine..."):
        # Display the user query immediately in conversational space
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Dispatch parallel pipeline retrieval mechanisms
        with st.spinner("Analyzing data vectors..."):
            try:
                output = pipeline.execute_pipeline(user_input, top_k_retrieval=10, top_n_rerank=3)
                ans = output.get("answer", "No synthesized response compiled.")

                # Update viewport state variables for real-time reference layout syncing
                st.session_state.last_references = output.get("retrieved_context", [])

                # Display the generative system output responses
                with st.chat_message("assistant"):
                    st.markdown(ans)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})

                # Force browser to trigger a programmatic viewport re-run to update the right column
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline Execution Exception: {str(e)}")

with ref_col:
    st.markdown(
        f'<h2 style="font-weight: 700; color: {active_cfg["text"]}; letter-spacing: -0.5px; margin-bottom: 5px;">📄 Evidence References</h2>',
        unsafe_allow_html=True)
    st.markdown(
        f'<p style="color: {active_cfg["muted"]}; font-size: 0.9rem; margin-bottom: 20px;">Top neural reranked document context nodes validating the response logic.</p>',
        unsafe_allow_html=True)

    if st.session_state.last_references:
        # Loop and render chunks beautifully into the reference column panel
        for i, chunk in enumerate(st.session_state.last_references, 1):
            score = chunk.get('rerank_score') or chunk.get('rrf_score', 0)
            header_html = f"Node {i} | Chunk ID: {chunk['chunk_id']} &nbsp;&nbsp;<span class='score-badge'>Rerank Score: {score:.4f}</span>"

            with st.expander(header_html, expanded=(i == 1)):
                st.markdown(f"<div style='white-space: pre-line; line-height:1.5;'>{chunk['text']}</div>",
                            unsafe_allow_html=True)
                st.markdown(
                    f"<p style='color: {active_cfg['badge_txt']}; font-size: 0.75rem; margin-top: 8px; margin-bottom: 0;'>Source: Vietnam Tourism Production Dataset</p>",
                    unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div style="background-color: {active_cfg["card"]}; border: 1px dashed {active_cfg["border"]}; padding: 40px 20px; text-align: center; border-radius: 8px; color: {active_cfg["muted"]}; font-size: 0.9rem; margin-top: 10px;">'
            'Awaiting interactive prompt execution to stream extracted citation evidences.'
            '</div>',
            unsafe_allow_html=True
        )