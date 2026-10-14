import os
import sys
import traceback

import streamlit as st


# =========================================================================
# 0. PROJECT PATH CONFIGURATION
# =========================================================================

project_root = os.path.dirname(os.path.abspath(__file__))

if project_root not in sys.path:
    sys.path.insert(0, project_root)


# =========================================================================
# 1. STREAMLIT CONFIGURATION
# =========================================================================

st.set_page_config(
    page_title="Vietnam Tourism AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================================
# 2. SESSION STATE INITIALIZATION
# =========================================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_references" not in st.session_state:
    st.session_state.last_references = []


# =========================================================================
# 3. SIDEBAR CONFIGURATION
# =========================================================================

with st.sidebar:
    st.markdown(
        """
        <h2 style="
            font-weight: 700;
            font-size: 1.3rem;
            margin-bottom: 5px;
        ">
            🤖 AI Assistant Panel
        </h2>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p style="
            color: #8b949e;
            font-size: 0.85rem;
            margin-bottom: 20px;
        ">
            Enterprise Hybrid RAG Platform
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <hr style="
            border-top: 1px solid #30363d;
            margin-top: 0;
            margin-bottom: 20px;
        ">
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p style="
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 8px;
        ">
            THEME ENGINE
        </p>
        """,
        unsafe_allow_html=True,
    )

    theme_selection = st.selectbox(
        "Active Theme Profile:",
        options=[
            "Streamlit Light",
            "IEEE Light",
            "Nordic Frost",
            "Monokai Retro",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown(
        "<div style='margin-top: 30px;'></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p style="
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 8px;
        ">
            PIPELINE PARAMETERS
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Top-K Retrieval Source: 10 Chunks")
    st.caption("Top-N Reranker Context: 3 Chunks")

    st.markdown(
        """
        <hr style="
            border-top: 1px solid #30363d;
            margin-top: 20px;
            margin-bottom: 15px;
        ">
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p style="
            font-size: 0.8rem;
            color: #8b949e;
        ">
            <b>Host:</b> Local Machine<br>
            <b>LLM:</b> Qwen2.5 14B via Ollama
        </p>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================
# 4. THEME CONFIGURATION
# =========================================================================

theme_palette = {
    "Streamlit Light": {
        "bg": "#fdfdfd",
        "card": "#f0f2f6",
        "border": "#e0e4ed",
        "text": "#31333f",
        "muted": "#555867",
        "accent": "#ff4b4b",
        "badge_bg": "rgba(255, 75, 75, 0.1)",
        "badge_txt": "#ff4b4b",
    },
    "IEEE Light": {
        "bg": "#f8f9fa",
        "card": "#ffffff",
        "border": "#d1d5db",
        "text": "#111827",
        "muted": "#4b5563",
        "accent": "#2563eb",
        "badge_bg": "rgba(16, 185, 129, 0.15)",
        "badge_txt": "#059669",
    },
    "Nordic Frost": {
        "bg": "#1a2332",
        "card": "#222e43",
        "border": "#3b4d6d",
        "text": "#ecf0f6",
        "muted": "#94a3b8",
        "accent": "#38bdf8",
        "badge_bg": "rgba(14, 165, 233, 0.15)",
        "badge_txt": "#38bdf8",
    },
    "Monokai Retro": {
        "bg": "#121212",
        "card": "#1e1e1e",
        "border": "#333333",
        "text": "#f8f8f2",
        "muted": "#75715e",
        "accent": "#a6e22e",
        "badge_bg": "rgba(230, 219, 116, 0.15)",
        "badge_txt": "#e6db74",
    },
}

active_cfg = theme_palette[theme_selection]


# =========================================================================
# 5. CUSTOM CSS
# =========================================================================

st.markdown(
    f"""
    <style>
        .reportview-container,
        .main {{
            background-color: {active_cfg["bg"]} !important;
            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                sans-serif;
        }}

        .stChatMessage {{
            background-color: {active_cfg["card"]} !important;
            border: 1px solid {active_cfg["border"]} !important;
            border-radius: 8px !important;
            padding: 15px !important;
            margin-bottom: 12px !important;
            color: {active_cfg["text"]} !important;
        }}

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

        .score-badge {{
            background-color: {active_cfg["badge_bg"]};
            color: {active_cfg["badge_txt"]};
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-family: monospace;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================================
# 6. ERROR REPORTING FUNCTION
# =========================================================================

def display_full_error(error, stage_name):
    """
    Print the full traceback in the terminal and display it in Streamlit.
    """
    full_traceback = traceback.format_exc()

    print("\n" + "!" * 80, flush=True)
    print(f"[{stage_name}]", flush=True)
    print(f"Exception type   : {type(error).__name__}", flush=True)
    print(f"Exception message: {error}", flush=True)
    print(f"Exception repr   : {repr(error)}", flush=True)
    print("Full traceback:", flush=True)
    print(full_traceback, flush=True)
    print("!" * 80 + "\n", flush=True)

    # Do not expose filesystem paths, environment values or internals to users.
    st.error(f"{stage_name}: Hệ thống gặp lỗi. Xem terminal để biết chi tiết.")
    if os.environ.get("TOURISM_DEBUG_ERRORS") == "1":
        with st.expander("Chi tiết lỗi (chế độ phát triển)"):
            st.code(full_traceback, language="text")


# =========================================================================
# 7. BACKEND CORE HOT-LOAD ROUTINE
# =========================================================================

@st.cache_resource
def load_rag_system():
    """
    Lazily import and initialize the RAG pipeline.

    Importing inside this function prevents the models from being loaded
    before Streamlit reaches the cached initialization stage.
    """
    from src.run import RAGPipeline

    return RAGPipeline()


try:
    with st.spinner(
        "🔄 INITIALIZATION: "
        "Loading Tourism Knowledge-Base & Vector Indices..."
    ):
        pipeline = load_rag_system()

except Exception as error:
    display_full_error(
        error,
        "STREAMLIT INITIALIZATION ERROR",
    )
    st.stop()


# =========================================================================
# 8. MAIN LAYOUT
# =========================================================================

chat_col, ref_col = st.columns(
    [5, 3],
    gap="large",
)


# =========================================================================
# 9. CHAT INTERFACE
# =========================================================================

with chat_col:
    st.markdown(
        f"""
        <h2 style="
            font-weight: 700;
            color: {active_cfg["text"]};
            letter-spacing: -0.5px;
            margin-bottom: 5px;
        ">
            💬 Conversational Intelligence Hub
        </h2>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <p style="
            color: {active_cfg["muted"]};
            font-size: 0.9rem;
            margin-bottom: 20px;
        ">
            Ask anything about Vietnamese tourism destinations
            and research data insights.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Display existing chat history.
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Receive a new user question.
    user_input = st.chat_input(
        "Message the RAG Engine..."
    )

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        with st.spinner("Analyzing data vectors..."):
            try:
                print("\n" + "=" * 80, flush=True)
                print("[PIPELINE] New query received", flush=True)
                print(f"[PIPELINE] Query: {user_input!r}", flush=True)
                print("=" * 80, flush=True)

                output = pipeline.execute_pipeline(
                    user_input,
                    top_k_retrieval=10,
                    top_n_rerank=3,
                )

                if not isinstance(output, dict):
                    raise TypeError(
                        "execute_pipeline() must return a dictionary, "
                        f"but returned {type(output).__name__}."
                    )

                answer = output.get(
                    "answer",
                    "No synthesized response compiled.",
                )

                references = output.get(
                    "retrieved_context",
                    [],
                )

                if references is None:
                    references = []

                if not isinstance(references, list):
                    raise TypeError(
                        "retrieved_context must be a list, "
                        f"but received {type(references).__name__}."
                    )

                st.session_state.last_references = references

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                with st.chat_message("assistant"):
                    st.markdown(answer)

                print(
                    "[PIPELINE] Query completed successfully.",
                    flush=True,
                )

                st.rerun()

            except Exception as error:
                st.session_state.last_references = []
                display_full_error(error, "STREAMLIT PIPELINE ERROR")
                # Record a user-readable assistant reply, without leaking details.
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "Xin lỗi, hệ thống gặp lỗi khi xử lý câu hỏi. Vui lòng thử lại.",
                })


# =========================================================================
# 10. EVIDENCE REFERENCES
# =========================================================================

with ref_col:
    st.markdown(
        f"""
        <h2 style="
            font-weight: 700;
            color: {active_cfg["text"]};
            letter-spacing: -0.5px;
            margin-bottom: 5px;
        ">
            📄 Evidence References
        </h2>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <p style="
            color: {active_cfg["muted"]};
            font-size: 0.9rem;
            margin-bottom: 20px;
        ">
            Retrieved context passages used by the generator;
            they do not guarantee answer correctness.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.last_references:
        for index, chunk in enumerate(
            st.session_state.last_references,
            start=1,
        ):
            try:
                if not isinstance(chunk, dict):
                    raise TypeError(
                        f"Reference chunk {index} is not a dictionary."
                    )

                chunk_id = chunk.get(
                    "chunk_id",
                    f"unknown-{index}",
                )

                chunk_text = chunk.get(
                    "text",
                    "No document text available.",
                )

                score_value = chunk.get("rerank_score")
                score_label = "Rerank Score"
                if score_value is None:
                    score_label = "RRF Score"
                    score_value = chunk.get("rrf_score", 0.0)

                try:
                    score_value = float(score_value)
                except (TypeError, ValueError):
                    score_value = 0.0

                header_text = (
                    f"Node {index} | "
                    f"Chunk ID: {chunk_id} | "
                    f"{score_label}: {score_value:.4f}"
                )

                with st.expander(
                    header_text,
                    expanded=(index == 1),
                ):
                    # Evidence text comes from the corpus: never interpolate it
                    # into unsafe HTML. Streamlit Markdown escapes raw HTML.
                    st.text(str(chunk_text))

                    st.markdown(
                        f"""
                        <p style="
                            color: {active_cfg["badge_txt"]};
                            font-size: 0.75rem;
                            margin-top: 8px;
                            margin-bottom: 0;
                        ">
                            Source: Vietnam Tourism Production Dataset
                        </p>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as error:
                display_full_error(
                    error,
                    f"REFERENCE RENDERING ERROR AT NODE {index}",
                )

    else:
        st.markdown(
            f"""
            <div style="
                background-color: {active_cfg["card"]};
                border: 1px dashed {active_cfg["border"]};
                padding: 40px 20px;
                text-align: center;
                border-radius: 8px;
                color: {active_cfg["muted"]};
                font-size: 0.9rem;
                margin-top: 10px;
            ">
                Awaiting interactive prompt execution
                to stream extracted citation evidences.
            </div>
            """,
            unsafe_allow_html=True,
        )