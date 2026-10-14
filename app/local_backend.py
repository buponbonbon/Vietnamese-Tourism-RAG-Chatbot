import multiprocessing
import os
import re
import sys
import traceback
from functools import lru_cache

from flask import Flask, abort, render_template, request
from markupsafe import Markup, escape


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

current_file = os.path.abspath(__file__)
app_directory = os.path.dirname(current_file)
project_root = os.path.dirname(app_directory)

if project_root not in sys.path:
    sys.path.insert(0, project_root)


# =============================================================================
# FLASK APPLICATION
# =============================================================================

app = Flask(__name__)

app.config["PROPAGATE_EXCEPTIONS"] = False
app.config["TRAP_HTTP_EXCEPTIONS"] = False


# Chỉ khai báo ở đây, tuyệt đối không tạo RAGPipeline tại cấp toàn cục.
rag_pipeline = None


# =============================================================================
# PIPELINE INITIALIZATION
# =============================================================================

def initialize_rag_pipeline():
    """
    Import và khởi tạo pipeline sau khi chương trình đã đi vào main.
    Việc này tránh lỗi multiprocessing spawn trên Windows.
    """
    global rag_pipeline

    try:
        print("\n" + "=" * 80, flush=True)
        print("[STARTUP] INITIALIZING VIETNAMESE TOURISM RAG BACKEND", flush=True)
        print("=" * 80, flush=True)
        print(f"[STARTUP] Current file : {current_file}", flush=True)
        print(f"[STARTUP] App directory: {app_directory}", flush=True)
        print(f"[STARTUP] Project root : {project_root}", flush=True)
        print(f"[STARTUP] Python       : {sys.executable}", flush=True)
        print(f"[STARTUP] Working dir  : {os.getcwd()}", flush=True)
        print("=" * 80, flush=True)

        print("[STARTUP] Importing RAGPipeline from src.run ...", flush=True)

        # Import bên trong hàm, không import ở đầu file.
        from src.run import RAGPipeline

        print("[STARTUP] RAGPipeline import succeeded.", flush=True)
        print("[STARTUP] Creating RAGPipeline instance ...", flush=True)

        rag_pipeline = RAGPipeline()
        execute_query_cached.cache_clear()

        print("[STARTUP] RAGPipeline initialized successfully.", flush=True)

    except Exception as error:
        rag_pipeline = None

        print("\n" + "!" * 80, flush=True)
        print("[CRITICAL] FAILED TO INITIALIZE RAG PIPELINE", flush=True)
        print(f"[CRITICAL] Exception type: {type(error).__name__}", flush=True)
        print(f"[CRITICAL] Message       : {error}", flush=True)
        print(f"[CRITICAL] repr(error)   : {repr(error)}", flush=True)
        print("[CRITICAL] Full traceback:", flush=True)
        traceback.print_exc()
        print("!" * 80 + "\n", flush=True)

        raise


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def highlight_keywords(text, query):
    """Return HTML-safe highlighted text without trusting corpus or query HTML.

    Escape each raw segment BEFORE inserting constant <mark> elements. Returning
    Markup lets Jinja render just these approved tags without using `|safe`.
    """
    text = "" if text is None else str(text)
    tokens = sorted({tok for tok in re.findall(r"\w+", query or "", flags=re.UNICODE)
                     if len(tok) > 1}, key=len, reverse=True)[:40]
    if not tokens:
        return escape(text)
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, tokens)) + r")\b",
                         flags=re.IGNORECASE | re.UNICODE)
    fragments = []
    pos = 0
    for match in pattern.finditer(text):
        fragments.append(escape(text[pos:match.start()]))
        fragments.append(Markup('<mark class="keyword-highlight">'))
        fragments.append(escape(match.group(0)))
        fragments.append(Markup('</mark>'))
        pos = match.end()
    fragments.append(escape(text[pos:]))
    return Markup('').join(fragments)


@lru_cache(maxsize=16)
def execute_query_cached(query):
    """Avoid rerunning LLM/model inference on every pagination click.

    Cache is local to this single-process dev server. It is cleared when the
    pipeline is reinitialized. Bound its size to limit stored context text.
    """
    if rag_pipeline is None:
        raise RuntimeError("RAG pipeline is not initialized")
    return rag_pipeline.execute_pipeline(query, top_k_retrieval=10, top_n_rerank=6)


def print_exception(stage_name, error):
    """In đầy đủ traceback của lỗi."""
    print("\n" + "!" * 80, flush=True)
    print(f"[RUNTIME ERROR] Stage         : {stage_name}", flush=True)
    print(f"[RUNTIME ERROR] Exception type: {type(error).__name__}", flush=True)
    print(f"[RUNTIME ERROR] Message       : {error}", flush=True)
    print(f"[RUNTIME ERROR] repr(error)   : {repr(error)}", flush=True)
    print("[RUNTIME ERROR] Full traceback:", flush=True)
    traceback.print_exc()
    print("!" * 80 + "\n", flush=True)


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/")
def home():
    try:
        return render_template("index.html")

    except Exception as error:
        print_exception("Rendering index.html", error)

        abort(500, description="Lỗi hiển thị giao diện. Vui lòng xem log máy chủ.")


@app.route("/health")
def health():
    ready = rag_pipeline is not None
    return {"status": "ok" if ready else "pipeline_offline"}, (200 if ready else 503)


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    page = request.args.get("page", 1, type=int)
    if len(query) > 1000:
        abort(400, description="Question must be at most 1000 characters.")
    if page < 1:
        abort(400, description="Page must be at least 1.")
    per_page = 3

    print("\n" + "-" * 80, flush=True)
    print("[REQUEST] GET /search", flush=True)
    print(f"[REQUEST] Query: {query!r}", flush=True)
    print(f"[REQUEST] Page : {page}", flush=True)
    print("-" * 80, flush=True)

    if not query:
        return render_template("index.html")

    if rag_pipeline is None:
        abort(
            500,
            description=(
                "RAG Pipeline chưa được khởi tạo. "
                "Kiểm tra traceback trong terminal."
            ),
        )

    try:
        print("[PIPELINE 1/5] Executing full RAG pipeline ...", flush=True)

        pipeline_response = execute_query_cached(query)

        if not isinstance(pipeline_response, dict):
            raise TypeError(
                "execute_pipeline() phải trả về dict, "
                f"nhưng nhận được {type(pipeline_response).__name__}."
            )

        print("[PIPELINE 2/5] Reading pipeline output ...", flush=True)

        all_chunks = pipeline_response.get("retrieved_context", [])
        llm_answer = pipeline_response.get("answer", "")

        if all_chunks is None:
            all_chunks = []

        if not isinstance(all_chunks, list):
            raise TypeError(
                "retrieved_context phải là list, "
                f"nhưng nhận được {type(all_chunks).__name__}."
            )

        print(f"[PIPELINE 2/5] Retrieved chunks: {len(all_chunks)}", flush=True)

        print("[PIPELINE 3/5] Applying pagination ...", flush=True)

        total_results = len(all_chunks)
        start_idx = max(0, (page - 1) * per_page)
        end_idx = start_idx + per_page

        paginated_chunks = all_chunks[start_idx:end_idx]
        total_pages = (total_results + per_page - 1) // per_page

        print("[PIPELINE 4/5] Highlighting keywords ...", flush=True)

        safe_chunks = []
        for index, chunk in enumerate(paginated_chunks):
            if not isinstance(chunk, dict):
                raise TypeError(
                    f"Chunk {index} phải là dict, "
                    f"nhưng nhận được {type(chunk).__name__}."
                )

            if "text" not in chunk:
                raise KeyError(
                    f"Chunk {index} thiếu khóa 'text'. "
                    f"Các khóa hiện có: {list(chunk.keys())}"
                )

            safe_chunk = dict(chunk)
            safe_chunk["highlighted_text"] = highlight_keywords(chunk["text"], query)
            safe_chunks.append(safe_chunk)

        print("[PIPELINE 5/5] Rendering results.html ...", flush=True)

        return render_template(
            "results.html",
            query=query,
            chunks=safe_chunks,
            llm_answer=llm_answer,
            current_page=page,
            total_pages=total_pages,
            total_results=total_results,
        )

    except Exception as error:
        print_exception("Executing /search pipeline", error)

        abort(500, description="Không thể xử lý truy vấn. Vui lòng xem log máy chủ.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Hữu ích trên Windows và khi đóng gói thành file thực thi.
    multiprocessing.freeze_support()

    # Chỉ khởi tạo model sau khi đã vào main.
    initialize_rag_pipeline()

    port = int(os.environ.get("TOURISM_FLASK_PORT", "5000"))
    if not 1 <= port <= 65535:
        raise ValueError("TOURISM_FLASK_PORT must be between 1 and 65535")
    print(f"\n[SERVER] Starting Flask server at http://127.0.0.1:{port}", flush=True)
    print(f"[SERVER] Health check: http://127.0.0.1:{port}/health", flush=True)
    print("[SERVER] Debug reloader disabled.\n", flush=True)

    app.run(
        host="127.0.0.1",
        port=port,
        debug=False,
        use_reloader=False,
        threaded=False,
    )