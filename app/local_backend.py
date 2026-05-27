import os
import sys
import re
from flask import Flask, render_template, request, abort

# Add project root to path to import src modules seamlessly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.run import RAGPipeline

app = Flask(__name__)

# Initialize the RAG Pipeline globally
try:
    rag_pipeline = RAGPipeline()
except Exception as e:
    print(f"[CRITICAL] Failed to load RAG Pipeline: {str(e)}")
    rag_pipeline = None


def highlight_keywords(text, query):
    """Highlights query keywords inside the text using HTML mark tags."""
    if not query:
        return text
    # Split query into individual words to highlight phrases/words
    keywords = re.findall(r'\w+', query)
    for kw in set(keywords):
        if len(kw) > 1:  # Avoid highlighting single characters
            compiled_regex = re.compile(r'\b(' + re.escape(kw) + r')\b', re.IGNORECASE)
            text = compiled_regex.sub(r'<mark class="keyword-highlight">\1</mark>', text)
    return text


@app.route('/')
def home():
    """Renders the minimalist academic search homepage."""
    return render_template('index.html')


@app.route('/search')
def search():
    """Executes search, handles pagination, and serves the results interface."""
    query = request.args.get('q', '').strip()
    page = request.args.get('page', 1, type=int)
    per_page = 3  # Number of document references displayed per page

    if not query:
        return render_template('index.html')

    if rag_pipeline is None:
        abort(500, description="RAG Pipeline Core Engine is offline.")

    try:
        # 1. Execute full pipeline routing (Retrieve -> Merge -> Rerank -> Generate)
        # We retrieve top_k=10 to have enough data for multi-page manual slicing if needed
        pipeline_response = rag_pipeline.execute_pipeline(query, top_k_retrieval=10, top_n_rerank=6)

        all_chunks = pipeline_response.get("retrieved_context", [])
        llm_answer = pipeline_response.get("answer", "")

        # 2. Manual Pagination Slicing for demonstration
        total_results = len(all_chunks)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_chunks = all_chunks[start_idx:end_idx]

        total_pages = (total_results + per_page - 1) // per_page

        # 3. Apply keyword highlighting to the text contents
        for chunk in paginated_chunks:
            chunk['highlighted_text'] = highlight_keywords(chunk['text'], query)

        return render_template(
            'results.html',
            query=query,
            chunks=paginated_chunks,
            llm_answer=llm_answer,
            current_page=page,
            total_pages=total_pages,
            total_results=total_results
        )
    except Exception as e:
        print(f"[RUNTIME ERROR] Search execution crashed: {str(e)}")
        abort(500)


if __name__ == '__main__':
    app.run(debug=True, port=8501)