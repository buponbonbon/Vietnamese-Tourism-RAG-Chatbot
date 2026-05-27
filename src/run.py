import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval import HybridRetriever
from src.reranking import TourismReranker
from src.generation import LanguageGenerator

class RAGPipeline:
    def __init__(self):
        print("\n==================================================")
        print("     COUPLED HYBRID RAG SYSTEM INITIALIZATION     ")
        print("==================================================")

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_file_path = os.path.join(BASE_DIR, "data", "processed", "vietnam_tourism_rag_dataset.csv")
        index_file_path = os.path.join(BASE_DIR, "data", "processed", "inverted_index.json")
        faiss_dir = os.path.join(BASE_DIR, "data", "vectors")

        # Khởi chạy 3 tầng độc lập, tuyệt đối không chèn self.client = OpenAI vào đây
        self.retriever = HybridRetriever(csv_path=csv_file_path, index_path=index_file_path, faiss_index_dir=faiss_dir)
        self.reranker = TourismReranker()
        self.generator = LanguageGenerator()

        print("\n[SUCCESS] Entire Hybrid RAG Pipeline is hot-loaded and online.")

    def execute_pipeline(self, query: str, top_k_retrieval: int = 10, top_n_rerank: int = 3) -> dict:
        """
        Executes the end-to-end RAG workflow: Retrieve -> Merge -> Rerank -> Generate.
        """
        print(f"\n[QUERY RECEIVED]: '{query}'")

        # 1. Execute Dual-Path Retrieval (BM25 + BGE-M3)
        bm25_res = self.retriever.bm25_search(query, top_k=top_k_retrieval)
        semantic_res = self.retriever.semantic_search(query, top_k=top_k_retrieval)

        # 2. Merge via Reciprocal Rank Fusion (RRF)
        # Output is list of tuples: [(chunk_id, rrf_score), ...]
        hybrid_tuples = self.retriever.rrf_merge(bm25_res, semantic_res, top_n=top_k_retrieval)

        # FIXED: Convert tuples to dictionary format with text content that reranker expects
        hybrid_chunks = []
        for chunk_id, rrf_score in hybrid_tuples:
            text_content = self.retriever.df[self.retriever.df['chunk_id'] == chunk_id].iloc[0]['text_content']
            hybrid_chunks.append({
                "chunk_id": chunk_id,
                "text": text_content,
                "rrf_score": rrf_score
            })

        # 3. Execute Neural Fine-Grained Reranking
        reranked_chunks = self.reranker.rerank(query, hybrid_chunks)

        # Slice the top N most relevant chunks to serve as the factual prompt context
        final_context_chunks = reranked_chunks[:top_n_rerank]

        # 4. Dispatch to DeepSeek Cloud API for Context-Grounded Generation
        generated_answer = self.generator.generate_answer(query, final_context_chunks)

        return {
            "query": query,
            "retrieved_context": final_context_chunks,
            "answer": generated_answer
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()
    sample_query = "Kinh nghiệm đi Vịnh Hạ Long"
    pipeline_output = pipeline.execute_pipeline(sample_query, top_k_retrieval=10, top_n_rerank=3)

    print("\n==================================================")
    print("             FINAL END-TO-END RESPONSE            ")
    print("==================================================")
    print(f"LLM ANSWER:\n{pipeline_output['answer']}")