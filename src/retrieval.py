import os
import json
import pandas as pd
import numpy as np
import math
import sentence_transformers
import faiss
import torch
print(torch.cuda.is_available())

class HybridRetriever:
    def __init__(self, csv_path, index_path, faiss_index_dir):
        """
        Initialize the Hybrid Retriever with Lexical (BM25) and Semantic (FAISS + BGE-M3) engines.
        """
        self.csv_path = csv_path
        self.index_path = index_path
        self.faiss_index_dir = faiss_index_dir
        self.faiss_index_path = os.path.join(faiss_index_dir, "faiss_index.bin")

        self.df = None
        self.inverted_index = None
        self.chunk_lengths = {}

        # ML/DL models placeholders
        self.model = None
        self.faiss_index = None

        # Load resources sequentially
        self._load_resources()
        self._init_semantic_engine()

    def _load_resources(self):
        """Load text data structures into RAM memory."""
        print("--- [1/3] Loading text retrieval resources ---")

        # Avoid PyArrow-backed string crash on Windows
        pd.options.future.infer_string = False
        pd.options.mode.string_storage = "python"

        self.df = pd.read_csv(
            self.csv_path,
            low_memory=False
        )

        with open(self.index_path, 'r', encoding='utf-8') as f:
            self.inverted_index = json.load(f)

        if 'word_count' not in self.df.columns:
            self.df['word_count'] = self.df['text_content'].apply(lambda x: len(str(x).split()))

        self.chunk_lengths = dict(zip(self.df['chunk_id'], self.df['word_count']))
        print(f"-> Successfully loaded dataset ({len(self.df)} chunks) and inverted index.")

    def _init_semantic_engine(self):
        """Initialize BGE-M3 model on GPU and build/load the FAISS vector database."""
        print("--- [2/3] Initializing Semantic Vector Engine ---")
        import torch
        from sentence_transformers import SentenceTransformer
        import faiss

        # 1. Explicitly check for CUDA availability and set device accordingly
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"-> CHÍNH THỨC SỬ DỤNG THIẾT BỊ: [{device.upper()}] ĐỂ XỬ LÝ")

        # 2. add BGE-M3 model loading with explicit device mapping
        self.model = SentenceTransformer('BAAI/bge-m3', device=device)

        if os.path.exists(self.faiss_index_path):
            print("-> Found precomputed FAISS vector index on disk. Loading into memory...")
            self.faiss_index = faiss.read_index(self.faiss_index_path)
        else:
            print("-> [WARNING] Precomputed FAISS index not found. Building a new vector database...")
            os.makedirs(self.faiss_index_dir, exist_ok=True)

            corpus_texts = self.df['text_content'].astype(str).tolist()

            # set batch_size=32 for efficient GPU processing, and enable progress bar for visibility
            print("-> Starting embedding generation...")
            embeddings = self.model.encode(
                corpus_texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            dimension = embeddings.shape[1]
            faiss.normalize_L2(embeddings)
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(embeddings)

            faiss.write_index(self.faiss_index, self.faiss_index_path)
            print(f"-> Successfully built and saved FAISS vector index to {self.faiss_index_path}")

    def bm25_search(self, query, top_k=10):
        """Execute BM25 lexical search aligned with single-token inverted index structure."""
        from underthesea import word_tokenize

        vietnamese_stopwords = {
            "và", "của", "các", "những", "cho", "được", "bởi", "tại", "trong",
            "trên", "dưới", "để", "với", "như", "khi", "sau", "trước", "từ",
            "đến", "này", "đó", "kia", "thì", "mà", "hoặc", "là", "có", "đang",
            "đã", "sẽ", "bị", "về", "theo", "ra", "vào", "lại", "nên", "một",
            "nhiều", "ít", "rằng"
        }

        raw_tokens = word_tokenize(query.lower().strip())
        query_tokens = []
        for token in raw_tokens:
            if token in vietnamese_stopwords:
                continue
            sub_tokens = token.replace("_", " ").split()
            for sub in sub_tokens:
                if sub.isalnum() and sub not in vietnamese_stopwords:
                    query_tokens.append(sub)

        scores = {}
        k1 = 1.5
        b = 0.75
        N = len(self.df)
        avgdl = self.df['word_count'].mean()

        for token in query_tokens:
            if token in self.inverted_index:
                doc_map = self.inverted_index[token]
                df_t = len(doc_map)
                idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1.0)

                for chunk_id, positions in doc_map.items():
                    tf = len(positions)
                    dl = self.chunk_lengths.get(chunk_id, avgdl)
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (dl / avgdl))

                    if chunk_id not in scores:
                        scores[chunk_id] = 0.0
                    scores[chunk_id] += idf * (numerator / denominator)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def semantic_search(self, query, top_k=10):
        """Execute dense semantic search using BGE-M3 embeddings and FAISS index lookup."""
        import faiss

        # Convert query text into coordinates vector
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)  # Match with normalized corpus vectors

        # Search the FAISS spatial database
        scores, indices = self.faiss_index.search(query_embedding, top_k)

        semantic_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS padding token for missing entries
                continue
            # Map index row back to its respective original chunk_id
            chunk_id = self.df.iloc[idx]['chunk_id']
            semantic_results.append((chunk_id, float(score)))

        return semantic_results

    def rrf_merge(self, bm25_results, semantic_results, k_rrf=60, top_n=5):
        """Combine ranks from Lexical and Semantic searches using Reciprocal Rank Fusion."""
        rrf_scores = {}

        # Process BM25 rankings
        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            rrf_scores[chunk_id] += 1.0 / (k_rrf + rank)

        # Process Semantic rankings
        for rank, (chunk_id, _) in enumerate(semantic_results, start=1):
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            rrf_scores[chunk_id] += 1.0 / (k_rrf + rank)

        # Sort chunks based on combined mathematical rank score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_chunks[:top_n]


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file_path = os.path.join(BASE_DIR, "data", "processed", "vietnam_tourism_rag_dataset.csv")
    index_file_path = os.path.join(BASE_DIR, "data", "processed", "inverted_index.json")
    faiss_dir = os.path.join(BASE_DIR, "data", "vectors")

    try:
        # Initialize the retrieval Search Engine
        retriever = HybridRetriever(csv_path=csv_file_path, index_path=index_file_path, faiss_index_dir=faiss_dir)

        sample_query = "Kinh nghiệm đi Vịnh Hạ Long"
        print(f"\n--- [3/3] RUNNING ENGINE TEST FOR QUERY: '{sample_query}' ---")

        # 1. Lexical Retrieval
        bm25_res = retriever.bm25_search(sample_query, top_k=10)

        # 2. Semantic Retrieval
        semantic_res = retriever.semantic_search(sample_query, top_k=10)

        # 3. Hybrid Reciprocal Rank Fusion
        final_hybrid_chunks = retriever.rrf_merge(bm25_res, semantic_res, top_n=5)

        print("\n==================================================")
        print("          HYBRID RETRIEVAL RESULTS (RRF)          ")
        print("==================================================")
        for rank, (chunk_id, rrf_score) in enumerate(final_hybrid_chunks, start=1):
            # Fetch text snippet from dataframe to verify quality
            text_snippet = retriever.df[retriever.df['chunk_id'] == chunk_id].iloc[0]['text_content']
            print(f"Rank {rank} | Chunk ID: {chunk_id} | RRF Score: {rrf_score:.5f}")
            print(f"Snippet: {text_snippet[:120]}...\n")
        print("==================================================")

    except Exception as e:
        print(f"An error occurred during retrieval testing: {e}")