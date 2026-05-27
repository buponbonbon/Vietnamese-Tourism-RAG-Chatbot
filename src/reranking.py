import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TourismReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        print("\n--- [4/5] Initializing Neural Reranker Engine ---")

        # 1. Hardware acceleration configuration (Prioritize CUDA GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"-> RERANKER EXECUTION DEVICE: [{str(self.device).upper()}]")  # Fixed type error here

        # 2. Instantiate Tokenizer and Cross-Encoder Model from Hugging Face Hub
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Deploy model weights to the target computational device
        self.model.to(self.device)
        self.model.eval()  # Switch to evaluation mode for inference efficiency
        print(f"-> Successfully loaded Reranker architecture: {model_name}")

    def rerank(self, query: str, hybrid_results: list) -> list:
        """
        Re-scores retrieved document chunks based on deep query-context cross-attention.
        Implements a Cross-Encoder architecture to capture fine-grained semantic interactions.
        """
        if not hybrid_results:
            return []

        # Construct raw text pairs for cross-encoding input: (Query, Passage Chunk)
        pairs = [[query, chunk['text']] for chunk in hybrid_results]

        # Tokenize inputs and map tensors onto the corresponding hardware accelerator
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)

            # Forward pass computation to extract semantic matching logits
            scores = self.model(**inputs).logits.view(-1).float().cpu().tolist()

        # Map computed neural relevance logits back to individual document metadata
        for i, score in enumerate(scores):
            hybrid_results[i]['rerank_score'] = score

        # Execute downstream sorting in descending order based on localized neural scores
        reranked_results = sorted(hybrid_results, key=lambda x: x['rerank_score'], reverse=True)
        return reranked_results


# =========================================================================
# SYSTEM VERIFICATION & UNIT TESTING (Independent Execution)
# =========================================================================
if __name__ == "__main__":
    # Mock data structure mimicking Stage 1 Hybrid Retrieval (RRF Output)
    mock_query = "Kinh nghiệm đi Vịnh Hạ Long"
    mock_hybrid_chunks = [
        {
            "chunk_id": "HP_008_chunk_2",
            "text": "hợp, các cơ quan có thẩm quyền sẽ cấp phép cho tàu du lịch vận chuyển khách theo 4 hành trình kết nối: Vịnh Hạ Long 3 - ...",
            "rrf_score": 0.01639
        },
        {
            "chunk_id": "QN_030_chunk_1",
            "text": "Đoạn chụp ảnh lưu niệm tại hang Sửng Sốt Trong 2 ngày đầu, đoàn đã chia thành 2 nhóm. Nhóm 1 gồm các đơn vị có nhu cầu t...",
            "rrf_score": 0.01639
        },
        {
            "chunk_id": "BK_054_chunk_3",
            "text": "nay trên Vịnh Hạ Long có 5 hành trình tham quan, mỗi hành trình có những đặc trưng riêng thu hút khách du lịch. Trung bì...",
            "rrf_score": 0.01613
        }
    ]

    # Initialize the neural fine-grained reranking engine
    reranker = TourismReranker()
    final_ordered_list = reranker.rerank(mock_query, mock_hybrid_chunks)

    print("\n==================================================")
    print("         FINAL RERANKED RESULTS (STAGE 2)         ")
    print("==================================================")
    for rank, chunk in enumerate(final_ordered_list, 1):
        print(f"Rank {rank} | ID: {chunk['chunk_id']} | Neural Score: {chunk['rerank_score']:.4f}")
        print(f"Passage: {chunk['text'][:120]}...\n")