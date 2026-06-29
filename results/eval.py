import os
import json
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


def calculate_mrr(pos_score, neg_scores):
    """
    Calculates Mean Reciprocal Rank (MRR) for a single query.
    - Ground truth (Positive document) is placed at index 0.
    - All other elements are hard negatives.
    """
    all_scores = [pos_score] + neg_scores
    # Sort indices in descending order based on prediction scores
    ranks = np.argsort(all_scores)[::-1]
    # Find the 1-based rank position of the positive document (index 0)
    pos_rank = np.where(ranks == 0)[0][0] + 1
    return 1.0 / pos_rank


def calculate_ndcg(pos_score, neg_scores, k=10):
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG@K).
    - Positive document has a true relevance score of 1.
    - Negative documents have a true relevance score of 0.
    """
    all_scores = [pos_score] + neg_scores
    true_relevance = [1] + [0] * len(neg_scores)

    # Get top-K ranked indices based on model predicted scores
    sorted_indices = np.argsort(all_scores)[::-1][:k]

    # Calculate Discounted Cumulative Gain (DCG@K)
    dcg = 0.0
    for i, idx in enumerate(sorted_indices):
        rel = true_relevance[idx]
        dcg += rel / np.log2(i + 2)

    # Ideal DCG (IDCG@K) where the positive document is perfectly ranked at position 1
    idcg = 1.0 / np.log2(0 + 2)

    return dcg / idcg


def calculate_precision_at_1(pos_score, neg_scores):
    """
    Calculates Precision@1 (Checks if the positive document is ranked highest).
    Returns 1.0 if the positive score strictly beats all negative scores, else 0.0.
    """
    max_neg = max(neg_scores) if neg_scores else -float('inf')
    return 1.0 if pos_score > max_neg else 0.0


def evaluate_pipeline():
    print("=" * 60)
    print("   LAUNCHING CROSS-ENCODER METRIC EVALUATION BENCHMARK  ")
    print("=" * 60)

    # 1. PATH CONFIGURATION (Relative paths from the 'results' directory)
    MODEL_PATH = "../models/vietnam_tourism_reranker_margin_2.0_epochs_5"
    TEST_DATA_PATH = "../data/processed/bge_train_data_local_RAW.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Loading checkpoint from: [{MODEL_PATH}]")
    print(f"-> Testing on dataset: [{TEST_DATA_PATH}]")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model path not found: {MODEL_PATH}. Please verify the folder name!")

    # 2. INITIALIZE MODEL AND TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    # 3. LOAD EVALUATION DATASET (TEST SET)
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"[DATA] Successfully loaded {len(test_data)} test samples.")

    mrr_list = []
    ndcg_list = []
    p1_list = []

    # 4. EVALUATION LOOP
    with torch.no_grad():
        for item in tqdm(test_data, desc="Evaluating"):
            query = item["query"]
            pos_docs = item["pos"]
            neg_docs = item["neg"]

            # Compute prediction score for Positive document(s)
            pos_scores = []
            for pos_doc in pos_docs:
                inputs = tokenizer(query, pos_doc, return_tensors="pt", truncation=True, max_length=512,
                                   padding=True).to(device)
                score = model(**inputs).logits.view(-1).item()
                pos_scores.append(score)
            avg_pos_score = np.mean(pos_scores)

            # Compute prediction scores for adversarial Hard-Negative documents
            neg_scores = []
            for neg_doc in neg_docs:
                inputs = tokenizer(query, neg_doc, return_tensors="pt", truncation=True, max_length=512,
                                   padding=True).to(device)
                score = model(**inputs).logits.view(-1).item()
                neg_scores.append(score)

            # Compute ranking metrics for the current sample
            mrr_list.append(calculate_mrr(avg_pos_score, neg_scores))
            ndcg_list.append(calculate_ndcg(avg_pos_score, neg_scores, k=10))
            p1_list.append(calculate_precision_at_1(avg_pos_score, neg_scores))

    # 5. PRINT FINAL METRICS REPORT
    print("         FINAL BENCHMARK METRIC RESULTS       ")
    print("=" * 45)
    print(f" Adv NDCG@10     : {np.mean(ndcg_list):.4f}")
    print(f" Adv MRR         : {np.mean(mrr_list):.4f}")
    print(f" Adv Precision@1 : {np.mean(p1_list):.4f}")



if __name__ == "__main__":
    evaluate_pipeline()