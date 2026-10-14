from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Put this file in:
# Vietnamese-Tourism-RAG-Chatbot/results/check_answerability.py

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent

TEST_PATH = ROOT / "data" / "processed" / "cross_encoder_TEST.json"
CORPUS_PATH = ROOT / "data" / "processed" / "vietnam_tourism_rag_dataset.csv"
FAISS_PATH = ROOT / "data" / "vectors" / "faiss_index.bin"
GENERATIONS_PATH = RESULTS / "faithfulness_outputs" / "faithfulness_generations.jsonl"
OUTPUT_PATH = RESULTS / "faithfulness_outputs" / "answerability_check.csv"

MODEL_NAME = "BAAI/bge-m3"
TOP_K = 5


def normalize_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_pilot_queries(path: Path):
    seen = set()
    queries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            q = row.get("query")
            if q and q not in seen:
                seen.add(q)
                queries.append(q)
    return queries


def load_test_map(path: Path):
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        for key in ("data", "items", "examples", "test"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break

    out = {}
    for item in raw:
        q = item.get("query")
        if q and q not in out:
            out[q] = item
    return out


def lexical_similarity(a: str, b: str) -> float:
    a_n = normalize_text(a)
    b_n = normalize_text(b)

    # Sequence ratio catches near-copy / chunked versions.
    return SequenceMatcher(None, a_n, b_n, autojunk=False).ratio()


def main():
    print("Loading pilot queries...")
    pilot_queries = load_pilot_queries(GENERATIONS_PATH)
    print(f"Pilot unique queries: {len(pilot_queries)}")

    print("Loading cross-encoder test set...")
    test_map = load_test_map(TEST_PATH)

    print("Loading corpus...")
    pd.options.future.infer_string = False
    pd.options.mode.string_storage = "python"
    df = pd.read_csv(CORPUS_PATH, low_memory=False)

    if "text_content" not in df.columns or "chunk_id" not in df.columns:
        raise ValueError("Corpus CSV must contain chunk_id and text_content columns.")

    print("Loading BGE-M3...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("Loading FAISS index...")
    index = faiss.read_index(str(FAISS_PATH))

    rows = []

    for q in tqdm(pilot_queries, desc="Checking answerability"):
        item = test_map.get(q)

        if item is None:
            rows.append({
                "query": q,
                "doc_id": "",
                "status": "QUERY_NOT_FOUND_IN_TEST",
            })
            continue

        positive = str(item.get("positive", item.get("pos", "")))
        doc_id = str(item.get("doc_id", ""))

        emb = model.encode([positive], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        scores, indices = index.search(emb, TOP_K)

        best = None
        candidates = []

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue

            corpus_row = df.iloc[int(idx)]
            chunk_id = str(corpus_row["chunk_id"])
            chunk_text = str(corpus_row["text_content"])
            seq = lexical_similarity(positive, chunk_text)

            candidate = {
                "rank": rank,
                "chunk_id": chunk_id,
                "dense_similarity": float(score),
                "sequence_similarity": float(seq),
                "chunk_text": chunk_text,
            }
            candidates.append(candidate)

            if best is None or seq > best["sequence_similarity"]:
                best = candidate

        if best is None:
            rows.append({
                "query": q,
                "doc_id": doc_id,
                "status": "NO_FAISS_RESULT",
            })
            continue

        # Conservative heuristic:
        # - near-copy: strong lexical evidence that the gold passage (or a chunk of it)
        #   actually exists in the RAG corpus.
        # - possible paraphrase: semantically close but not near-copy; requires manual check.
        # - weak/no evidence: not suitable to call corpus-answerable without inspection.
        if best["sequence_similarity"] >= 0.70:
            status = "LIKELY_GOLD_EVIDENCE_IN_CORPUS"
        elif best["dense_similarity"] >= 0.85:
            status = "POSSIBLE_PARAPHRASE_MANUAL_CHECK"
        else:
            status = "WEAK_OR_NO_EVIDENCE"

        rows.append({
            "query": q,
            "doc_id": doc_id,
            "status": status,
            "best_chunk_id": best["chunk_id"],
            "best_dense_similarity": best["dense_similarity"],
            "best_sequence_similarity": best["sequence_similarity"],
            "gold_positive": positive,
            "best_chunk_text": best["chunk_text"],
        })

    out = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n================ SUMMARY ================")
    print(out["status"].value_counts(dropna=False).to_string())
    print("=========================================")
    print(f"\nSaved: {OUTPUT_PATH}")
    print("\nInterpretation:")
    print("- LIKELY_GOLD_EVIDENCE_IN_CORPUS: gold passage is very similar to a corpus chunk.")
    print("- POSSIBLE_PARAPHRASE_MANUAL_CHECK: semantically close, but inspect manually.")
    print("- WEAK_OR_NO_EVIDENCE: likely not aligned with the RAG corpus.")


if __name__ == "__main__":
    main()
