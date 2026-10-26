#!/usr/bin/env python3
"""Non-training corpus-link and lexical candidate preflight for Tourism RAG.

Safety constraints:
- Does not import the project's retrieval module (which can automatically build FAISS).
- Does not load GPU models or modify training files/checkpoints.
- BM25 mining produces UNLABELED candidates only; never silently calls them negatives.
- No semantic or listwise quality claim is made without human/independent labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import tempfile
import unicodedata

STOPWORDS = {
    "và", "của", "các", "những", "cho", "được", "bởi", "tại", "trong",
    "trên", "dưới", "để", "với", "như", "khi", "sau", "trước", "từ",
    "đến", "này", "đó", "kia", "thì", "mà", "hoặc", "là", "có", "đang",
    "đã", "sẽ", "bị", "về", "theo", "ra", "vào", "lại", "nên", "một",
    "nhiều", "ít", "rằng"
}
MAP_FIELDS = ["train_index", "doc_id", "query", "margin", "positive_length",
              "negative_length", "positive_equals_negative", "negative_empty",
              "positive_mapping", "matched_chunk_id", "matched_doc_id",
              "matched_count", "source_doc_id_consistent", "eligible_strict",
              "candidate_ids_preview", "source_doc_in_corpus",
              "best_same_doc_chunk", "best_same_doc_token_jaccard"]
REVIEW_FIELDS = ["train_index", "query", "gold_chunk_id_provisional", "gold_doc_id",
                 "positive", "candidate_chunk_id", "candidate_doc_id", "candidate_text",
                 "retriever", "retriever_rank", "bm25_score", "label_status",
                 "risk_same_source_doc", "risk_matches_positive", "annotator_relevance_label",
                 "annotator_notes"]


def norm(text: object) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text or "")).split())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"Expected JSON list of dictionaries: {path}")
    return data


def read_corpus(path: Path) -> tuple[list[dict], dict[str, dict], dict[str, list[str]], dict[str, list[str]], list[str]]:
    rows = []
    by_id = {}
    by_doc = defaultdict(list)
    by_text = defaultdict(list)
    warnings = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        mandatory = {"chunk_id", "doc_id", "text_content"}
        if not mandatory.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Corpus missing required columns {sorted(mandatory)}; found {reader.fieldnames}")
        for line, item in enumerate(reader, start=2):
            chunk = norm(item.get("chunk_id"))
            doc = norm(item.get("doc_id"))
            passage = norm(item.get("text_content"))
            if not chunk:
                warnings.append(f"blank chunk_id at CSV line {line}")
                continue
            if chunk in by_id:
                warnings.append(f"duplicate chunk_id {chunk} at CSV line {line}")
                # Do not let ambiguous corpus IDs enter mapping/mining.
                by_id[chunk]["duplicate_id"] = True
                continue
            if not passage:
                warnings.append(f"empty corpus passage {chunk} at CSV line {line}")
            try:
                wc = float(item.get("word_count", ""))
                if not math.isfinite(wc) or wc <= 0:
                    raise ValueError()
            except (ValueError, TypeError):
                wc = max(1, len(passage.split()))
            entry = {"chunk_id": chunk, "doc_id": doc, "text": passage,
                     "word_count": wc, "duplicate_id": False}
            rows.append(entry)
            by_id[chunk] = entry
            by_doc[doc].append(chunk)
            if passage:
                by_text[passage].append(chunk)
    if not rows:
        raise ValueError("Corpus is empty")
    return rows, by_id, by_doc, by_text, warnings


def locate_positive(row: dict, by_id: dict, by_doc: dict, by_text: dict) -> dict:
    pos = norm(row.get("positive", row.get("pos", "")))
    doc = norm(row.get("doc_id", ""))
    if not pos:
        return {"type": "empty_positive", "ids": [], "gold": "", "consistent": False}
    exact = [cid for cid in by_text.get(pos, []) if not by_id[cid]["duplicate_id"]]
    if len(exact) == 1:
        cid = exact[0]
        consistency = doc == by_id[cid]["doc_id"]
        return {"type": "exact_unique", "ids": exact, "gold": cid, "consistent": consistency}
    if len(exact) > 1:
        # Even a unique ID within same doc is still text-duplicated in the corpus.
        return {"type": "exact_ambiguous", "ids": exact, "gold": "", "consistent": False}
    # Weak diagnostic ONLY: a sufficiently long training positive is a substring of
    # one chunk in its own source doc. Do not auto-approve this for training.
    if doc and not doc.startswith("EXT_PSEUDO_DOC_") and len(pos) >= 80:
        partial = [cid for cid in by_doc.get(doc, [])
                   if not by_id[cid]["duplicate_id"] and pos in by_id[cid]["text"]]
        if len(partial) == 1:
            return {"type": "substring_same_doc_provisional", "ids": partial,
                    "gold": partial[0], "consistent": True}
        if len(partial) > 1:
            return {"type": "substring_ambiguous", "ids": partial,
                    "gold": "", "consistent": False}
    return {"type": "unmapped", "ids": [], "gold": "", "consistent": False}


def closest_same_doc(pos: str, doc_id: str, by_id: dict, by_doc: dict) -> tuple[str, float]:
    """Diagnostic only. Never promotes a lexical-similarity match to a gold label."""
    p_tokens = set(re.findall(r"\w+", pos.casefold(), flags=re.UNICODE))
    if not p_tokens:
        return "", 0.0
    best_id, best = "", 0.0
    for cid in by_doc.get(doc_id, []):
        entry = by_id[cid]
        if entry["duplicate_id"] or not entry["text"]:
            continue
        c_tokens = set(re.findall(r"\w+", entry["text"].casefold(), flags=re.UNICODE))
        union = len(p_tokens | c_tokens)
        score = len(p_tokens & c_tokens) / union if union else 0.0
        if score > best:
            best_id, best = cid, score
    return best_id, best


def save_csv(path: Path, records: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def tokenise_query(query: str, word_tokenize) -> list[str]:
    raw = word_tokenize(query.lower().strip())
    tokens = []
    for token in raw:
        if token in STOPWORDS:
            continue
        for sub in token.replace("_", " ").split():
            if sub.isalnum() and sub not in STOPWORDS:
                tokens.append(sub)
    return tokens


def bm25_topk(query: str, inverted: dict, by_id: dict, n_docs: int,
              avg_dl: float, topk: int, word_tokenize) -> list[tuple[str, float]]:
    scores = defaultdict(float)
    # Preserve duplicated query tokens to mirror src/retrieval.py.
    for token in tokenise_query(query, word_tokenize):
        postings = inverted.get(token, {})
        if not isinstance(postings, dict):
            continue
        df_t = len(postings)
        idf = math.log((n_docs - df_t + 0.5) / (df_t + 0.5) + 1.0) if df_t <= n_docs else 0.0
        for cid, positions in postings.items():
            entry = by_id.get(str(cid))
            if not entry or entry["duplicate_id"]:
                continue
            try:
                tf = len(positions)
            except TypeError:
                continue
            dl = entry["word_count"]
            numerator = tf * 2.5  # k1 = 1.5 in current src/retrieval.py
            denominator = tf + 1.5 * (1 - 0.75 + 0.75 * (dl / avg_dl))
            if denominator:
                scores[str(cid)] += idf * (numerator / denominator)
    return sorted(scores.items(), key=lambda x: -x[1])[:topk]


def evaluate(args) -> dict:
    root = args.project_root.resolve()
    train_path = (args.train or root / "data/processed/train_moc_13.json").resolve()
    val_path = (args.val or root / "data/processed/cross_encoder_VAL.json").resolve()
    corpus_path = (args.corpus or root / "data/processed/vietnam_tourism_rag_dataset.csv").resolve()
    index_path = (args.inverted_index or root / "data/processed/inverted_index.json").resolve()
    out = (args.output or root / "results/candidate_set_preflight").resolve()
    for path in (train_path, val_path, corpus_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Override with --train/--val/--corpus if needed.")
    if out == root or out == root / "data":
        raise ValueError("Output must be a dedicated directory, not project root/data")
    out.mkdir(parents=True, exist_ok=True)
    print("[1/4] Reading train, validation and corpus (no models are loaded)...", flush=True)
    train = read_json_rows(train_path)
    val = read_json_rows(val_path)
    corpus, by_id, by_doc, by_text, corpus_warnings = read_corpus(corpus_path)
    print(f"  train={len(train)} | val={len(val)} | corpus={len(corpus)}", flush=True)

    print("[2/4] Mapping positive passages to corpus chunk IDs...", flush=True)
    mapping_records = []
    mapping_counts = Counter()
    strict = []
    for idx, row in enumerate(train):
        pos = norm(row.get("positive", row.get("pos", "")))
        neg = norm(row.get("negative", row.get("neg", "")))
        info = locate_positive(row, by_id, by_doc, by_text)
        invalid = not neg or pos == neg or not pos
        best_near_id, best_jaccard = closest_same_doc(pos, norm(row.get("doc_id")), by_id, by_doc)
        eligible = bool(info["type"] == "exact_unique" and info["consistent"] and not invalid)
        if eligible:
            strict.append(idx)
        mapping_counts[info["type"]] += 1
        mapping_records.append({
            "train_index": idx, "doc_id": norm(row.get("doc_id")),
            "query": str(row.get("query", "")), "margin": row.get("margin", ""),
            "positive_length": len(pos), "negative_length": len(neg),
            "positive_equals_negative": int(bool(pos and pos == neg)),
            "negative_empty": int(not neg), "positive_mapping": info["type"],
            "matched_chunk_id": info["gold"],
            "matched_doc_id": by_id[info["gold"]]["doc_id"] if info["gold"] else "",
            "matched_count": len(info["ids"]),
            "source_doc_id_consistent": int(info["consistent"]),
            "eligible_strict": int(eligible),
            "candidate_ids_preview": ";".join(info["ids"][:5]),
            "source_doc_in_corpus": int(norm(row.get("doc_id")) in by_doc),
            "best_same_doc_chunk": best_near_id,
            "best_same_doc_token_jaccard": round(best_jaccard, 5),
        })
    save_csv(out / "train_positive_corpus_mapping.csv", mapping_records, MAP_FIELDS)
    val_mapping_counts = Counter(locate_positive(row, by_id, by_doc, by_text)["type"] for row in val)
    train_doc_ids = {norm(row.get("doc_id")) for row in train if norm(row.get("doc_id"))}
    val_doc_ids = {norm(row.get("doc_id")) for row in val if norm(row.get("doc_id"))}
    print("  train positive mapping:", dict(mapping_counts), flush=True)
    print(f"  strict candidates eligible for BM25 mining: {len(strict)}", flush=True)

    summary = {
        "tool_version": "1.0.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": False, "neural_inference_performed": False,
        "split": {"train_rows": len(train), "val_rows": len(val),
                  "train_val_doc_id_overlap_count": len(train_doc_ids & val_doc_ids)},
        "corpus": {"rows": len(corpus), "unique_chunk_ids": len(by_id),
                   "corpus_warnings_count": len(corpus_warnings),
                   "corpus_warning_examples": corpus_warnings[:30]},
        "train": {"mapping_distribution": dict(mapping_counts),
                  "exact_unique_doc_consistent_clean": len(strict),
                  "positive_equals_negative": sum(r["positive_equals_negative"] for r in mapping_records),
                  "negative_empty": sum(r["negative_empty"] for r in mapping_records),
                  "source_doc_in_corpus": sum(r["source_doc_in_corpus"] for r in mapping_records),
                  "non_exact_with_same_doc_possible_review": sum(
                      r["source_doc_in_corpus"] and not r["eligible_strict"]
                      for r in mapping_records)},
        "validation": {"mapping_distribution": dict(val_mapping_counts)},
        "files_sha256": {"train": file_sha256(train_path),
                         "val": file_sha256(val_path), "corpus": file_sha256(corpus_path)},
        "mined_bm25": {"run": False},
        "interpretation": [
            "Exact corpus-text matching demonstrates identity, NOT query relevance validation.",
            "Substring matching and best same-doc token Jaccard are provisional; neither is a strict training label.",
            "BM25 candidates are unlabeled; automatically treating them as negatives is unsafe.",
            "The existing training data provide at most one labeled negative per row; 3-negative readiness cannot be claimed.",
            "This script does not test semantic/BGE-M3 retrieval or RRF fusion.",
            "Benchmark draft annotations are not publication-ready."
        ]
    }

    if args.mine_bm25:
        if not index_path.is_file():
            raise FileNotFoundError(f"BM25 index not found: {index_path}. Omit --mine-bm25 to run mapping-only.")
        try:
            from underthesea import word_tokenize
        except ImportError as exc:
            raise RuntimeError("BM25 mining requires underthesea in env_rag: pip install underthesea") from exc
        print("[3/4] Loading existing lexical inverted index (CPU only)...", flush=True)
        inverted = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(inverted, dict):
            raise ValueError("Inverted index must be JSON object token -> chunk postings")
        avgdl = sum(row["word_count"] for row in corpus) / len(corpus)
        rng = random.Random(args.seed)
        chosen = sorted(rng.sample(strict, min(args.sample, len(strict)))) if strict else []
        review = []
        mining_stats = Counter()
        gold_ranks = []
        for n, idx in enumerate(chosen, start=1):
            orig = train[idx]
            m = mapping_records[idx]
            gold = m["matched_chunk_id"]
            ranked = bm25_topk(str(orig.get("query", "")), inverted, by_id, len(corpus),
                                avgdl, args.topk, word_tokenize)
            rank = next((r for r, (cid, _) in enumerate(ranked, start=1) if cid == gold), None)
            gold_ranks.append(rank)
            mining_stats["queries_processed"] += 1
            if rank is not None:
                mining_stats["gold_found_in_bm25_topk"] += 1
            usable = 0
            for position, (cid, score) in enumerate(ranked, start=1):
                if cid == gold:
                    continue
                c = by_id[cid]
                equal_pos = norm(orig.get("positive", "")) == c["text"]
                if equal_pos:
                    mining_stats["excluded_identical_positive"] += 1
                    continue
                same_doc = (c["doc_id"] == m["matched_doc_id"])
                review.append({
                    "train_index": idx, "query": orig["query"],
                    "gold_chunk_id_provisional": gold,
                    "gold_doc_id": m["matched_doc_id"],
                    "positive": orig["positive"],
                    "candidate_chunk_id": cid, "candidate_doc_id": c["doc_id"],
                    "candidate_text": c["text"], "retriever": "BM25_ONLY",
                    "retriever_rank": position, "bm25_score": f"{score:.8f}",
                    "label_status": "UNLABELED_REQUIRES_REVIEW",
                    "risk_same_source_doc": int(same_doc),
                    "risk_matches_positive": 0,
                    "annotator_relevance_label": "", "annotator_notes": "",
                })
                usable += 1
                if same_doc:
                    mining_stats["same_source_doc_candidates"] += 1
            if usable >= args.min_negatives:
                mining_stats["queries_with_at_least_n_unlabeled_candidates"] += 1
            if n % 25 == 0 or n == len(chosen):
                print(f"  {n}/{len(chosen)} query candidates mined", flush=True)
        save_csv(out / "bm25_candidate_review_UNLABELED.csv", review, REVIEW_FIELDS)
        summary["mined_bm25"] = {
            "run": True, "query_sample": len(chosen), "seed": args.seed,
            "topk": args.topk, "min_negatives": args.min_negatives,
            "retriever": "BM25_ONLY (not RRF; BGE-M3/FAISS not evaluated)",
            **dict(mining_stats),
            "gold_in_topk_rate_on_strict_mapped_sample": (
                mining_stats["gold_found_in_bm25_topk"] / len(chosen) if chosen else None
            ),
            "queries_with_at_least_n_HUMAN_VERIFIED_negatives": 0,
            "note": "Zero verified because no review labels were provided, not because negatives do not exist."
        }
    else:
        print("[3/4] BM25 mining skipped (use --mine-bm25 to run a small CPU sample).", flush=True)
    (out / "candidate_preflight_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[4/4] Saved mapping CSV and summary:", out, flush=True)
    print("[DONE] No model training, neural inference or corpus file mutation.", flush=True)
    return summary


def self_test() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tiny_corpus.csv"
        with p.open("w", encoding="utf8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["doc_id", "chunk_id", "text_content"])
            w.writeheader()
            w.writerow({"doc_id": "A", "chunk_id": "A_chunk_1", "text_content": "Một câu chuyện du lịch Việt Nam"})
            w.writerow({"doc_id": "B", "chunk_id": "B_chunk_1", "text_content": "Câu chuyện du lịch vùng núi"})
            w.writerow({"doc_id": "B", "chunk_id": "B_chunk_2", "text_content": "Câu chuyện du lịch vùng núi"})
        corpus, by_id, by_doc, by_text, warnings = read_corpus(p)
        assert len(corpus) == 3 and not warnings
        good = locate_positive({"positive": "Một câu chuyện du lịch Việt Nam", "doc_id": "A"}, by_id, by_doc, by_text)
        assert good["type"] == "exact_unique" and good["consistent"] and good["gold"] == "A_chunk_1"
        ambig = locate_positive({"positive": "Câu chuyện du lịch vùng núi", "doc_id": "B"}, by_id, by_doc, by_text)
        assert ambig["type"] == "exact_ambiguous" and not ambig["gold"]
        mismatch = locate_positive({"positive": "Một câu chuyện du lịch Việt Nam", "doc_id": "X"}, by_id, by_doc, by_text)
        assert mismatch["type"] == "exact_unique" and not mismatch["consistent"]
        empty = locate_positive({"positive": "  ", "doc_id": "A"}, by_id, by_doc, by_text)
        assert empty["type"] == "empty_positive"
        missing = locate_positive({"positive": "Đây không phải đoạn trong corpus", "doc_id": "A"}, by_id, by_doc, by_text)
        assert missing["type"] == "unmapped"
        assert norm("a\u00a0 b\n c") == "a b c"
        nearest_id, jac = closest_same_doc("Câu chuyện du lịch vùng", "B", by_id, by_doc)
        assert nearest_id in {"B_chunk_1", "B_chunk_2"} and 0 < jac < 1
        assert tokenise_query("du lịch Việt Nam", lambda q: q.split()) == ["du", "lịch", "việt", "nam"]
        fake_idx = {"việt": {"A_chunk_1": [1]}, "nam": {"A_chunk_1": [2]}}
        results = bm25_topk("Việt Nam", fake_idx, by_id, len(corpus), 6, 2, lambda q: q.split())
        assert results[0][0] == "A_chunk_1"
        print("[PASS] 9 preflight checks: exact/ambiguous/mismatch/empty/unmapped/normalization/near-doc/tokenization/BM25")


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="Tourism RAG safe candidate-set feasibility preflight")
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent,
                    help="Repository root (default: folder holding this script)")
    ap.add_argument("--train", type=Path, help="Override TRAIN JSON path")
    ap.add_argument("--val", type=Path, help="Override validation JSON path")
    ap.add_argument("--corpus", type=Path, help="Override corpus CSV path")
    ap.add_argument("--inverted-index", type=Path, help="Override existing BM25 inverted index JSON")
    ap.add_argument("--output", type=Path, help="Write diagnostics here (default results/candidate_set_preflight)")
    ap.add_argument("--mine-bm25", action="store_true", help="Optional CPU mining on strictly mapped queries")
    ap.add_argument("--sample", type=int, default=100, help="BM25 sampled queries (default 100)")
    ap.add_argument("--topk", type=int, default=10, help="BM25 top K (default 10)")
    ap.add_argument("--min-negatives", type=int, default=3, help="Minimum candidate count per query (default 3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self-test", action="store_true", help="Test with tiny synthetic corpus; no project needed")
    args = ap.parse_args(argv)
    if args.sample < 1 or args.topk < 1 or args.min_negatives < 1:
        ap.error("--sample, --topk and --min-negatives must be positive")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    evaluate(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
