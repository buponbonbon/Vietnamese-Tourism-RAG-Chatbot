from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Place this file at:
#   results/downstream_benchmark/build_downstream_benchmark_template.py
# Run from the Vietnamese-Tourism-RAG-Chatbot repository root:
#   python -m results.downstream_benchmark.build_downstream_benchmark_template
#
# Produces NEW V2 candidate templates (JSON and CSV) using the CORPUS, not
# the previously annotated benchmark_annotation_draft_200.csv.
# These outputs are intentionally named *_v2 to avoid overwriting the
# existing benchmark_annotation_template.csv or the annotated DRAFT.
# Do NOT use --overwrite unless you intend to replace the V2 templates.

ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "data" / "processed" / "vietnam_tourism_rag_dataset.csv"
OUT_DIR = ROOT / "results" / "downstream_benchmark"

DEFAULT_N = 200
DEFAULT_SEED = 42
DEFAULT_ANSWER_RATIO = 0.60

MIN_CHARS = 420
MAX_CHARS = 1700
MIN_WORDS = 75

TOURISM_TERMS = [
    "du lịch", "du khách", "khách du lịch", "điểm đến", "tham quan",
    "lễ hội", "văn hóa", "di tích", "danh thắng", "tour", "lữ hành",
    "ẩm thực", "khu du lịch", "vườn quốc gia", "sinh thái", "nghỉ dưỡng",
    "resort", "khách sạn", "homestay", "biển", "đảo", "bãi biển",
    "bảo tàng", "chùa", "đền", "tháp", "hang động", "cảnh quan",
    "trải nghiệm", "di sản", "làng nghề", "phố cổ", "cửa khẩu",
]

BAD_PATTERNS = [
    r"Trung tâm Thông tin du lịch\s*Địa chỉ",
    r"Địa chỉ:\s*80 Quán Sứ",
    r"Giấy phép số\s*:",
    r"Chịu trách nhiệm nội dung",
    r"contact@vietnamtourism\.gov\.vn",
    r"\bTTHC\b",
    r"thủ tục hành chính",
    r"giấy Tiếp nhận hồ sơ",
    r"nộp hồ sơ trực tiếp",
    r"nhận kết quả TTHC",
    r"thông tin tài khoản của Cục Du lịch",
    r"tên chủ tài khoản",
    r"mã đơn vị quan hệ ngân sách",
    r"Kho bạc Nhà nước",
    r"chuyển khoản từ ngân hàng",
]

BAD_RE = re.compile("|".join(f"(?:{p})" for p in BAD_PATTERNS), re.I)

NUMBER_OR_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}|\d+(?:[.,]\d+)?\s*(?:%|km|ha|MW|VNĐ|đồng|triệu|tỷ|lượt|ngày|tháng|năm))\b",
    re.I,
)

# A practical Vietnamese proper-name/location anchor heuristic.
PROPER_NOUN_RE = re.compile(
    r"\b[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-Ỹ]"
    r"[A-Za-zĐđÀ-ỹ]+(?:\s+[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-Ỹ]"
    r"[A-Za-zĐđÀ-ỹ]+){1,5}\b"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--answer_ratio", type=float, default=DEFAULT_ANSWER_RATIO)
    p.add_argument("--min_chars", type=int, default=MIN_CHARS)
    p.add_argument("--max_chars", type=int, default=MAX_CHARS)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def canonical_for_dedup(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def infer_doc_id(chunk_id: str) -> str:
    return re.sub(r"_chunk_\d+$", "", str(chunk_id))


def infer_source_group(doc_id: str) -> str:
    m = re.match(r"^([A-Za-z]+)_", str(doc_id))
    return m.group(1) if m else "OTHER"


def tourism_relevance(text: str) -> int:
    low = text.lower()
    return sum(term in low for term in TOURISM_TERMS)


def sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?](?:\s|$)", text))


def transformable_anchor_count(text: str) -> int:
    return len(NUMBER_OR_DATE_RE.findall(text)) + len(PROPER_NOUN_RE.findall(text))


def usable(text: str, min_chars: int, max_chars: int) -> bool:
    text = normalize_text(text)

    if not (min_chars <= len(text) <= max_chars):
        return False
    if len(text.split()) < MIN_WORDS:
        return False
    if BAD_RE.search(text):
        return False
    if tourism_relevance(text) < 1:
        return False
    if sentence_count(text) < 2:
        return False

    # Avoid chunks dominated by metadata / markup / URLs.
    if text.lower().count("http://") + text.lower().count("https://") >= 2:
        return False
    if text.count("@") >= 2:
        return False

    readable = sum(ch.isalnum() or ch.isspace() or ch in ".,;:!?%/-–—()“”'\"" for ch in text)
    if readable / max(len(text), 1) < 0.90:
        return False

    return True


def quality_score(text: str) -> float:
    score = 0.0
    score += min(tourism_relevance(text), 5) * 1.5
    score += min(sentence_count(text), 8) * 0.35
    score += min(transformable_anchor_count(text), 10) * 0.15

    # Prefer chunks that begin somewhat like a sentence rather than an obvious fragment.
    first = text[:1]
    if first and (first.isupper() or first.isdigit() or first in "“\"'"):
        score += 1.0

    # Penalize obvious caption-heavy content.
    score -= 0.4 * text.count("Ảnh:")
    return score


def collect_candidates(df: pd.DataFrame, min_chars: int, max_chars: int) -> List[Dict]:
    if "chunk_id" not in df.columns or "text_content" not in df.columns:
        raise ValueError(
            "Corpus CSV must contain 'chunk_id' and 'text_content'. "
            f"Found: {list(df.columns)}"
        )

    rows = []
    seen_text = set()

    for _, row in df.iterrows():
        chunk_id = str(row["chunk_id"])
        text = normalize_text(row["text_content"])

        if not usable(text, min_chars, max_chars):
            continue

        canon = canonical_for_dedup(text)
        if canon in seen_text:
            continue
        seen_text.add(canon)

        doc_id = infer_doc_id(chunk_id)
        rows.append({
            "gold_chunk_id": chunk_id,
            "gold_doc_id": doc_id,
            "source_group": infer_source_group(doc_id),
            "gold_evidence": text,
            "_quality": quality_score(text),
            "_transformable": transformable_anchor_count(text) > 0,
        })

    return rows


def choose_unique_docs(
    candidates: List[Dict],
    n_answer: int,
    n_correct: int,
    seed: int,
) -> List[Tuple[str, Dict]]:
    rng = random.Random(seed)

    # Random jitter prevents deterministic over-selection from a few source styles
    # while still favoring better chunks.
    for row in candidates:
        row["_rank_score"] = row["_quality"] + rng.uniform(0, 1.5)

    transformable = sorted(
        [r for r in candidates if r["_transformable"]],
        key=lambda r: r["_rank_score"],
        reverse=True,
    )
    general = sorted(
        candidates,
        key=lambda r: r["_rank_score"],
        reverse=True,
    )

    used_docs = set()
    selected: List[Tuple[str, Dict]] = []

    # Select correction/trap rows first; do not select any if n_correct == 0.
    if n_correct > 0:
        for row in transformable:
            if row["gold_doc_id"] in used_docs:
                continue
            selected.append(("correct", row))
            used_docs.add(row["gold_doc_id"])
            if sum(b == "correct" for b, _ in selected) >= n_correct:
                break

    if sum(b == "correct" for b, _ in selected) < n_correct:
        raise RuntimeError("Not enough transformable unique-document chunks for 'correct' samples.")

    if n_answer > 0:
        for row in general:
            if row["gold_doc_id"] in used_docs:
                continue
            selected.append(("answer", row))
            used_docs.add(row["gold_doc_id"])
            if sum(b == "answer" for b, _ in selected) >= n_answer:
                break

    if sum(b == "answer" for b, _ in selected) < n_answer:
        raise RuntimeError("Not enough unique-document chunks for 'answer' samples.")

    rng.shuffle(selected)
    return selected


def build_rows(selected: List[Tuple[str, Dict]]) -> List[Dict]:
    rows = []
    for i, (behavior, source) in enumerate(selected, start=1):
        rows.append({
            "benchmark_id": f"DSV2_{i:04d}",
            "target_behavior": behavior,
            "query": "",
            "reference_answer": "",
            "trap_type": "",
            "trap_original_value": "",
            "trap_replacement_value": "",
            "annotation_status": "TODO",
            "gold_chunk_id": source["gold_chunk_id"],
            "gold_doc_id": source["gold_doc_id"],
            "source_group": source["source_group"],
            "gold_evidence": source["gold_evidence"],
            "annotator_notes": "",
        })
    return rows


def main():
    args = parse_args()

    if args.n < 1:
        raise ValueError("--n must be a positive integer.")
    if not 0 <= args.answer_ratio <= 1:
        raise ValueError("--answer_ratio must be in [0, 1].")
    if args.min_chars < 1 or args.max_chars < args.min_chars:
        raise ValueError("Require 1 <= --min_chars <= --max_chars.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "benchmark_annotation_template_v2.json"
    csv_path = OUT_DIR / "benchmark_annotation_template_v2.csv"

    if not args.overwrite and (json_path.exists() or csv_path.exists()):
        raise FileExistsError(
            f"V2 output already exists in {OUT_DIR}. Use --overwrite to replace it."
        )

    print(f"Loading corpus: {CORPUS_PATH}")

    # Keep literal IDs and empty text fields intact; no pandas-version-specific
    # global options are needed.
    df = pd.read_csv(CORPUS_PATH, low_memory=False, keep_default_na=False,
                     dtype={"chunk_id": str})

    print(f"Corpus chunks: {len(df)}")
    candidates = collect_candidates(df, args.min_chars, args.max_chars)
    print(f"High-quality tourism candidates after filtering: {len(candidates)}")

    n_answer = round(args.n * args.answer_ratio)
    n_correct = args.n - n_answer

    selected = choose_unique_docs(
        candidates=candidates,
        n_answer=n_answer,
        n_correct=n_correct,
        seed=args.seed,
    )
    rows = build_rows(selected)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    answer_count = sum(r["target_behavior"] == "answer" for r in rows)
    correct_count = sum(r["target_behavior"] == "correct" for r in rows)

    print("\n================ V2 BENCHMARK TEMPLATE ================")
    print(f"Rows: {len(rows)}")
    print(f"answer: {answer_count}")
    print(f"correct: {correct_count}")
    print(f"Unique documents: {len(set(r['gold_doc_id'] for r in rows))}")
    print(f"Unique source groups: {len(set(r['source_group'] for r in rows))}")
    assert len({r["benchmark_id"] for r in rows}) == len(rows)
    assert len({r["gold_doc_id"] for r in rows}) == len(rows)
    print("=======================================================")
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")

    print("\nNext step: manually create query/reference_answer.")
    print("For target_behavior='correct', alter EXACTLY ONE factual element and record:")
    print("  trap_type, trap_original_value, trap_replacement_value.")
    print("Use only gold_evidence; do not add outside knowledge.")


if __name__ == "__main__":
    main()
