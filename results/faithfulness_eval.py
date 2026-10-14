from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import traceback
import faulthandler
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

faulthandler.enable()

# ---------------------------------------------------------------------
# Put this file in:
#   Vietnamese-Tourism-RAG-Chatbot/results/faithfulness_eval_v2.py
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


TEST_PATH = ROOT / "data" / "processed" / "cross_encoder_TEST.json"
OUT_DIR = RESULTS / "faithfulness_outputs"

LLM_MODEL = "qwen2.5:14b-instruct-q4_K_M"
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

ENTAILMENT_THRESHOLD = 0.50
DEFAULT_MAX_QUERIES = 30

GENERATIONS_PATH = OUT_DIR / "faithfulness_generations.jsonl"
# v2 is intentionally retained for scientific provenance; never overwrite it.
SCORED_V2_PATH = OUT_DIR / "faithfulness_scored_v2.jsonl"
SCORED_V3_PATH = OUT_DIR / "faithfulness_scored_v3.jsonl"
SUMMARY_V3_PATH = OUT_DIR / "faithfulness_summary_v3.json"
GROUND_TRUTH_PATH = OUT_DIR / "ground_truth_adversarial_v3.json"
SCORING_SCHEMA = "faithfulness-pilot-v3-correction-audit"

# Generation-level behavior labels.
VALID_BEHAVIORS = {"answer", "refuse", "correct"}

# Refusal / insufficient-evidence patterns.
# These are deliberately broader than the v1 list so that a valid abstention
# is not accidentally counted as a hallucinated factual claim.
REFUSAL_PATTERNS = [
    r"không có thông tin",
    r"không đủ thông tin",
    r"không tìm thấy thông tin",
    r"không có dữ liệu",
    r"không tìm thấy dữ liệu",
    r"không có (?:chi tiết|thông tin|dữ liệu) cụ thể",
    r"không có (?:năm|mốc thời gian)(?:\s+cụ thể)?",
    r"không có đề cập cụ thể",
    r"không thể trả lời",
    r"không thể xác định",
    r"không thể xác minh",
    r"không đủ cơ sở",
    r"không có đủ cơ sở",
    r"không được cung cấp",
    r"không được đề cập",
    r"không đề cập",
    r"không nêu",
    r"unavailable in the database",
    r"not enough information",
    r"insufficient information",
    r"cannot determine",
    r"cannot verify",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Faithfulness + generation-level adversarial behavior evaluation."
    )
    p.add_argument(
        "--mode",
        choices=["generate", "init_gt", "score", "rescore_v2", "all"],
        default="score",
        help=(
            "generate: generate baseline/proposed answers; "
            "init_gt: create a manual ground-truth template; "
            "score: recompute NLI and behavior into v3; "
            "rescore_v2: reuse archived v2 NLI and fix behavior labels (no GPU); "
            "all: generate, then score if GT is already complete."
        ),
    )
    p.add_argument("--max_queries", type=int, default=DEFAULT_MAX_QUERIES,
                   help="0 = all unique queries")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--top_n", type=int, default=3)
    p.add_argument("--llm_model", default=LLM_MODEL)
    p.add_argument("--nli_model", default=NLI_MODEL)
    p.add_argument("--nli_device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threshold", type=float, default=ENTAILMENT_THRESHOLD)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--allow_legacy_generations", action="store_true", help="Explicitly acknowledge that v2 generations have no manifest.")
    return p.parse_args()


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_test_container(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("data", "items", "examples", "test"):
            if isinstance(raw.get(key), list):
                return raw[key]
    raise ValueError("Unsupported cross_encoder_TEST.json structure.")


def load_test_items(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return normalize_test_container(json.load(f))


def load_queries(path: Path, max_queries: int, seed: int) -> List[str]:
    items = load_test_items(path)
    seen = set()
    queries = []

    for item in items:
        q = item.get("query", item.get("question"))
        if isinstance(q, str):
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                queries.append(q)

    random.Random(seed).shuffle(queries)
    return queries if max_queries == 0 else queries[:max_queries]


def test_item_map() -> Dict[str, Dict[str, Any]]:
    out = {}
    for item in load_test_items(TEST_PATH):
        q = item.get("query", item.get("question"))
        if isinstance(q, str) and q.strip() and q.strip() not in out:
            out[q.strip()] = item
    return out


def clean_context(c: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "chunk_id": str(c.get("chunk_id", "")),
        "text": str(c.get("text", "")),
    }
    for k in ("rrf_score", "rerank_score"):
        if k in c:
            try:
                out[k] = float(c[k])
            except Exception:
                out[k] = None
    return out


def detect_refusal(text: str) -> bool:
    low = (text or "").lower()
    return any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in REFUSAL_PATTERNS)


def sentence_is_refusal(sentence: str) -> bool:
    return detect_refusal(sentence)


def split_factual_claims(answer: str) -> List[str]:
    """
    Sentence-level factual claim segmentation.

    Refusal / insufficient-evidence sentences are deliberately excluded.
    A pure refusal therefore has no factual claims and receives
    context_faithfulness=None rather than an artificial score of 0.
    """
    text = re.sub(r"\r\n?", "\n", (answer or "").strip())
    text = re.sub(r"\n\s*[-*•]\s*", "\n", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)

    claims = []
    for part in parts:
        part = re.sub(r"^\s*\d+[\).\:-]?\s*", "", part).strip(" \t-•")
        if len(part) < 8:
            continue
        if sentence_is_refusal(part):
            continue
        claims.append(part)

    return claims


# ---------------------------------------------------------------------
# Ground-truth template
# ---------------------------------------------------------------------
def create_ground_truth_template(overwrite: bool = False) -> None:
    """
    Creates generation-level GT for the exact queries already generated.

    expected_behavior:
      answer  = answer the question from evidence
      refuse  = refuse / abstain because evidence is insufficient or premise is unsupported
      correct = explicitly correct a false/misleading premise

    reference_answer:
      A short canonical answer or correction written by the annotator.
      Required for 'answer' and 'correct'; optional for 'refuse'.
    """
    if GROUND_TRUTH_PATH.exists() and not overwrite:
        print(f"[INFO] Ground-truth file already exists: {GROUND_TRUTH_PATH}")
        print("Use --overwrite only if you intentionally want to recreate the template.")
        return

    generations = read_jsonl(GENERATIONS_PATH)
    if generations:
        seen = set()
        queries = []
        for row in generations:
            q = row.get("query")
            if isinstance(q, str) and q not in seen:
                seen.add(q)
                queries.append(q)
    else:
        queries = load_queries(TEST_PATH, DEFAULT_MAX_QUERIES, 42)

    item_map = test_item_map()
    template = []

    for q in queries:
        item = item_map.get(q, {})
        template.append({
            "query": q,
            "expected_behavior": "",
            "reference_answer": "",
            "gold_evidence": str(item.get("positive", item.get("pos", ""))),
            "doc_id": str(item.get("doc_id", "")),
            "notes": "",
            "trap_original_value": "",
            "trap_replacement_value": "",
        })

    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GROUND_TRUTH_PATH.open("w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] Ground-truth template created:")
    print(f"  {GROUND_TRUTH_PATH}")
    print("\nFill ONLY these fields manually:")
    print('  "expected_behavior": "answer" | "refuse" | "correct"')
    print('  "reference_answer": short canonical answer/correction')
    print('  "notes": optional')
    print('  "trap_original_value" and "trap_replacement_value": required for correct')
    print("\nDo not change query text.")


def load_ground_truth(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth not found: {path}\n"
            "Run: python -m results.faithfulness_eval_v2 --mode init_gt"
        )

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("ground_truth_adversarial_v3.json must be a JSON list.")

    out: Dict[str, Dict[str, Any]] = {}
    unlabeled = []

    for item in raw:
        q = str(item.get("query", "")).strip()
        behavior = str(item.get("expected_behavior", "")).strip().lower()
        if not q:
            continue

        if q in out:
            raise ValueError(f"Duplicate ground-truth query: {q}")
        if behavior not in VALID_BEHAVIORS:
            unlabeled.append(q)
        if behavior in {"answer", "correct"} and not str(item.get("reference_answer", "")).strip():
            raise ValueError(f"Missing reference_answer: {q}")
        if behavior == "correct":
            correct = str(item.get("trap_original_value", "")).strip()
            false = str(item.get("trap_replacement_value", "")).strip()
            if not correct or not false or correct.casefold() == false.casefold():
                raise ValueError(f"Correction query needs distinct trap_original_value/trap_replacement_value: {q}")
            if not fact_value_present(str(item.get("gold_evidence", "")), correct):
                raise ValueError(f"Correct value absent from gold_evidence: {q}")
            if not fact_value_present(q, false):
                raise ValueError(f"False value absent from query: {q}")
        out[q] = item

    if unlabeled:
        preview = "\n".join(f"  - {q}" for q in unlabeled[:10])
        raise ValueError(
            f"{len(unlabeled)} ground-truth rows are still unlabeled.\n"
            f"Expected one of: {sorted(VALID_BEHAVIORS)}\n"
            f"Examples:\n{preview}"
        )

    return out


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------
def hybrid_chunks(pipeline: RAGPipeline, query: str, top_k: int, text_map: Dict[Any, str]):
    bm25 = pipeline.retriever.bm25_search(query, top_k=top_k)
    dense = pipeline.retriever.semantic_search(query, top_k=top_k)
    fused = pipeline.retriever.rrf_merge(bm25, dense, top_n=top_k)

    out = []
    for chunk_id, rrf_score in fused:
        if chunk_id not in text_map:
            continue
        out.append({
            "chunk_id": str(chunk_id),
            "text": str(text_map[chunk_id]),
            "rrf_score": float(rrf_score),
        })
    return out


def print_startup_diagnostics():
    csv_path = ROOT / "data" / "processed" / "vietnam_tourism_rag_dataset.csv"
    inv_path = ROOT / "data" / "processed" / "inverted_index.json"
    faiss_path = ROOT / "data" / "vectors" / "faiss_index.bin"
    reranker_path = ROOT / "models" / "run_3" / "vietnam_tourism_reranker_moc_13_seed_42"

    print("\n========== STARTUP DIAGNOSTICS ==========")
    for label, p in [
        ("TEST", TEST_PATH),
        ("CSV", csv_path),
        ("INVERTED INDEX", inv_path),
        ("FAISS", faiss_path),
        ("RERANKER", reranker_path),
    ]:
        exists = p.exists()
        if exists and p.is_file():
            print(f"{label:15s}: {p} | exists=True | size={p.stat().st_size/(1024**2):.2f} MB")
        else:
            print(f"{label:15s}: {p} | exists={exists}")
    print("=========================================\n")


def generate(args: argparse.Namespace) -> None:
    from src.run import RAGPipeline
    if args.overwrite and GENERATIONS_PATH.exists():
        GENERATIONS_PATH.unlink()

    # Old generations lack model/corpus/parameter provenance. Never silently append
    # new rows to that cache under a potentially changed experimental setup.
    legacy_rows = read_jsonl(GENERATIONS_PATH)
    if legacy_rows and not args.overwrite:
        raise RuntimeError(
            "Existing generations may have been produced with different model settings. "
            "Archive faithfulness_generations.jsonl before rerunning generate; "
            "then use --overwrite. For the original pilot, use --mode rescore_v2."
        )
    existing = {(r["query"], r["condition"]) for r in read_jsonl(GENERATIONS_PATH)}
    queries = load_queries(TEST_PATH, args.max_queries, args.seed)

    print(f"\nQueries: {len(queries)} | Generator: {args.llm_model}")
    print("Comparing baseline_rrf vs proposed_reranker")

    print_startup_diagnostics()

    print("[DEBUG] About to initialize RAGPipeline()...")
    pipeline = RAGPipeline()
    print("[DEBUG] RAGPipeline() initialized successfully.")

    df = pipeline.retriever.df
    text_map = dict(zip(df["chunk_id"], df["text_content"]))

    GENERATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for i, query in enumerate(tqdm(queries, desc="Generating"), start=1):
        candidates = hybrid_chunks(pipeline, query, args.top_k, text_map)
        if not candidates:
            continue

        baseline = candidates[:args.top_n]
        reranked = pipeline.reranker.rerank(query, [dict(x) for x in candidates])
        proposed = reranked[:args.top_n]

        for condition, contexts in {
            "baseline_rrf": baseline,
            "proposed_reranker": proposed,
        }.items():
            if (query, condition) in existing:
                continue

            answer = pipeline.generator.generate_answer(
                query, contexts, model_name=args.llm_model
            )

            row = {
                "query_index": i,
                "query": query,
                "condition": condition,
                "generator": args.llm_model,
                "top_k_retrieval": args.top_k,
                "top_n_context": args.top_n,
                "contexts": [clean_context(c) for c in contexts],
                "answer": answer,
            }

            with GENERATIONS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            existing.add((query, condition))

    print(f"\n[SUCCESS] Saved generations: {GENERATIONS_PATH}")


# ---------------------------------------------------------------------
# NLI evaluator
# ---------------------------------------------------------------------
class NLIEvaluator:
    def __init__(self, model_name: str, device: str = "cpu", threshold: float = 0.50):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for NLI but CUDA is unavailable.")

        self.model_name = model_name
        self.device = torch.device(device)
        self.threshold = float(threshold)

        print(f"\nLoading NLI evaluator: {model_name}")
        print(f"NLI device: {self.device}")

        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

        self.entail_idx = None
        for idx, label in (self.model.config.id2label or {}).items():
            if "entail" in str(label).lower():
                self.entail_idx = int(idx)
                break

        if self.entail_idx is None and "moritzlaurer/mdeberta-v3-base-mnli-xnli" in model_name.lower():
            self.entail_idx = 0

        if self.entail_idx is None:
            raise RuntimeError(
                f"Cannot identify entailment label: {self.model.config.id2label}"
            )

    def windows(self, text: str, window: int = 360, stride: int = 300) -> List[str]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= window:
            return [text]

        result = []
        start = 0
        while start < len(ids):
            end = min(start + window, len(ids))
            result.append(self.tokenizer.decode(ids[start:end], skip_special_tokens=True))
            if end == len(ids):
                break
            start += stride
        return result

    @torch.inference_mode()
    def entail_prob(self, premise: str, hypothesis: str) -> float:
        x = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        x = {k: v.to(self.device) for k, v in x.items()}
        probs = torch.softmax(self.model(**x).logits[0], dim=-1)
        return float(probs[self.entail_idx].item())

    def context_faithfulness(
        self,
        answer: str,
        contexts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        claims = split_factual_claims(answer)

        context_windows: List[str] = []
        for c in contexts:
            text = str(c.get("text", "")).strip()
            if text:
                context_windows.extend(self.windows(text))

        details = []
        for claim in claims:
            best = max(
                (self.entail_prob(window, claim) for window in context_windows),
                default=0.0,
            )
            details.append({
                "claim": claim,
                "entailment_probability": best,
                "supported": best >= self.threshold,
            })

        if not details:
            # IMPORTANT: a pure refusal is not a factual hallucination.
            return {
                "context_faithfulness": None,
                "context_hallucination_rate": None,
                "num_factual_claims": 0,
                "supported_factual_claims": 0,
                "claim_details": [],
            }

        supported = sum(int(d["supported"]) for d in details)
        faith = supported / len(details)

        return {
            "context_faithfulness": faith,
            "context_hallucination_rate": 1.0 - faith,
            "num_factual_claims": len(details),
            "supported_factual_claims": supported,
            "claim_details": details,
        }

    def reference_correctness(
        self,
        answer: str,
        reference_answer: str,
    ) -> Optional[float]:
        reference_answer = (reference_answer or "").strip()
        if not reference_answer:
            return None

        # Does the generated answer entail the canonical reference/correction?
        return self.entail_prob(answer, reference_answer)


# ---------------------------------------------------------------------
# Ground-truth-aware behavior scoring
# ---------------------------------------------------------------------
def normalize_for_match(value: str) -> str:
    import unicodedata
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or "")).casefold()).strip()


def fact_value_present(text: str, value: str) -> bool:
    """Match complete Unicode word boundaries, not substrings inside other words."""
    needle = normalize_for_match(value)
    if not needle:
        return False
    return bool(re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)",
                          normalize_for_match(text), flags=re.UNICODE))


def false_value_asserted(answer: str, false_value: str, correct_value: str) -> bool:
    """Conservative trap check: a mere unnegated mention is considered unsafe.

    This is a heuristic; cases with complex quotations or coreference require
    human adjudication before publication. It is NOT a validated factuality test.
    """
    a = normalize_for_match(answer)
    bad = normalize_for_match(false_value)
    good = normalize_for_match(correct_value)
    if not bad:
        return False
    for match in re.finditer(r"(?<!\w)" + re.escape(bad) + r"(?!\w)", a):
        left = re.split(r"[.!?;\n]", a[max(0, match.start()-100):match.start()])[-1]
        right = re.split(r"[.!?;\n]", a[match.end():match.end()+100])[0]
        # 'Yên Định (không phải Yên Bình)' is a correction, not trap acceptance.
        negated = bool(re.search(r"(?:không phải|chứ không phải|thay vì|nhầm|sai là|not|rather than)\s*[(\"'“”]*\s*$", left))
        negated |= bool(re.search(r"^\s*[)\"'“”]*\s*(?:là sai|sai|không đúng|không chính xác|is wrong)", right))
        negated |= bool(good and re.search(
            r"(?:từ|đổi|thay|sửa)\s+" + re.escape(bad) +
            r".{0,30}(?:thành|sang|bằng)\s+" + re.escape(good),
            left + bad + right))
        if not negated:
            return True
    return False


def behavior_outcome(
    answer: str,
    expected_behavior: str,
    reference_correctness: Optional[float],
    threshold: float,
    truth: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    refusal = detect_refusal(answer)
    expected = expected_behavior.strip().lower()
    correction_data = {}

    if expected == "refuse":
        success = refusal
        category = "correct_refusal" if success else "should_refuse_but_answered"
    elif expected == "answer":
        if reference_correctness is None:
            raise ValueError("Answer items require reference correctness scores")
        success = bool(not refusal and reference_correctness >= threshold)
        category = "false_refusal" if refusal else (
            "correct_answer" if success else "incorrect_or_unsupported_answer")
    elif expected == "correct":
        if not truth:
            raise ValueError("Correction requires a ground-truth row with trap values")
        correct_value = str(truth.get("trap_original_value", "")).strip()
        false_value = str(truth.get("trap_replacement_value", "")).strip()
        if not correct_value or not false_value:
            raise ValueError("Correction needs trap_original_value and trap_replacement_value")
        correct_present = fact_value_present(answer, correct_value)
        trap_accepted = false_value_asserted(answer, false_value, correct_value)
        # Do not require reference NLI: it can give false positive for a wrong entity.
        success = bool(correct_present and not trap_accepted)
        category = "correct_correction" if success else "failed_to_correct"
        correction_data = {
            "correct_value_present": correct_present,
            "trap_accepted_heuristic": trap_accepted,
            "correction_scoring_method": "explicit_value_rule_v3",
        }
    else:
        raise ValueError(f"Unknown expected_behavior: {expected}")

    return {
        "contains_refusal": refusal,
        "behavior_success": bool(success),
        "behavior_outcome": category,
        **correction_data,
    }


# ---------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------
def bootstrap_ci(values: List[float], seed: int = 42, n_boot: int = 2000):
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]

    rng = random.Random(seed)
    means = []
    n = len(values)

    for _ in range(n_boot):
        means.append(statistics.mean(values[rng.randrange(n)] for _ in range(n)))

    means.sort()
    return [
        means[int(0.025 * (n_boot - 1))],
        means[int(0.975 * (n_boot - 1))],
    ]


def condition_summary(rows: List[Dict[str, Any]], condition: str, seed: int):
    sub = [r for r in rows if r["condition"] == condition]

    faith = [
        float(r["context_faithfulness"])
        for r in sub
        if r.get("context_faithfulness") is not None
    ]

    behavior_values = [int(bool(r["behavior_success"])) for r in sub]
    total_claims = sum(int(r.get("num_factual_claims", 0)) for r in sub)
    supported_claims = sum(int(r.get("supported_factual_claims", 0)) for r in sub)

    behavior_counts: Dict[str, int] = {}
    for r in sub:
        key = str(r.get("behavior_outcome", ""))
        behavior_counts[key] = behavior_counts.get(key, 0) + 1

    return {
        "answers": len(sub),
        "faithfulness_scored_answers": len(faith),
        "mean_context_faithfulness": statistics.mean(faith) if faith else None,
        "context_faithfulness_95ci": bootstrap_ci(faith, seed),
        "micro_context_faithfulness": (
            supported_claims / total_claims if total_claims else None
        ),
        "context_hallucination_rate": (
            1.0 - supported_claims / total_claims if total_claims else None
        ),
        "behavior_success_rate": (
            statistics.mean(behavior_values) if behavior_values else None
        ),
        "behavior_success_95ci": bootstrap_ci(behavior_values, seed),
        "refusal_rate": (
            sum(int(bool(r.get("contains_refusal"))) for r in sub) / len(sub)
            if sub else None
        ),
        "total_factual_claims": total_claims,
        "supported_factual_claims": supported_claims,
        "behavior_outcome_counts": behavior_counts,
    }


def paired_comparison(rows: List[Dict[str, Any]], seed: int):
    b = {
        r["query"]: r
        for r in rows
        if r["condition"] == "baseline_rrf"
    }
    p = {
        r["query"]: r
        for r in rows
        if r["condition"] == "proposed_reranker"
    }

    common = sorted(set(b) & set(p))

    behavior_diffs = [
        int(bool(p[q]["behavior_success"])) - int(bool(b[q]["behavior_success"]))
        for q in common
    ]

    out = {
        "paired_queries": len(common),
        "mean_behavior_delta_proposed_minus_baseline": (
            statistics.mean(behavior_diffs) if behavior_diffs else None
        ),
        "behavior_delta_95ci": bootstrap_ci(behavior_diffs, seed),
    }

    # Faithfulness is compared only where BOTH systems produced factual claims.
    faith_common = [
        q for q in common
        if b[q].get("context_faithfulness") is not None
        and p[q].get("context_faithfulness") is not None
    ]

    faith_diffs = [
        float(p[q]["context_faithfulness"]) - float(b[q]["context_faithfulness"])
        for q in faith_common
    ]

    out["paired_faithfulness_queries"] = len(faith_common)
    out["mean_faithfulness_delta_proposed_minus_baseline"] = (
        statistics.mean(faith_diffs) if faith_diffs else None
    )
    out["faithfulness_delta_95ci"] = bootstrap_ci(faith_diffs, seed)

    # McNemar is the natural paired test for binary behavior success.
    try:
        from statsmodels.stats.contingency_tables import mcnemar

        n01 = sum(
            int((not bool(b[q]["behavior_success"])) and bool(p[q]["behavior_success"]))
            for q in common
        )
        n10 = sum(
            int(bool(b[q]["behavior_success"]) and (not bool(p[q]["behavior_success"])))
            for q in common
        )

        table = [[0, n01], [n10, 0]]
        test = mcnemar(table, exact=True)

        out["mcnemar_baseline_fail_proposed_success"] = n01
        out["mcnemar_baseline_success_proposed_fail"] = n10
        out["mcnemar_p_value"] = float(test.pvalue)
    except Exception as exc:
        out["mcnemar_note"] = f"Not computed: {exc}"

    # Wilcoxon for paired continuous faithfulness when defined.
    if faith_diffs:
        try:
            from scipy.stats import wilcoxon

            bv = [float(b[q]["context_faithfulness"]) for q in faith_common]
            pv = [float(p[q]["context_faithfulness"]) for q in faith_common]

            if any(abs(x - y) > 1e-12 for x, y in zip(bv, pv)):
                stat, pval = wilcoxon(pv, bv)
                out["faithfulness_wilcoxon_statistic"] = float(stat)
                out["faithfulness_wilcoxon_p_value"] = float(pval)
            else:
                out["faithfulness_wilcoxon_statistic"] = 0.0
                out["faithfulness_wilcoxon_p_value"] = 1.0
        except Exception as exc:
            out["faithfulness_wilcoxon_note"] = f"Not computed: {exc}"

    return out


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def write_jsonl_atomic(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def check_v3_outputs(args: argparse.Namespace, mode: str) -> Path:
    """Disallow mixing V3 records from different GT, NLI, or old V2 sources."""
    manifest = OUT_DIR / "faithfulness_v3_manifest.json"
    source = SCORED_V2_PATH if mode == "rescore_v2" else GENERATIONS_PATH
    signature = {
        "schema": SCORING_SCHEMA,
        "mode": mode,
        "ground_truth_sha256": sha256_file(GROUND_TRUTH_PATH),
        "source_sha256": sha256_file(source),
        "script_sha256": sha256_file(Path(__file__)),
        "nli_model": args.nli_model,
        "threshold": args.threshold,
        "seed": args.seed,
    }
    signature["sha256"] = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    existing = [p for p in (SCORED_V3_PATH, SUMMARY_V3_PATH, manifest) if p.exists()]
    if existing and not args.overwrite:
        if manifest.exists():
            old = json.loads(manifest.read_text(encoding="utf-8"))
            if old.get("sha256") == signature["sha256"] and SCORED_V3_PATH.exists() and SUMMARY_V3_PATH.exists():
                print("[INFO] Complete v3 outputs already exist and match all inputs; nothing overwritten.")
                return Path("")
        raise RuntimeError("Existing V3 files or provenance differ. Back them up and explicitly use --overwrite.")
    return manifest


def validate_pairs(generations: List[Dict[str, Any]]) -> None:
    keys = [(r.get("query"), r.get("condition")) for r in generations]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate (query, condition) records")
    by_query: Dict[str, set] = {}
    for q, cond in keys:
        by_query.setdefault(q, set()).add(cond)
    for q, conds in by_query.items():
        if conds != {"baseline_rrf", "proposed_reranker"}:
            raise ValueError(f"Incomplete or unexpected conditions for query: {q}: {conds}")


def make_summary(scored_rows: List[Dict[str, Any]], gt: Dict[str, Dict[str, Any]],
                 args: argparse.Namespace, mode: str) -> Dict[str, Any]:
    summary = {
        "baseline_rrf": condition_summary(scored_rows, "baseline_rrf", args.seed),
        "proposed_reranker": condition_summary(scored_rows, "proposed_reranker", args.seed),
        "paired_comparison": paired_comparison(scored_rows, args.seed),
        "ground_truth_distribution": {},
        "metadata": {
            "generator": args.llm_model,
            "faithfulness_evaluator": args.nli_model,
            "entailment_threshold": args.threshold,
            "seed": args.seed,
            "ground_truth_sha256": sha256_file(GROUND_TRUTH_PATH),
            "generation_sha256": sha256_file(GENERATIONS_PATH),
            "scoring_schema": SCORING_SCHEMA,
            "score_mode": mode,
            "correction_rule": "explicit corrected value present AND false value not unnegated",
            "important_limitations": [
                "Pilot of 30 queries, not the final 200-query benchmark",
                "Heuristic correction/refusal decisions require human review",
                "26/30 gold documents were external pseudo-documents in the prior corpus audit",
            ],
        },
    }
    if mode == "rescore_v2":
        summary["metadata"]["nli_recomputed"] = False
        summary["metadata"]["faithfulness_metrics"] = "REUSED FROM ARCHIVED V2; not a newly recomputed NLI evaluation"
        summary["metadata"]["original_v2_scored_sha256"] = sha256_file(SCORED_V2_PATH)
    else:
        summary["metadata"]["nli_recomputed"] = True
    for item in gt.values():
        behavior = str(item.get("expected_behavior", "")).strip().lower()
        summary["ground_truth_distribution"][behavior] = (
            summary["ground_truth_distribution"].get(behavior, 0) + 1)
    return summary


def finalize_v3(scored_rows: List[Dict[str, Any]], gt: Dict[str, Dict[str, Any]],
                args: argparse.Namespace, mode: str, manifest: Path) -> None:
    if len(scored_rows) != 2 * len(gt):
        raise ValueError(f"Expected 2 records per ground-truth query: got {len(scored_rows)} for {len(gt)}")
    summary = make_summary(scored_rows, gt, args, mode)
    write_jsonl_atomic(SCORED_V3_PATH, scored_rows)
    write_json_atomic(SUMMARY_V3_PATH, summary)
    source = SCORED_V2_PATH if mode == "rescore_v2" else GENERATIONS_PATH
    signature = {
        "schema": SCORING_SCHEMA,
        "mode": mode,
        "ground_truth_sha256": sha256_file(GROUND_TRUTH_PATH),
        "source_sha256": sha256_file(source),
        "script_sha256": sha256_file(Path(__file__)),
        "nli_model": args.nli_model,
        "threshold": args.threshold,
        "seed": args.seed,
    }
    signature["sha256"] = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    write_json_atomic(manifest, signature)
    print("\nV3 OUTPUTS (V2 is unchanged):")
    print("  ", SCORED_V3_PATH)
    print("  ", SUMMARY_V3_PATH)
    print("  ", manifest)
    print("  baseline behavior success:", summary["baseline_rrf"]["behavior_success_rate"])
    print("  proposed behavior success:", summary["proposed_reranker"]["behavior_success_rate"])


def score(args: argparse.Namespace, reuse_v2: bool = False) -> None:
    mode = "rescore_v2" if reuse_v2 else "score"
    generations = read_jsonl(GENERATIONS_PATH)
    if not generations:
        raise FileNotFoundError(f"Missing original generations: {GENERATIONS_PATH}")
    validate_pairs(generations)
    gt = load_ground_truth(GROUND_TRUTH_PATH)
    if {r["query"] for r in generations} != set(gt):
        raise ValueError("Ground truth and generation queries do not match exactly")

    manifest = check_v3_outputs(args, mode)
    if str(manifest) == ".":
        return

    scored_rows = []
    if reuse_v2:
        old_rows = read_jsonl(SCORED_V2_PATH)
        validate_pairs(old_rows)
        old_map = {(r["query"], r["condition"]): r for r in old_rows}
        if len(old_map) != len(generations):
            raise ValueError("Archived V2 scores do not cover generations")
        for row in generations:
            old = old_map.get((row["query"], row["condition"]))
            if old is None or old.get("answer") != row.get("answer") or old.get("contexts") != row.get("contexts"):
                raise ValueError(f"V2 scores do not match generation: {row['query']}")
            if str(old.get("faithfulness_evaluator")) != args.nli_model or abs(float(old.get("entailment_threshold", -1))-args.threshold)>1e-9:
                raise ValueError("Cannot reuse different NLI model or threshold")
            truth = gt[row["query"]]
            result = dict(old)
            result.update(behavior_outcome(
                row["answer"], truth["expected_behavior"],
                old.get("reference_correctness"), args.threshold, truth))
            result.update({
                "scoring_schema": SCORING_SCHEMA,
                "trap_original_value": truth.get("trap_original_value", ""),
                "trap_replacement_value": truth.get("trap_replacement_value", ""),
                "nli_recomputed": False,
                "legacy_v2_context_faithfulness": True,
            })
            scored_rows.append(result)
    else:
        evaluator = NLIEvaluator(args.nli_model, args.nli_device, args.threshold)
        for row in tqdm(generations, desc="Scoring v3"):
            truth = gt[row["query"]]
            ref_correctness = evaluator.reference_correctness(row["answer"], truth.get("reference_answer", ""))
            result = dict(row)
            result.update(evaluator.context_faithfulness(row["answer"], row.get("contexts", [])))
            result.update(behavior_outcome(row["answer"], truth["expected_behavior"],
                                           ref_correctness, args.threshold, truth))
            result.update({
                "expected_behavior": truth["expected_behavior"],
                "reference_answer": truth.get("reference_answer", ""),
                "reference_correctness": ref_correctness,
                "gold_evidence": truth.get("gold_evidence", ""),
                "ground_truth_notes": truth.get("notes", ""),
                "faithfulness_evaluator": args.nli_model,
                "entailment_threshold": args.threshold,
                "trap_original_value": truth.get("trap_original_value", ""),
                "trap_replacement_value": truth.get("trap_replacement_value", ""),
                "nli_recomputed": True,
                "scoring_schema": SCORING_SCHEMA,
            })
            scored_rows.append(result)
    finalize_v3(scored_rows, gt, args, mode, manifest)


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "generate":
        generate(args)

    elif args.mode == "init_gt":
        create_ground_truth_template(overwrite=args.overwrite)

    elif args.mode == "score":
        score(args)

    elif args.mode == "rescore_v2":
        score(args, reuse_v2=True)

    elif args.mode == "all":
        generate(args)
        if not GROUND_TRUTH_PATH.exists():
            create_ground_truth_template(overwrite=False)
            print(
                "\n[STOP] Generation finished and a GT template was created. "
                "Fill the GT file manually, then run --mode score."
            )
            return
        score(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] Interrupted by user.")
        raise
    except Exception as exc:
        print("\n================ FATAL PYTHON ERROR ================")
        print(f"Type   : {type(exc).__name__}")
        print(f"Message: {exc}")
        print("Full traceback:")
        traceback.print_exc()
        print("====================================================")
        raise
