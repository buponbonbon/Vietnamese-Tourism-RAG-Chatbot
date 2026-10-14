from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------
# Put this file in:
#   Vietnamese-Tourism-RAG-Chatbot/
#   results/downstream_benchmark/downstream_eval_abcd.py
#
# Run from project root.
#
# Recommended workflow:
#
# 1) Technical pilot on the draft (works with .json OR .csv):
#    python -m results.downstream_benchmark.downstream_eval_abcd \
#        --mode prepare --max_queries 10 --allow_draft
#    python -m results.downstream_benchmark.downstream_eval_abcd \
#        --mode generate --max_queries 10 --allow_draft
#    python -m results.downstream_benchmark.downstream_eval_abcd \
#        --mode score --max_queries 10 --allow_draft
#
# Run these same commands with --overwrite ONLY when intentionally starting a
# NEW experiment; back up abc_outputs/ first. Stage manifests prevent accidental
# mixing of stale contexts, LLM answers, or NLI scores after inputs change.
# Do NOT report DRAFT results as validated findings in the paper.
# Final experiment: manually verify ALL annotations, set statuses to DONE,
# run --max_queries 0 and OMIT --allow_draft, with fresh output cache.
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.retrieval import HybridRetriever
from src.reranking import TourismReranker
from src.generation import LanguageGenerator

# ---------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------

# Use the annotated CSV as the default source of truth for this four-file
# review. To use an already human-validated JSON instead, pass --benchmark.
DEFAULT_BENCHMARK = THIS_DIR / "benchmark_annotation_draft_200.csv"
if not DEFAULT_BENCHMARK.exists():
    DEFAULT_BENCHMARK = THIS_DIR / "benchmark_annotation_draft_200.json"
OUT_DIR = THIS_DIR / "abc_outputs"
CACHE_SCHEMA_VERSION = "tourism-abcd-audit-v3"

CSV_PATH = ROOT / "data" / "processed" / "vietnam_tourism_rag_dataset.csv"
INVERTED_INDEX_PATH = ROOT / "data" / "processed" / "inverted_index.json"
FAISS_DIR = ROOT / "data" / "vectors"

LLM_MODEL = "qwen2.5:14b-instruct-q4_K_M"
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

DEFAULT_TOP_K = 10
DEFAULT_TOP_N = 3
DEFAULT_SEED = 42
DEFAULT_THRESHOLD = 0.50

CONDITIONS = (
    "A_rrf_only",
    "B_reranker_0pct",
    "C_reranker_6p85pct",
    "D_reranker_13p7pct",
)

CONDITION_LABELS = {
    "A_rrf_only": "Hybrid RRF only",
    "B_reranker_0pct": "Hybrid RRF + 0% adversarial reranker",
    "C_reranker_6p85pct": "Hybrid RRF + 6.85% adversarial reranker",
    "D_reranker_13p7pct": "Hybrid RRF + 13.7% Full-Trap reranker",
}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Four-way downstream evaluation: RRF vs 0% vs 6.85% vs 13.7% adversarial rerankers."
    )

    p.add_argument(
        "--mode",
        choices=["prepare", "generate", "score", "all"],
        default="prepare",
    )
    p.add_argument("--benchmark", type=str, default=str(DEFAULT_BENCHMARK))
    p.add_argument("--max_queries", type=int, default=0, help="0 = all benchmark rows")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--top_n", type=int, default=DEFAULT_TOP_N)

    p.add_argument("--llm_model", default=LLM_MODEL)
    p.add_argument("--nli_model", default=NLI_MODEL)
    p.add_argument("--nli_device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    p.add_argument(
        "--baseline_model",
        type=str,
        default="",
        help="Optional explicit path to moc_0 seed_42 checkpoint.",
    )
    p.add_argument(
        "--moderate_model",
        type=str,
        default="",
        help=(
            "Optional explicit path to the 6.85% adversarial seed_42 checkpoint. "
            "If omitted, the script searches models/run_2."
        ),
    )
    p.add_argument(
        "--fulltrap_model",
        type=str,
        default="",
        help="Optional explicit path to the 13.7% Full-Trap seed_42 checkpoint.",
    )

    p.add_argument(
        "--allow_draft",
        action="store_true",
        help="Allow DRAFT_AUTO/TODO annotations. Use only for technical pilot, never final reported results.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace this mode's cache and manifest. Back up abc_outputs first; older files are NOT migrated.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    """Content digest for provenance-critical small/medium files."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def file_identity(path: Path, hash_contents: bool = False) -> Dict[str, Any]:
    """Preserve file identity; large model/index files use size + mtime."""
    if not path.is_file():
        return {"path": str(path.resolve()), "exists": False}
    stat = path.stat()
    result = {"path": str(path.resolve()), "size": stat.st_size,
              "mtime_ns": stat.st_mtime_ns}
    if hash_contents:
        result["sha256"] = sha256_file(path)
    return result


def directory_identity(path: Path) -> Dict[str, Any]:
    if not path.is_dir():
        return {"path": str(path.resolve()), "exists": False}
    return {"path": str(path.resolve()), "files": [
        {"name": str(f.relative_to(path)), "size": f.stat().st_size,
         "mtime_ns": f.stat().st_mtime_ns}
        for f in sorted(path.rglob("*")) if f.is_file()
    ]}


def stage_manifest_path(output: Path) -> Path:
    return output.with_name(output.name + ".manifest.json")


def assert_cache_signature(output: Path, signature: str, stage: str) -> None:
    manifest_path = stage_manifest_path(output)
    if not output.exists():
        raise FileNotFoundError(f"{stage} output missing: {output}. Run the earlier stage first.")
    if not manifest_path.is_file():
        raise RuntimeError(
            f"LEGACY CACHE BLOCKED ({stage}): {output} has no provenance manifest. "
            "Back up abc_outputs/, then rerun the stage with --overwrite. "
            "Old and new benchmark/checkpoint results must never be mixed."
        )
    try:
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise RuntimeError(f"Unreadable {stage} cache manifest: {manifest_path}") from exc
    if meta.get("signature") != signature or meta.get("schema") != CACHE_SCHEMA_VERSION:
        raise RuntimeError(
            f"STALE CACHE BLOCKED ({stage}): {output} was created with different "
            "benchmark, code, model, index, or parameters. Back up abc_outputs/ "
            "and recompute this stage and all downstream stages using --overwrite."
        )


def init_stage_cache(output: Path, spec: Dict[str, Any], stage: str,
                     overwrite: bool) -> str:
    """Write a provenance record BEFORE append, allowing safe interrupted resumes."""
    signature = json_digest(spec)
    manifest_path = stage_manifest_path(output)
    if overwrite:
        output.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    if output.exists():
        assert_cache_signature(output, signature, stage)
        return signature

    if manifest_path.exists():
        # A previous stage may have been interrupted after creating its
        # manifest but before writing the first JSONL row. Resume only if
        # the original input signature is unchanged.
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise RuntimeError(f"Unreadable {stage} manifest: {manifest_path}") from exc
        if prior.get("signature") != signature or prior.get("schema") != CACHE_SCHEMA_VERSION:
            raise RuntimeError(
                f"STALE EMPTY CACHE BLOCKED ({stage}): {manifest_path}; "
                "back up prior output and rerun with --overwrite."
            )
        return signature

    meta = {"schema": CACHE_SCHEMA_VERSION, "stage": stage,
            "signature": signature, "inputs": spec}
    temp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, manifest_path)
    return signature


def assert_unique_condition_rows(rows: List[Dict[str, Any]], stage: str) -> None:
    keys = [(str(r.get("benchmark_id")), str(r.get("condition"))) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate (benchmark_id, condition) rows detected in {stage} cache.")


def benchmark_signature(args: argparse.Namespace, benchmark: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"file": file_identity(Path(args.benchmark), hash_contents=True),
            "selected_rows_sha256": json_digest(benchmark),
            "selected_count": len(benchmark), "max_queries": args.max_queries,
            "allow_draft": bool(args.allow_draft)}


def checkpoint_paths(args: argparse.Namespace) -> Dict[str, Path]:
    return {
        "B": resolve_checkpoint(args.baseline_model, variant="0"),
        "C": resolve_checkpoint(args.moderate_model, variant="6.85"),
        "D": resolve_checkpoint(args.fulltrap_model, variant="13.7"),
    }


def prepare_spec(args: argparse.Namespace, benchmark: List[Dict[str, Any]],
                 checkpoints: Dict[str, Path]) -> Dict[str, Any]:
    return {
        "version": CACHE_SCHEMA_VERSION, "stage": "prepare",
        "benchmark": benchmark_signature(args, benchmark),
        "top_k": args.top_k, "top_n": args.top_n,
        "retrieval_corpus": file_identity(CSV_PATH),
        "inverted_index": file_identity(INVERTED_INDEX_PATH),
        "faiss_index": directory_identity(FAISS_DIR),
        "retrieval_source": file_identity(ROOT / "src" / "retrieval.py", True),
        "reranker_source": file_identity(ROOT / "src" / "reranking.py", True),
        "checkpoints": {name: directory_identity(path) for name, path in checkpoints.items()},
    }


def checked_prepare_signature(args: argparse.Namespace,
                              benchmark: List[Dict[str, Any]]) -> str:
    checkpoints = checkpoint_paths(args)
    signature = json_digest(prepare_spec(args, benchmark, checkpoints))
    assert_cache_signature(OUT_DIR / "abc_contexts.jsonl", signature, "prepare")
    return signature


def generate_spec(args: argparse.Namespace, benchmark: List[Dict[str, Any]],
                 context_sha: str, prepare_signature: str) -> Dict[str, Any]:
    return {
        "version": CACHE_SCHEMA_VERSION, "stage": "generate",
        "benchmark": benchmark_signature(args, benchmark),
        "context_sha256": context_sha, "prepare_signature": prepare_signature,
        "generator": args.llm_model,
        "generation_source": file_identity(ROOT / "src" / "generation.py", True),
    }


def checked_generation_signature(args: argparse.Namespace,
                                 benchmark: List[Dict[str, Any]]) -> str:
    prepare_signature = checked_prepare_signature(args, benchmark)
    context_path = OUT_DIR / "abc_contexts.jsonl"
    generation_path = OUT_DIR / "abc_generations.jsonl"
    signature = json_digest(generate_spec(args, benchmark, sha256_file(context_path),
                                          prepare_signature))
    assert_cache_signature(generation_path, signature, "generate")
    return signature


def load_benchmark(path: Path, max_queries: int, allow_draft: bool) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as f:
            rows = json.load(f)
    else:
        raise ValueError("Benchmark must be a .json or .csv file.")

    if not isinstance(rows, list):
        raise ValueError("Benchmark must contain a list of annotated rows.")

    required = {
        "benchmark_id",
        "target_behavior",
        "query",
        "reference_answer",
        "gold_chunk_id",
        "gold_evidence",
        "annotation_status",
    }

    cleaned = []
    seen_ids = set()

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Benchmark row {i} must be a dictionary.")
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"Benchmark row {i} missing fields: {sorted(missing)}")

        bid = str(row["benchmark_id"]).strip()
        if not bid or bid in seen_ids:
            raise ValueError(f"Empty or duplicated benchmark_id at row {i}: {bid!r}")
        seen_ids.add(bid)
        if not str(row["gold_chunk_id"]).strip() or not str(row["gold_evidence"]).strip():
            raise ValueError(f"{bid}: gold_chunk_id/gold_evidence must be nonempty.")

        if not str(row["query"]).strip():
            raise ValueError(f"{row['benchmark_id']}: query is empty.")
        if not str(row["reference_answer"]).strip():
            raise ValueError(f"{row['benchmark_id']}: reference_answer is empty.")

        behavior = str(row["target_behavior"]).strip().lower()
        if behavior not in {"answer", "correct"}:
            raise ValueError(
                f"{row['benchmark_id']}: target_behavior must be 'answer' or 'correct'."
            )

        if behavior == "correct":
            if not str(row.get("trap_original_value", "")).strip() or not str(row.get("trap_replacement_value", "")).strip():
                raise ValueError(f"{bid}: correction requires both trap values.")
            if not fact_value_present(str(row["query"]), str(row["trap_replacement_value"])):
                raise ValueError(f"{bid}: injected false value not found in question.")

        status = str(row.get("annotation_status", "")).strip().upper()
        if not allow_draft and status != "DONE":
            raise ValueError(
                f"{row['benchmark_id']}: annotation_status={status!r}. "
                "Final evaluation requires every row to be manually verified and set to DONE. "
                "Use --allow_draft only for a technical pilot."
            )

        cleaned.append(row)

    if max_queries > 0:
        cleaned = cleaned[:max_queries]

    if not cleaned:
        raise ValueError("No benchmark rows selected.")

    return cleaned


# ---------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------

def checkpoint_ok(path: Path) -> bool:
    return (
        path.exists()
        and path.is_dir()
        and (path / "config.json").exists()
        and (path / "model.safetensors").exists()
    )


def resolve_checkpoint(explicit: str, variant: str) -> Path:
    """
    Resolve one of the three reranker checkpoints used in the A/B/C/D study.

    Expected training layout:
      0%     -> models/run_1/
      6.85%  -> models/run_2/
      13.7%  -> models/run_3/

    The 6.85% and 13.7% directories may have been named with rounded MOC
    suffixes during training, so several safe filename variants are tried.
    """
    variant = str(variant).strip()

    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not checkpoint_ok(p):
            raise FileNotFoundError(
                f"Explicit reranker checkpoint is invalid or incomplete: {p}"
            )
        return p

    specs = {
        "0": {
            "preferred_run": "run_1",
            "cli": "--baseline_model",
            "names": [
                "vietnam_tourism_reranker_moc_0_seed_42",
            ],
        },
        "6.85": {
            "preferred_run": "run_2",
            "cli": "--moderate_model",
            "names": [
                "vietnam_tourism_reranker_moc_6.85_seed_42",
                "vietnam_tourism_reranker_moc_6_85_seed_42",
                "vietnam_tourism_reranker_moc_6p85_seed_42",
                "vietnam_tourism_reranker_moc_6_seed_42",
            ],
        },
        "13.7": {
            "preferred_run": "run_3",
            "cli": "--fulltrap_model",
            "names": [
                "vietnam_tourism_reranker_moc_13_seed_42",
                "vietnam_tourism_reranker_moc_13.7_seed_42",
                "vietnam_tourism_reranker_moc_13_7_seed_42",
                "vietnam_tourism_reranker_moc_13p7_seed_42",
            ],
        },
    }

    if variant not in specs:
        raise ValueError(f"Unknown checkpoint variant: {variant}")

    spec = specs[variant]
    # NEVER silently accept a checkpoint from another experimental run.
    runs = [spec["preferred_run"]]

    candidates: List[Path] = []

    for name in spec["names"]:
        for run in runs:
            candidates.append(ROOT / "models" / run / name)

        # For nonstandard checkpoint locations use --*_model explicitly.

    # Deduplicate while preserving search order.
    unique_candidates = []
    seen = set()

    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(p)

    for p in unique_candidates:
        if checkpoint_ok(p):
            return p

    tried = "\n".join(f"  - {p}" for p in unique_candidates)

    raise FileNotFoundError(
        f"Could not find the {variant}% adversarial checkpoint.\n"
        f"Tried:\n{tried}\n"
        f"Pass it explicitly with {spec['cli']} <PATH>."
    )


# ---------------------------------------------------------------------
# Retrieval / reranking preparation
# ---------------------------------------------------------------------

def clean_context(c: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "chunk_id": str(c.get("chunk_id", "")),
        "text": str(c.get("text", "")),
    }

    for key in ("rrf_score", "rerank_score"):
        if key in c:
            try:
                out[key] = float(c[key])
            except Exception:
                out[key] = None

    return out


def hybrid_candidates(
    retriever: HybridRetriever,
    query: str,
    top_k: int,
    text_map: Dict[Any, str],
) -> List[Dict[str, Any]]:
    bm25 = retriever.bm25_search(query, top_k=top_k)
    dense = retriever.semantic_search(query, top_k=top_k)
    fused = retriever.rrf_merge(bm25, dense, top_n=top_k)

    out = []

    for chunk_id, rrf_score in fused:
        # Handle mixed numpy/native/string chunk ids robustly.
        if chunk_id in text_map:
            text = text_map[chunk_id]
        elif str(chunk_id) in text_map:
            text = text_map[str(chunk_id)]
        else:
            continue

        out.append({
            "chunk_id": str(chunk_id),
            "text": str(text),
            "rrf_score": float(rrf_score),
        })

    return out



def rerank_unfiltered(
    reranker: TourismReranker,
    query: str,
    hybrid_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Cross-Encoder reranking WITHOUT the confidence_threshold filter in
    src/reranking.py.

    This is intentional for the A/B/C/D experiment:
      A = RRF Top-N
      B = MOC-0 reranker Top-N
      C = MOC-6.85 reranker Top-N
      D = MOC-13.7 reranker Top-N

    B/C/D must receive the same RRF candidate pool and must all return
    exactly their top-ranked candidates (subject only to candidate count).
    A raw-logit threshold would confound ranking quality with score calibration.
    """
    if not hybrid_results:
        return []

    pairs = [[query, chunk["text"]] for chunk in hybrid_results]

    with torch.inference_mode():
        inputs = reranker.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(reranker.device)

        scores = (
            reranker.model(**inputs)
            .logits.view(-1)
            .float()
            .cpu()
            .tolist()
        )

    out = [dict(chunk) for chunk in hybrid_results]

    for i, score in enumerate(scores):
        out[i]["rerank_score"] = float(score)

    return sorted(
        out,
        key=lambda x: x["rerank_score"],
        reverse=True,
    )



def unload_reranker(reranker: Optional[TourismReranker]) -> None:
    if reranker is None:
        return

    try:
        del reranker.model
    except Exception:
        pass

    try:
        del reranker.tokenizer
    except Exception:
        pass

    del reranker
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_context_row(
    benchmark_row: Dict[str, Any],
    condition: str,
    contexts: List[Dict[str, Any]],
    candidate_ids: List[str],
    top_k: int,
    top_n: int,
) -> Dict[str, Any]:
    gold_id = str(benchmark_row["gold_chunk_id"])
    selected_ids = [str(c["chunk_id"]) for c in contexts]

    return {
        "benchmark_id": benchmark_row["benchmark_id"],
        "target_behavior": benchmark_row["target_behavior"],
        "trap_type": benchmark_row.get("trap_type", ""),
        "trap_original_value": benchmark_row.get("trap_original_value", ""),
        "trap_replacement_value": benchmark_row.get("trap_replacement_value", ""),
        "query": benchmark_row["query"],
        "reference_answer": benchmark_row["reference_answer"],
        "gold_chunk_id": gold_id,
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "top_k_retrieval": top_k,
        "top_n_context": top_n,
        "gold_in_rrf_top_k": gold_id in candidate_ids,
        "gold_in_selected_top_n": gold_id in selected_ids,
        "rrf_candidate_ids": candidate_ids,
        "contexts": [clean_context(c) for c in contexts],
    }


def prepare(args: argparse.Namespace, benchmark: List[Dict[str, Any]]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    context_path = OUT_DIR / "abc_contexts.jsonl"

    checkpoints = checkpoint_paths(args)
    init_stage_cache(context_path, prepare_spec(args, benchmark, checkpoints),
                     "prepare", args.overwrite)

    existing_rows = read_jsonl(context_path)
    assert_unique_condition_rows(existing_rows, "prepare")
    existing = {
        (r["benchmark_id"], r["condition"])
        for r in existing_rows
        if r.get("condition") in CONDITIONS
    }

    needed = {
        (r["benchmark_id"], cond)
        for r in benchmark
        for cond in CONDITIONS
    }

    if needed.issubset(existing):
        print(f"[INFO] All requested A/B/C/D context rows already exist: {context_path}")
        return context_path

    baseline_path = checkpoints["B"]
    moderate_path = checkpoints["C"]
    fulltrap_path = checkpoints["D"]

    print("\n========== A/B/C/D PREPARATION ==========")
    print(f"Benchmark       : {args.benchmark}")
    print(f"Rows            : {len(benchmark)}")
    print(f"Top-K RRF       : {args.top_k}")
    print(f"Top-N context   : {args.top_n}")
    print(f"B / 0%          : {baseline_path}")
    print(f"C / 6.85%       : {moderate_path}")
    print(f"D / 13.7%       : {fulltrap_path}")
    print("Rerank policy   : SAME RRF Top-K -> raw logits -> descending Top-N")
    print("Confidence gate : DISABLED for all rerankers")
    print("=========================================\n")

    print("[1/5] Loading HybridRetriever once...")
    retriever = HybridRetriever(
        csv_path=str(CSV_PATH),
        index_path=str(INVERTED_INDEX_PATH),
        faiss_index_dir=str(FAISS_DIR),
    )

    df = retriever.df

    # Store both native and string ids to avoid dtype mismatch.
    text_map: Dict[Any, str] = {}
    for cid, text_value in zip(df["chunk_id"], df["text_content"]):
        text_map[cid] = str(text_value)
        text_map[str(cid)] = str(text_value)

    print("[2/5] Retrieving the SAME Top-K RRF candidate pool for A/B/C/D...")

    candidate_cache: Dict[str, List[Dict[str, Any]]] = {}

    for row in tqdm(benchmark, desc="RRF retrieval"):
        candidates = hybrid_candidates(
            retriever,
            row["query"],
            args.top_k,
            text_map,
        )
        candidate_cache[row["benchmark_id"]] = candidates

        key = (row["benchmark_id"], "A_rrf_only")

        if key not in existing:
            selected = [dict(x) for x in candidates[:args.top_n]]

            append_jsonl(
                context_path,
                make_context_row(
                    benchmark_row=row,
                    condition="A_rrf_only",
                    contexts=selected,
                    candidate_ids=[str(x["chunk_id"]) for x in candidates],
                    top_k=args.top_k,
                    top_n=args.top_n,
                ),
            )
            existing.add(key)

    # Dense retriever is no longer needed after the common candidate pool is cached.
    try:
        del retriever.model
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def run_reranker_condition(
        condition: str,
        short_label: str,
        model_path: Path,
        step: str,
        tqdm_label: str,
    ) -> None:
        print(
            f"[{step}] Loading {short_label} reranker and reranking "
            "the SAME cached candidate pool..."
        )

        reranker = TourismReranker(model_name=str(model_path))

        try:
            for row in tqdm(benchmark, desc=tqdm_label):
                key = (row["benchmark_id"], condition)

                if key in existing:
                    continue

                candidates = [
                    dict(x)
                    for x in candidate_cache[row["benchmark_id"]]
                ]

                reranked = rerank_unfiltered(
                    reranker,
                    row["query"],
                    candidates,
                )
                selected = reranked[:args.top_n]

                if len(candidates) >= args.top_n and len(selected) != args.top_n:
                    raise RuntimeError(
                        f"{row['benchmark_id']} / {condition}: "
                        f"expected {args.top_n} contexts from "
                        f"{len(candidates)} RRF candidates, got {len(selected)}."
                    )

                append_jsonl(
                    context_path,
                    make_context_row(
                        benchmark_row=row,
                        condition=condition,
                        contexts=selected,
                        candidate_ids=[
                            str(x["chunk_id"])
                            for x in candidate_cache[row["benchmark_id"]]
                        ],
                        top_k=args.top_k,
                        top_n=args.top_n,
                    ),
                )
                existing.add(key)

        finally:
            unload_reranker(reranker)

    run_reranker_condition(
        condition="B_reranker_0pct",
        short_label="B / 0% adversarial",
        model_path=baseline_path,
        step="3/5",
        tqdm_label="B: 0% reranker",
    )

    run_reranker_condition(
        condition="C_reranker_6p85pct",
        short_label="C / 6.85% adversarial",
        model_path=moderate_path,
        step="4/5",
        tqdm_label="C: 6.85% reranker",
    )

    run_reranker_condition(
        condition="D_reranker_13p7pct",
        short_label="D / 13.7% Full-Trap",
        model_path=fulltrap_path,
        step="5/5",
        tqdm_label="D: 13.7% Full-Trap",
    )

    del retriever
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[SUCCESS] Saved A/B/C/D context rows to:\n  {context_path}")
    return context_path


# ---------------------------------------------------------------------
# Qwen generation
# ---------------------------------------------------------------------

def generate(args: argparse.Namespace, benchmark: List[Dict[str, Any]]) -> Path:
    context_path = OUT_DIR / "abc_contexts.jsonl"
    generation_path = OUT_DIR / "abc_generations.jsonl"

    prepare_signature = checked_prepare_signature(args, benchmark)
    context_sha = sha256_file(context_path)
    init_stage_cache(generation_path,
                     generate_spec(args, benchmark, context_sha, prepare_signature),
                     "generate", args.overwrite)

    wanted_ids = {r["benchmark_id"] for r in benchmark}
    context_rows_all = read_jsonl(context_path)
    assert_unique_condition_rows(context_rows_all, "prepare")
    context_rows = [
        r for r in context_rows_all
        if r["benchmark_id"] in wanted_ids
        and r.get("condition") in CONDITIONS
    ]

    expected_keys = {
        (r["benchmark_id"], condition)
        for r in benchmark
        for condition in CONDITIONS
    }
    available_keys = {
        (r["benchmark_id"], r["condition"])
        for r in context_rows
    }
    missing_keys = sorted(expected_keys - available_keys)
    expected_count = len(expected_keys)

    if missing_keys:
        preview = "\n".join(
            f"  - {bid} / {condition}"
            for bid, condition in missing_keys[:12]
        )
        raise ValueError(
            f"Context cache is incomplete: missing {len(missing_keys)} A/B/C/D rows.\n"
            f"{preview}\n"
            "Run --mode prepare first (use --overwrite when switching from the old A/B/C design)."
        )

    existing_rows = read_jsonl(generation_path)
    assert_unique_condition_rows(existing_rows, "generate")
    existing = {(r["benchmark_id"], r["condition"]) for r in existing_rows}

    print("\n========== GENERATION ==========")
    print(f"Generator : {args.llm_model}")
    print(f"Rows      : {len(benchmark)} x 4 = {expected_count}")
    print("================================\n")

    generator = LanguageGenerator()

    # Stable A -> B -> C -> D ordering inside each benchmark row.
    by_id: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in context_rows:
        by_id[row["benchmark_id"]][row["condition"]] = row

    for b in tqdm(benchmark, desc="Qwen generation"):
        bid = b["benchmark_id"]

        for condition in CONDITIONS:
            key = (bid, condition)
            if key in existing:
                continue

            row = by_id[bid][condition]
            answer = generator.generate_answer(
                row["query"],
                row["contexts"],
                model_name=args.llm_model,
            )

            out = dict(row)
            out["generator"] = args.llm_model
            out["answer"] = answer

            append_jsonl(generation_path, out)
            existing.add(key)

    print(f"\n[SUCCESS] Saved generations to:\n  {generation_path}")
    return generation_path


# ---------------------------------------------------------------------
# NLI scorer
# ---------------------------------------------------------------------

REFUSAL_PATTERNS = [
    r"không có thông tin",
    r"không đủ thông tin",
    r"không tìm thấy thông tin",
    r"không thể trả lời",
    r"không thể xác định",
    r"không thể xác minh",
    r"không được cung cấp",
    r"không được đề cập",
    r"không đề cập",
    r"không nêu",
    r"not enough information",
    r"insufficient information",
    r"cannot determine",
    r"cannot verify",
]


def contains_refusal(text: str) -> bool:
    low = (text or "").lower()
    return any(re.search(p, low, flags=re.I) for p in REFUSAL_PATTERNS)


def normalize_for_match(text: str) -> str:
    """Normalize text conservatively for factual-value matching."""
    s = unicodedata.normalize("NFKC", str(text or "")).lower()
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r"\b0+(\d+)\b", r"\1", s)
    s = re.sub(r"\s*([/,:;()\-])\s*", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \t\r\n.,;:!?\"'")


def fact_value_present(text: str, value: str) -> bool:
    """Conservative Unicode-token-boundary factual value match.

    Prevent 2024 matching 20245, 6/10 matching 16/10, and place names
    being mistaken for substrings of larger word tokens. Equivalent number
    formats are still handled by normalize_for_match().
    """
    value_n = normalize_for_match(value)
    if not value_n:
        return False

    text_n = normalize_for_match(text)
    candidates = [value_n]
    for prefix in ("theo nguồn", "thông tin đúng là", "địa danh đúng là"):
        if value_n.startswith(prefix):
            tail = value_n[len(prefix):].lstrip(" ,:")
            if len(tail) >= 2:
                candidates.append(tail)

    return any(re.search(r"(?<!\w)" + re.escape(candidate) + r"(?!\w)",
                         text_n, flags=re.UNICODE) is not None
               for candidate in candidates)


def false_value_asserted(
    answer: str,
    false_value: str,
    correct_value: str = "",
) -> bool:
    """
    Return True only when the injected false value is positively asserted.

    These are treated as explicit REJECTIONS of the trap:
      - "không phải X mà là Y"
      - "từ X thành Y"
      - "đổi X thành Y"
      - "thay X bằng Y"
      - "sửa X thành Y"
      - a quoted false premise followed by "không chính xác"

    This avoids falsely counting a correct explanation that mentions the
    user's false value as "trap acceptance".
    """
    false_n = normalize_for_match(false_value)
    correct_n = normalize_for_match(correct_value)

    if not false_n:
        return False

    answer_n = normalize_for_match(answer)

    # Whole-answer replacement/correction patterns.
    # If the answer explicitly maps FALSE -> CORRECT, the false value is rejected.
    if correct_n:
        replacement_patterns = [
            rf"\btừ\s+[\"']?{re.escape(false_n)}[\"']?\s+(?:sang|thành|đến)\s+[\"']?{re.escape(correct_n)}",
            rf"\b(?:đổi|thay|sửa|chuyển)\s+[\"']?{re.escape(false_n)}[\"']?\s+(?:sang|thành|bằng|thành\s+địa\s+danh)\s+[\"']?{re.escape(correct_n)}",
            rf"\b{re.escape(false_n)}\s*(?:->|→|=>|⇒)\s*{re.escape(correct_n)}",
            rf"\bkhông\s+phải\s+[\"']?{re.escape(false_n)}[\"']?.{{0,80}}\b(?:mà\s+là|mà|đúng\s+là)\s+[\"']?{re.escape(correct_n)}",
        ]

        if any(re.search(p, answer_n, flags=re.I) for p in replacement_patterns):
            return False

    starts = [m.start() for m in re.finditer(
        r"(?<!\w)" + re.escape(false_n) + r"(?!\w)", answer_n)]
    if not starts:
        return False

    for pos in starts:
        left = answer_n[max(0, pos - 100):pos]
        right = answer_n[pos + len(false_n):pos + len(false_n) + 100]
        # Restrict negation to the local clause rather than any unrelated
        # "không" elsewhere in a long answer.
        left_clause = re.split(r"[.!?;\n]", left)[-1].strip()
        right_clause = re.split(r"[.!?;\n]", right)[0].strip()
        window = left_clause + " " + false_n + " " + right_clause

        rejected = bool(re.search(
            r"(?:không phải|chứ không phải|thay vì|not|rather than)\s*$",
            left_clause, flags=re.I,
        )) or bool(re.search(
            r"^\s*(?:là sai|sai|không đúng|không chính xác|is wrong|is incorrect)\b",
            right_clause, flags=re.I,
        ))

        # The false premise may be quoted/repeated and then explicitly rejected.
        if re.search(
            rf"{re.escape(false_n)}.{{0,100}}(?:không\s+chính\s+xác|không\s+đúng|là\s+sai)",
            window,
            flags=re.I,
        ):
            rejected = True

        # Local replacement pattern around this occurrence.
        if correct_n and re.search(
            rf"(?:từ|đổi|thay|sửa|chuyển).{{0,30}}"
            rf"{re.escape(false_n)}.{{0,30}}"
            rf"(?:sang|thành|bằng).{{0,30}}"
            rf"{re.escape(correct_n)}",
            window,
            flags=re.I,
        ):
            rejected = True

        if not rejected:
            return True

    return False


def score_correction_behavior(
    answer: str,
    benchmark_row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Score a single-fact correction using the benchmark's explicit trap fields.

    trap_original_value    = correct/gold fact
    trap_replacement_value = injected false fact
    """
    correct_value = str(benchmark_row.get("trap_original_value", "")).strip()
    false_value = str(benchmark_row.get("trap_replacement_value", "")).strip()

    correct_present = fact_value_present(answer, correct_value)
    false_present = fact_value_present(answer, false_value)
    false_asserted = false_value_asserted(answer, false_value, correct_value)
    refused = contains_refusal(answer)

    # Automatic proxy for correction success; not a substitute for validating
    # whether the corrected fact is asserted about the RIGHT subject/entity.
    success = bool(
        correct_value
        and correct_present
        and not false_asserted
        and not refused
    )

    return {
        "correct_value": correct_value,
        "false_value": false_value,
        "correct_value_present": bool(correct_present),
        "false_value_present": bool(false_present),
        "trap_accepted": bool(false_asserted),
        "correction_behavior_success": success,
    }


_ABBREVIATIONS = [
    "TP.", "Tp.", "tp.",
    "TS.", "ThS.", "PGS.", "GS.",
    "Q.", "P.",
]


def _protect_abbreviations(text: str) -> str:
    out = text
    for abbr in _ABBREVIATIONS:
        out = out.replace(abbr, abbr[:-1] + "<DOT>")
    return out


def _restore_abbreviations(text: str) -> str:
    return text.replace("<DOT>", ".")


META_CLAIM_PATTERNS = (
    r"^thông tin (?:trong|ở|trên) (?:câu|câu hỏi).*(?:không chính xác|không đúng|sai)",
    r"^câu (?:đúng|chính xác) (?:sẽ )?là",
    r"^vì vậy\b",
    r"^do đó\b",
    r"^theo nguồn[,:\s]*$",
    r"^không[.!]?$",
)


def clean_claim_for_faithfulness(claim: str) -> str:
    """
    Remove conversational correction scaffolding before source-grounding NLI.

    We KEEP the corrected factual proposition itself, but drop meta language
    such as "the sentence is wrong" or "replace X with Y".

    Example:
      'Vì vậy, câu đúng sẽ là: "Sự kiện diễn ra 01-13/10..."'
    becomes:
      'Sự kiện diễn ra 01-13/10...'
    """
    c = str(claim or "").strip(" \t\r\n-•\"“”")
    if not c:
        return ""

    # If a corrected factual sentence follows a meta prefix, keep the sentence.
    c = re.sub(
        r"^(?:vì\s+vậy|vậy|do\s+đó)[,:\s]*"
        r"(?:câu|thông\s+tin)\s+(?:đúng|chính\s+xác)\s+"
        r"(?:sẽ\s+)?là[:\s]*",
        "",
        c,
        flags=re.I,
    ).strip(" \t\r\n-•\"“”")

    c = re.sub(
        r"^(?:câu|thông\s+tin)\s+(?:đúng|chính\s+xác)\s+"
        r"(?:sẽ\s+)?là[:\s]*",
        "",
        c,
        flags=re.I,
    ).strip(" \t\r\n-•\"“”")

    low = c.lower().strip()

    # Pure meta-evaluation is not a factual claim from the source.
    if re.search(
        r"^thông tin (?:trong|ở|trên) (?:câu|câu hỏi).*"
        r"(?:không chính xác|không đúng|sai)",
        low,
        flags=re.I,
    ):
        return ""

    # Pure editing instructions are not source-grounded factual claims.
    if re.search(
        r"(?:cần|nên)\s+(?:được\s+)?(?:sửa|sửa đổi|thay|thay đổi)"
        r"|địa danh cần sửa"
        r"|sửa đổi là từ"
        r"|thay đổi thành",
        low,
        flags=re.I,
    ):
        return ""

    # "không phải X mà là Y" -> keep the positive Y proposition if present.
    m = re.search(r"\bkhông phải\b.+?\bmà là\b\s*(.+)$", c, flags=re.I)
    if m:
        c = m.group(1).strip()

    # Keep the positive proposition and remove the rejected-trap tail.
    for marker in (
        ", chứ không phải",
        " chứ không phải",
        ", không phải",
        "; không phải",
        " thay vì",
    ):
        idx = c.lower().find(marker)
        if idx > 0:
            c = c[:idx].strip()
            break

    c = re.sub(
        r"\s+(?:như|theo)\s+(?:trong\s+)?(?:câu hỏi|câu|thông tin)\s*"
        r"(?:ban đầu|đã nêu|đề cập)?\.?$",
        "",
        c,
        flags=re.I,
    ).strip()

    return c.strip(" \t\r\n-•\"“”")


def split_claims(answer: str) -> List[str]:
    text = re.sub(r"\r\n?", "\n", (answer or "").strip())
    text = re.sub(r"\n\s*[-*•]\s*", "\n", text)
    text = _protect_abbreviations(text)

    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    claims = []

    for p in parts:
        p = _restore_abbreviations(p)
        p = re.sub(r"^\s*\d+[\).\:-]?\s*", "", p).strip(" \t-•")

        if len(p) < 8 or contains_refusal(p):
            continue

        cleaned = clean_claim_for_faithfulness(p)
        if len(cleaned) < 8:
            continue

        claims.append(cleaned)

    return claims


class NLIScorer:
    def __init__(self, model_name: str, device: str, threshold: float):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for NLI but CUDA is unavailable.")

        self.device = torch.device(device)
        self.threshold = float(threshold)
        self.model_name = model_name

        print(f"\nLoading NLI scorer: {model_name}")
        print(f"NLI device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

        self.entail_idx = None

        for idx, label in (self.model.config.id2label or {}).items():
            if "entail" in str(label).lower():
                self.entail_idx = int(idx)
                break

        if (
            self.entail_idx is None
            and "moritzlaurer/mdeberta-v3-base-mnli-xnli" in model_name.lower()
        ):
            self.entail_idx = 0

        if self.entail_idx is None:
            raise RuntimeError(
                f"Cannot identify entailment label from {self.model.config.id2label}"
            )

    def windows(self, text: str, window: int = 360, stride: int = 300) -> List[str]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)

        if len(ids) <= window:
            return [text]

        out = []
        start = 0

        while start < len(ids):
            end = min(start + window, len(ids))
            out.append(
                self.tokenizer.decode(
                    ids[start:end],
                    skip_special_tokens=True,
                )
            )

            if end == len(ids):
                break

            start += stride

        return out

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

    def reference_correctness(self, answer: str, reference: str) -> Dict[str, float]:
        """
        Bidirectional NLI diagnostic.

        For normal ANSWER questions, a concise answer can be semantically correct
        even when it does not entail every detail in a longer canonical reference.
        """
        a_to_r = self.entail_prob(answer, reference)
        r_to_a = self.entail_prob(reference, answer)

        return {
            "answer_to_reference": a_to_r,
            "reference_to_answer": r_to_a,
            "max_bidirectional": max(a_to_r, r_to_a),
        }

    def faithfulness(
        self,
        answer: str,
        contexts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        claims = split_claims(answer)

        context_texts = [
            str(c.get("text", "")).strip()
            for c in contexts
            if str(c.get("text", "")).strip()
        ]

        context_windows = []
        for text in context_texts:
            context_windows.extend(self.windows(text))

        normalized_contexts = [
            normalize_for_match(t)
            for t in context_texts
        ]

        details = []

        for claim in claims:
            claim_n = normalize_for_match(claim)

            exact_support = bool(
                claim_n
                and any(claim_n in ctx_n for ctx_n in normalized_contexts)
            )

            best = max(
                (
                    self.entail_prob(window, claim)
                    for window in context_windows
                ),
                default=0.0,
            )

            supported = bool(
                exact_support
                or best >= self.threshold
            )

            details.append({
                "claim": claim,
                "entailment_probability": best,
                "exact_context_support": exact_support,
                "supported": supported,
                "support_method": (
                    "exact_context_match"
                    if exact_support
                    else ("nli" if best >= self.threshold else "unsupported")
                ),
            })

        if not details:
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


def corrected_query_statement(benchmark_row: Dict[str, Any]) -> str:
    """
    Build the positive corrected proposition from a single-fact trap query.

    Example:
      query:  "Lăng Hoàng Gia ở xã Hà Nội..."
      false:  "Hà Nội"
      gold:   "Long Hưng"
    ->        "Lăng Hoàng Gia ở xã Long Hưng..."
    """
    query = str(benchmark_row.get("query", ""))
    false_value = str(benchmark_row.get("trap_replacement_value", "")).strip()
    correct_value = str(benchmark_row.get("trap_original_value", "")).strip()

    # Prefer the quoted premise.
    m = re.search(r"[“\"](.+?)[”\"]", query)
    statement = m.group(1).strip() if m else query

    if false_value and correct_value:
        statement = re.sub(
            re.escape(false_value),
            correct_value,
            statement,
            count=1,
            flags=re.I,
        )

    # Remove common wrapper text if quotation extraction failed.
    statement = re.sub(
        r"^\s*thông tin sau có đúng không[:\s]*",
        "",
        statement,
        flags=re.I,
    )
    statement = re.sub(
        r"\?\s*nếu sai.*$",
        "",
        statement,
        flags=re.I,
    )

    return statement.strip(" \t\r\n-•\"“”?")


def evidence_support_top_n(
    scorer: NLIScorer,
    contexts: List[Dict[str, Any]],
    benchmark_row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evidence-support diagnostic that does NOT require the exact gold chunk ID.

    For CORRECT queries:
      - primary evidence signal: the selected context contains the correct
        single-fact value;
      - stronger diagnostic: context NLI entails the corrected proposition.

    For ANSWER queries:
      - context NLI entails the canonical reference answer.

    This is reported alongside exact_gold_chunk_hit, not as a replacement for it.
    """
    behavior = str(benchmark_row.get("target_behavior", "")).strip().lower()

    if behavior == "correct":
        correct_value = str(benchmark_row.get("trap_original_value", "")).strip()
        hypothesis = corrected_query_statement(benchmark_row)
    else:
        correct_value = ""
        hypothesis = str(benchmark_row.get("reference_answer", "")).strip()

    details = []

    for c in contexts:
        text = str(c.get("text", "")).strip()
        if not text:
            continue

        value_present = bool(
            behavior == "correct"
            and correct_value
            and fact_value_present(text, correct_value)
        )

        nli_prob = (
            scorer.entail_prob(text, hypothesis)
            if hypothesis
            else 0.0
        )

        supported = bool(
            value_present
            or nli_prob >= scorer.threshold
        )

        details.append({
            "chunk_id": str(c.get("chunk_id", "")),
            "correct_value_present": value_present,
            "evidence_nli_probability": nli_prob,
            "supported": supported,
        })

    return {
        "evidence_support_at_top_n": any(d["supported"] for d in details),
        "evidence_support_details": details,
        "corrected_evidence_hypothesis": hypothesis,
    }


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def bootstrap_ci(
    values: List[float],
    seed: int = DEFAULT_SEED,
    n_boot: int = 2000,
) -> Optional[List[float]]:
    if not values:
        return None

    if len(values) == 1:
        return [values[0], values[0]]

    rng = random.Random(seed)
    n = len(values)
    means = []

    for _ in range(n_boot):
        means.append(
            statistics.mean(
                values[rng.randrange(n)]
                for _ in range(n)
            )
        )

    means.sort()

    return [
        means[int(0.025 * (n_boot - 1))],
        means[int(0.975 * (n_boot - 1))],
    ]


def condition_summary(rows: List[Dict[str, Any]], condition: str, seed: int) -> Dict[str, Any]:
    sub = [r for r in rows if r["condition"] == condition]

    success = [int(bool(r["behavior_success"])) for r in sub]
    faith = [
        float(r["context_faithfulness"])
        for r in sub
        if r.get("context_faithfulness") is not None
    ]

    total_claims = sum(int(r.get("num_factual_claims", 0)) for r in sub)
    supported_claims = sum(int(r.get("supported_factual_claims", 0)) for r in sub)

    answer_sub = [r for r in sub if r["target_behavior"] == "answer"]
    correct_sub = [r for r in sub if r["target_behavior"] == "correct"]

    def success_rate(xs: List[Dict[str, Any]]) -> Optional[float]:
        if not xs:
            return None
        return sum(int(bool(x["behavior_success"])) for x in xs) / len(xs)

    trap_breakdown = {}

    for trap in sorted(set(str(r.get("trap_type", "")) for r in correct_sub)):
        if not trap:
            continue
        xs = [r for r in correct_sub if str(r.get("trap_type", "")) == trap]
        trap_breakdown[trap] = {
            "n": len(xs),
            "correction_success_rate": success_rate(xs),
        }

    return {
        "n_queries": len(sub),
        "answers": len(sub),  # backward-compatible alias

        "rrf_candidate_recall_at_k": (
            sum(int(bool(r["gold_in_rrf_top_k"])) for r in sub) / len(sub)
            if sub else None
        ),

        "gold_hit_at_top_n": (
            sum(int(bool(r["gold_in_selected_top_n"])) for r in sub) / len(sub)
            if sub else None
        ),

        "exact_gold_chunk_hit_at_top_n": (
            sum(int(bool(r["gold_in_selected_top_n"])) for r in sub) / len(sub)
            if sub else None
        ),

        "evidence_support_at_top_n": (
            sum(int(bool(r.get("evidence_support_at_top_n", False))) for r in sub) / len(sub)
            if sub else None
        ),

        "behavior_success_rate": statistics.mean(success) if success else None,
        "behavior_success_95ci": bootstrap_ci(success, seed),

        "answer_success_rate": success_rate(answer_sub),
        "correction_success_rate": success_rate(correct_sub),

        "trap_acceptance_rate": (
            sum(int(bool(r.get("trap_accepted", False))) for r in correct_sub) / len(correct_sub)
            if correct_sub else None
        ),

        "correct_value_presence_rate": (
            sum(int(bool(r.get("correct_value_present", False))) for r in correct_sub) / len(correct_sub)
            if correct_sub else None
        ),

        "mean_reference_entailment": (
            statistics.mean(float(r["reference_correctness"]) for r in sub)
            if sub else None
        ),

        "faithfulness_scored_answers": len(faith),
        "mean_context_faithfulness": statistics.mean(faith) if faith else None,
        "context_faithfulness_95ci": bootstrap_ci(faith, seed),

        "micro_context_faithfulness": (
            supported_claims / total_claims
            if total_claims else None
        ),

        "context_hallucination_rate": (
            1.0 - supported_claims / total_claims
            if total_claims else None
        ),

        "refusal_rate": (
            sum(int(bool(r["contains_refusal"])) for r in sub) / len(sub)
            if sub else None
        ),

        "total_factual_claims": total_claims,
        "supported_factual_claims": supported_claims,
        "trap_breakdown": trap_breakdown,
    }


def paired_binary_test(
    rows: List[Dict[str, Any]],
    condition_x: str,
    condition_y: str,
    field: str,
) -> Dict[str, Any]:
    x = {
        r["benchmark_id"]: bool(r[field])
        for r in rows
        if r["condition"] == condition_x
    }
    y = {
        r["benchmark_id"]: bool(r[field])
        for r in rows
        if r["condition"] == condition_y
    }

    common = sorted(set(x) & set(y))

    x0_y1 = sum((not x[i]) and y[i] for i in common)
    x1_y0 = sum(x[i] and (not y[i]) for i in common)

    out = {
        "paired_n": len(common),
        "x_fail_y_success": int(x0_y1),
        "x_success_y_fail": int(x1_y0),
        "delta_y_minus_x": (
            statistics.mean(
                int(y[i]) - int(x[i])
                for i in common
            )
            if common else None
        ),
    }

    try:
        from statsmodels.stats.contingency_tables import mcnemar

        table = [
            [0, x0_y1],
            [x1_y0, 0],
        ]
        test = mcnemar(table, exact=True)
        out["mcnemar_p_value"] = float(test.pvalue)

    except Exception as exc:
        out["mcnemar_note"] = f"Not computed: {exc}"

    return out


def paired_continuous(
    rows: List[Dict[str, Any]],
    condition_x: str,
    condition_y: str,
    field: str,
    seed: int,
) -> Dict[str, Any]:
    x = {
        r["benchmark_id"]: float(r[field])
        for r in rows
        if r["condition"] == condition_x
        and r.get(field) is not None
    }
    y = {
        r["benchmark_id"]: float(r[field])
        for r in rows
        if r["condition"] == condition_y
        and r.get(field) is not None
    }

    common = sorted(set(x) & set(y))
    diffs = [y[i] - x[i] for i in common]

    out = {
        "paired_n": len(common),
        "mean_delta_y_minus_x": statistics.mean(diffs) if diffs else None,
        "delta_95ci": bootstrap_ci(diffs, seed),
    }

    if diffs:
        try:
            from scipy.stats import wilcoxon

            xv = [x[i] for i in common]
            yv = [y[i] for i in common]

            if any(abs(a - b) > 1e-12 for a, b in zip(xv, yv)):
                stat, pval = wilcoxon(yv, xv)
                out["wilcoxon_statistic"] = float(stat)
                out["wilcoxon_p_value"] = float(pval)
            else:
                out["wilcoxon_statistic"] = 0.0
                out["wilcoxon_p_value"] = 1.0

        except Exception as exc:
            out["wilcoxon_note"] = f"Not computed: {exc}"

    return out


def correct_only_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        r for r in rows
        if str(r.get("target_behavior", "")).strip().lower() == "correct"
    ]


def comparison_summary(
    rows: List[Dict[str, Any]],
    condition_x: str,
    condition_y: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Paired comparison bundle for two conditions on the exact same query subset.
    """
    correct_rows = correct_only_rows(rows)

    return {
        "behavior_success": paired_binary_test(
            rows,
            condition_x,
            condition_y,
            "behavior_success",
        ),
        "exact_gold_chunk_hit_at_top_n": paired_binary_test(
            rows,
            condition_x,
            condition_y,
            "gold_in_selected_top_n",
        ),
        "evidence_support_at_top_n": paired_binary_test(
            rows,
            condition_x,
            condition_y,
            "evidence_support_at_top_n",
        ),
        "correction_success": paired_binary_test(
            correct_rows,
            condition_x,
            condition_y,
            "behavior_success",
        ),
        "trap_acceptance": paired_binary_test(
            [
                r for r in correct_rows
                if r.get("trap_accepted") is not None
            ],
            condition_x,
            condition_y,
            "trap_accepted",
        ),
        "context_faithfulness": paired_continuous(
            rows,
            condition_x,
            condition_y,
            "context_faithfulness",
            seed,
        ),
    }


def condition_block(
    rows: List[Dict[str, Any]],
    seed: int,
) -> Dict[str, Any]:
    return {
        condition: condition_summary(rows, condition, seed)
        for condition in CONDITIONS
    }


def pairwise_reranker_block(
    rows: List[Dict[str, Any]],
    seed: int,
    include_overall_system: bool = False,
) -> Dict[str, Any]:
    """
    Core ablation comparisons:
      B vs C : 0% vs 6.85%
      B vs D : 0% vs 13.7%
      C vs D : 6.85% vs 13.7%

    A vs D is additionally reported for the overall RAG system when requested.
    """
    out = {
        "B_0pct_vs_C_6p85pct": comparison_summary(
            rows,
            "B_reranker_0pct",
            "C_reranker_6p85pct",
            seed,
        ),
        "B_0pct_vs_D_13p7pct": comparison_summary(
            rows,
            "B_reranker_0pct",
            "D_reranker_13p7pct",
            seed,
        ),
        "C_6p85pct_vs_D_13p7pct": comparison_summary(
            rows,
            "C_reranker_6p85pct",
            "D_reranker_13p7pct",
            seed,
        ),
    }

    if include_overall_system:
        out["A_RRF_vs_D_13p7pct"] = comparison_summary(
            rows,
            "A_rrf_only",
            "D_reranker_13p7pct",
            seed,
        )

    return out


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------

def score(args: argparse.Namespace, benchmark: List[Dict[str, Any]]) -> Tuple[Path, Path]:
    generation_path = OUT_DIR / "abc_generations.jsonl"
    scored_path = OUT_DIR / "abc_scored.jsonl"
    summary_path = OUT_DIR / "abc_summary.json"

    generation_signature = checked_generation_signature(args, benchmark)
    generation_sha = sha256_file(generation_path)
    wanted_ids = {r["benchmark_id"] for r in benchmark}
    generation_rows_all = read_jsonl(generation_path)
    assert_unique_condition_rows(generation_rows_all, "generate")
    generation_rows = [
        r for r in generation_rows_all
        if r["benchmark_id"] in wanted_ids
        and r.get("condition") in CONDITIONS
    ]

    expected_keys = {
        (r["benchmark_id"], condition)
        for r in benchmark
        for condition in CONDITIONS
    }
    available_keys = {
        (r["benchmark_id"], r["condition"])
        for r in generation_rows
    }
    missing_keys = sorted(expected_keys - available_keys)
    expected_count = len(expected_keys)

    if missing_keys:
        preview = "\n".join(
            f"  - {bid} / {condition}"
            for bid, condition in missing_keys[:12]
        )
        raise ValueError(
            f"Generation file is incomplete: missing {len(missing_keys)} A/B/C/D rows.\n"
            f"{preview}\n"
            "Run --mode generate after preparing the four-condition contexts."
        )

    score_spec = {
        "version": CACHE_SCHEMA_VERSION, "stage": "score",
        "benchmark": benchmark_signature(args, benchmark),
        "generation_sha256": generation_sha,
        "generation_signature": generation_signature,
        "nli_model": args.nli_model,
        "nli_device": args.nli_device,
        "threshold": args.threshold,
    }
    init_stage_cache(scored_path, score_spec, "score", args.overwrite)

    previous_rows = read_jsonl(scored_path)
    assert_unique_condition_rows(previous_rows, "score")
    previous = {(r["benchmark_id"], r["condition"]): r for r in previous_rows}

    scorer = NLIScorer(
        model_name=args.nli_model,
        device=args.nli_device,
        threshold=args.threshold,
    )

    benchmark_map = {
        str(r["benchmark_id"]): r
        for r in benchmark
    }

    final_rows = []

    for row in tqdm(generation_rows, desc="NLI scoring"):
        key = (row["benchmark_id"], row["condition"])

        if key in previous and not args.overwrite:
            final_rows.append(previous[key])
            continue

        benchmark_row = benchmark_map[str(row["benchmark_id"])]
        target_behavior = str(
            benchmark_row["target_behavior"]
        ).strip().lower()

        reference_nli = scorer.reference_correctness(
            row["answer"],
            benchmark_row["reference_answer"],
        )

        faith = scorer.faithfulness(
            row["answer"],
            row["contexts"],
        )

        evidence_support = evidence_support_top_n(
            scorer,
            row["contexts"],
            benchmark_row,
        )

        refused = contains_refusal(row["answer"])

        correction_diag = {
            "correct_value": "",
            "false_value": "",
            "correct_value_present": None,
            "false_value_present": None,
            "trap_accepted": None,
            "correction_behavior_success": None,
        }

        if target_behavior == "correct":
            correction_diag = score_correction_behavior(
                row["answer"],
                benchmark_row,
            )
            behavior_success = bool(
                correction_diag["correction_behavior_success"]
            )
            behavior_scoring_method = "single_fact_correction_rule"
        else:
            behavior_success = bool(
                reference_nli["max_bidirectional"] >= args.threshold
                and not refused
            )
            behavior_scoring_method = "bidirectional_nli"

        result = dict(row)

        # Benchmark JSON is the annotation source of truth even when scoring
        # old generation rows that do not yet contain trap fields.
        result["target_behavior"] = target_behavior
        result["reference_answer"] = benchmark_row["reference_answer"]
        result["trap_type"] = benchmark_row.get("trap_type", "")
        result["trap_original_value"] = benchmark_row.get("trap_original_value", "")
        result["trap_replacement_value"] = benchmark_row.get("trap_replacement_value", "")

        result.update(faith)
        result.update(evidence_support)
        result.update(correction_diag)
        result.update({
            "reference_entailment_answer_to_reference": reference_nli["answer_to_reference"],
            "reference_entailment_reference_to_answer": reference_nli["reference_to_answer"],
            "reference_correctness": reference_nli["max_bidirectional"],
            "behavior_success": bool(behavior_success),
            "behavior_scoring_method": behavior_scoring_method,
            "contains_refusal": refused,
            "nli_model": args.nli_model,
            "entailment_threshold": args.threshold,
        })

        append_jsonl(scored_path, result)
        final_rows.append(result)

    # -------------------------------------------------------------
    # Two complementary evaluation views
    # -------------------------------------------------------------
    # 1) Full benchmark:
    #    end-to-end RAG utility over every query.
    #
    # 2) Candidate-conditioned reranker subset:
    #    only queries whose annotated gold chunk was already present in
    #    the common RRF Top-K candidate pool. This isolates reranking
    #    quality from first-stage retrieval failure and is determined
    #    BEFORE seeing B/C/D reranker outputs.
    # -------------------------------------------------------------
    a_rows = [
        r for r in final_rows
        if r["condition"] == "A_rrf_only"
    ]

    eligible_ids = {
        str(r["benchmark_id"])
        for r in a_rows
        if bool(r.get("gold_in_rrf_top_k"))
    }

    conditioned_rows = [
        r for r in final_rows
        if str(r["benchmark_id"]) in eligible_ids
    ]

    full_conditions = condition_block(
        final_rows,
        args.seed,
    )
    conditioned_conditions = condition_block(
        conditioned_rows,
        args.seed,
    )

    summary = {
        "full_benchmark": {
            "definition": (
                "All benchmark queries. Measures end-to-end RAG behavior, "
                "including first-stage retrieval failures."
            ),
            "n_queries": len(benchmark),
            "conditions": full_conditions,
            "pairwise": pairwise_reranker_block(
                final_rows,
                args.seed,
                include_overall_system=True,
            ),
        },

        "candidate_conditioned": {
            "definition": (
                "Pre-reranking subset where the annotated gold_chunk_id is "
                f"already present in the common RRF Top-{args.top_k}. "
                "This subset is fixed before B/C/D reranking and is used "
                "to isolate reranker quality from candidate-retrieval failure."
            ),
            "eligibility_rule": "gold_in_rrf_top_k == true",
            "n_queries": len(eligible_ids),
            "eligible_fraction": (
                len(eligible_ids) / len(benchmark)
                if benchmark else None
            ),
            "eligible_benchmark_ids": sorted(eligible_ids),
            "conditions": conditioned_conditions,
            "pairwise": pairwise_reranker_block(
                conditioned_rows,
                args.seed,
                include_overall_system=False,
            ),
        },

        "benchmark": {
            "n_queries": len(benchmark),
            "answer": sum(
                r["target_behavior"] == "answer"
                for r in benchmark
            ),
            "correct": sum(
                r["target_behavior"] == "correct"
                for r in benchmark
            ),
            "trap_distribution": dict(
                Counter(
                    str(r.get("trap_type", ""))
                    for r in benchmark
                    if r["target_behavior"] == "correct"
                )
            ),
            "allow_draft": bool(args.allow_draft),
            "benchmark_file": str(Path(args.benchmark).resolve()),
        },

        "metadata": {
            "generator": args.llm_model,
            "nli_model": args.nli_model,
            "entailment_threshold": args.threshold,
            "top_k_retrieval": args.top_k,
            "top_n_context": args.top_n,
            "seed": args.seed,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "benchmark_sha256": sha256_file(Path(args.benchmark)),
            "generation_sha256": generation_sha,
            "generator_randomness": "Ollama generation is not guaranteed deterministic by this script.",
            "conditions": CONDITION_LABELS,
            "primary_ablation_comparisons": [
                "B_0pct_vs_C_6p85pct",
                "B_0pct_vs_D_13p7pct",
                "C_6p85pct_vs_D_13p7pct",
            ],
            "candidate_conditioning": (
                "Uses only pre-reranking RRF candidate availability "
                "(gold_in_rrf_top_k); never filters by which reranker wins."
            ),
            "behavior_scoring": {
                "answer": (
                    "bidirectional NLI: max(answer->reference, "
                    "reference->answer) >= threshold"
                ),
                "correct": (
                    "trap_original_value present AND "
                    "trap_replacement_value not positively asserted (heuristic)"
                ),
            },
            "faithfulness_scoring": (
                "clean factual claims; remove correction scaffolding; "
                "support by exact context match OR NLI entailment"
            ),
            "retrieval_metrics": {
                "exact_gold_chunk_hit_at_top_n": (
                    "strict: selected Top-N must contain the annotated gold_chunk_id"
                ),
                "evidence_support_at_top_n": (
                    "lenient evidence diagnostic: another selected chunk may count "
                    "if it contains the correct trap value or entails the canonical answer"
                ),
            },
        },
    }


    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n================ A/B/C/D SUMMARY ================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("================================================")
    print(f"\nSaved scored rows: {scored_path}")
    print(f"Saved summary    : {summary_path}")

    return scored_path, summary_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_path = Path(args.benchmark).expanduser().resolve()
    args.benchmark = str(benchmark_path)

    benchmark = load_benchmark(
        benchmark_path,
        max_queries=args.max_queries,
        allow_draft=args.allow_draft,
    )

    print("\n========== BENCHMARK ==========")
    print(f"File   : {benchmark_path}")
    print(f"Rows   : {len(benchmark)}")
    print(
        "Labels : "
        f"answer={sum(r['target_behavior']=='answer' for r in benchmark)}, "
        f"correct={sum(r['target_behavior']=='correct' for r in benchmark)}"
    )
    print("===============================\n")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.allow_draft:
        print(
            "[WARNING] --allow_draft is ON. "
            "This run is for technical validation only and must not be reported as a final paper result.\n"
        )

    if args.mode == "prepare":
        prepare(args, benchmark)

    elif args.mode == "generate":
        generate(args, benchmark)

    elif args.mode == "score":
        score(args, benchmark)

    elif args.mode == "all":
        prepare(args, benchmark)
        generate(args, benchmark)
        score(args, benchmark)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] Interrupted by user.")
        raise
    except Exception as exc:
        print("\n================ FATAL ERROR ================")
        print(f"{type(exc).__name__}: {exc}")
        print("=============================================")
        raise
