#!/usr/bin/env python3
"""Read-only, checkpoint-specific diagnostic of tourism rerankers B/C/D.

The metric compares the given TEST positive and negative within each fixed pair.
It does NOT measure RRF candidate retrieval, multi-candidate reranking, or RAG quality.
No fine-tuning or checkpoint modification occurs.

Run from repository root:
  python diagnose_pairwise_checkpoints.py --check-only --seeds 42 123 999
  python diagnose_pairwise_checkpoints.py --seeds 42 --batch-size 1 --device auto
  python diagnose_pairwise_checkpoints.py --seeds 42 123 999 --batch-size 1

Training used BF16 CUDA autocast and fixed padding to max_length=512.
To compare as closely as feasible with the original TEST evaluation, pass
  --amp bf16 --padding max_length
ONLY if the current CUDA GPU supports BF16. The default fp32 is a separate
numerical diagnostic and may differ for borderline score gaps.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

REGIMES = {
    "B": ("run_1", "moc_0"),
    "C": ("run_2", "moc_6"),
    "D": ("run_3", "moc_13"),
}
SCRIPT_VERSION = "checkpoint-pairwise-diagnostic-v2.1"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", type=Path, default=Path.cwd(),
                   help="Repository root; defaults to current working directory")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--batch-size", type=int, default=1,
                   help="Number of TEST pairs per model forward pass; use 1 on 4GB GPUs")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--padding", choices=["max_length", "dynamic"], default="max_length")
    p.add_argument("--amp", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="Diagnostic pilot: first N TEST rows; 0 means full TEST")
    p.add_argument("--hash-weights", action="store_true",
                   help="Also SHA256-hash every model.safetensors (slow on large checkpoints)")
    p.add_argument("--check-only", action="store_true",
                   help="Validate TEST and checkpoint file layout without model inference")
    p.add_argument("--self-test", action="store_true",
                   help="Run pure-Python metric/TEST unit checks; requires no torch")
    return p.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_metrics(differences: list[float]) -> dict[str, Any]:
    """Strict positive > negative; ties count as incorrect/rank=2."""
    if not differences:
        raise ValueError("No scored pairs")
    if any(not math.isfinite(v) for v in differences):
        raise ValueError("Non-finite score difference")
    hits = sum(diff > 0 for diff in differences)
    ties = sum(diff == 0 for diff in differences)
    n = len(differences)
    accuracy = hits / n
    # Exactly the pairwise ranking semantics from the training scripts.
    return {
        "n": n,
        "pairwise_accuracy": accuracy,
        "precision_at_1": accuracy,
        "mrr": (hits + 0.5 * (n - hits)) / n,
        "ndcg_at_2": (hits + (n - hits) / math.log2(3)) / n,
        "correct": hits,
        "incorrect": n - hits,
        "ties": ties,
    }


def read_test(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"TEST file not found: {path}")
    sha = sha256_file(path)
    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list) or not rows:
        raise ValueError("TEST must be a nonempty JSON list")
    mandatory = ("query", "positive", "negative")
    empty_fields = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"TEST sample {i} is not an object")
        for key in mandatory:
            value = row.get(key)
            # Match the training loader: the pair tokenizer accepts empty strings.
            # Do NOT silently drop such rows, or the 1136-row historical TEST
            # scores/correctness masks will cease to be comparable.
            if not isinstance(value, str):
                raise ValueError(
                    f"TEST sample {i} has non-string or missing field {key!r}: "
                    f"type={type(value).__name__}. Inspect TEST before evaluation."
                )
            if not value.strip():
                empty_fields.append((i, key))
        if "margin" in row and not math.isfinite(float(row["margin"])):
            raise ValueError(f"TEST sample {i} has invalid margin")
    if empty_fields:
        print(
            f"[DATA QUALITY WARNING] {len(empty_fields)} empty/whitespace-only "
            "TEST text fields; preserved to reproduce original training TEST scoring."
        )
        print(f"[DATA QUALITY WARNING] First affected (0-based index, field): {empty_fields[:20]}")
        print("[NOTE] Empty passage pairs should be reported separately as a TEST-quality limitation.")
    return rows, sha


def checkpoint_path(root: Path, condition: str, seed: int) -> Path:
    run, tag = REGIMES[condition]
    path = root / "models" / run / f"vietnam_tourism_reranker_{tag}_seed_{seed}"
    for filename in ("config.json", "model.safetensors"):
        if not (path / filename).is_file():
            raise FileNotFoundError(f"Missing checkpoint component: {path / filename}")
    if not any((path / name).is_file() for name in ("tokenizer.json", "sentencepiece.bpe.model", "spiece.model")):
        raise FileNotFoundError(f"Missing local tokenizer files in {path}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "xlm-roberta" or len(config.get("id2label", {})) != 1:
        raise ValueError(f"Unexpected checkpoint architecture/label count: {path}")
    if (path / "model.safetensors").stat().st_size == 0:
        raise ValueError(f"Checkpoint weight file is empty: {path}")
    return path


def checkpoint_identity(path: Path, hash_weights: bool = False) -> dict[str, Any]:
    weights = path / "model.safetensors"
    st = weights.stat()
    result = {
        "path": str(path.resolve()),
        "config_sha256": sha256_file(path / "config.json"),
        "model_size_bytes": st.st_size,
        "model_mtime_ns": st.st_mtime_ns,
    }
    if hash_weights:
        result["model_sha256"] = sha256_file(weights)
    return result


def original_mask(root: Path, condition: str, seed: int, expected_size: int, limit: int):
    tag = REGIMES[condition][1].split("_")[-1]
    path = root / "data" / "processed" / f"test_correct_mask_moc_{tag}_seed_{seed}.json"
    if not path.is_file():
        return None, {"status": "not_found", "path": str(path)}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list) or len(obj) != expected_size or any(x not in (0, 1, False, True) for x in obj):
        return None, {"status": "invalid_length_or_values", "path": str(path), "length": len(obj) if isinstance(obj, list) else None}
    return [int(x) for x in (obj[:limit] if limit else obj)], {"status": "found", "path": str(path), "sha256": sha256_file(path)}


def predict(rows, checkpoint: Path, device, batch_size: int, max_length: int,
            padding: str, amp: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint), local_files_only=True
    ).to(device).eval()
    if getattr(model.config, "num_labels", None) != 1:
        raise RuntimeError(f"Model is not a scalar-score reranker: {checkpoint}")
    all_scores = []
    try:
        for begin in range(0, len(rows), batch_size):
            group = rows[begin:begin + batch_size]
            # Preserve the training order (query, passage) and TEST row order.
            sequences = ([ (r["query"], r["positive"]) for r in group] +
                         [ (r["query"], r["negative"]) for r in group])
            encoded = tokenizer(
                [p[0] for p in sequences], [p[1] for p in sequences],
                padding=("max_length" if padding == "max_length" else True),
                truncation=True, max_length=max_length, return_tensors="pt")
            # Training passed exactly input_ids and attention_mask, not other keys.
            inputs = {key: encoded[key].to(device) for key in ("input_ids", "attention_mask")}
            cast_context = (torch.autocast(device_type="cuda", dtype={
                "bf16": torch.bfloat16, "fp16": torch.float16}[amp])
                if amp != "fp32" else contextlib.nullcontext())
            with torch.inference_mode(), cast_context:
                logits = model(**inputs).logits
            if logits.ndim != 2 or logits.shape != (len(sequences), 1):
                raise RuntimeError(f"Expected (batch,1) logits; got {tuple(logits.shape)}")
            values = logits[:, 0].float().cpu().tolist()
            count = len(group)
            all_scores.extend(zip(values[:count], values[count:]))
            processed = begin + count
            if processed == len(rows) or processed % 200 < batch_size:
                print(f"  {processed}/{len(rows)} pairs scored", flush=True)
    finally:
        del model, tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return all_scores


def save_atomic_json(path: Path, value: dict):
    """Replace output only after complete serialization."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".pairwise_tmp_", suffix=".json", delete=False) as f:
        temp = Path(f.name)
        json.dump(value, f, ensure_ascii=False, indent=2)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_atomic_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("No CSV diagnostic records")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=path.parent,
                                     prefix=".pairwise_tmp_", suffix=".csv", delete=False) as f:
        temp = Path(f.name)
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def self_test():
    eps = 1.0 / math.log2(3)
    assert pair_metrics([1, 0, -1])["pairwise_accuracy"] == 1 / 3
    assert abs(pair_metrics([1, 0, -1])["mrr"] - 2 / 3) < 1e-12
    assert abs(pair_metrics([1, 0, -1])["ndcg_at_2"] - (1 + 2 * eps) / 3) < 1e-12
    assert pair_metrics([0, 0])["ties"] == 2
    assert pair_metrics([-1])["mrr"] == 0.5
    assert pair_metrics([1])["ndcg_at_2"] == 1.0
    # Regression: original training tokenized empty-string passages rather than
    # dropping them. The diagnostic must preserve their row positions.
    with tempfile.TemporaryDirectory() as td:
        dataset_file = Path(td) / "test.json"
        dataset_file.write_text(json.dumps([
            {"query": "Example", "positive": "Evidence", "negative": "", "margin": 1.0},
        ]), encoding="utf-8")
        checked, _ = read_test(dataset_file)
        assert len(checked) == 1 and checked[0]["negative"] == ""
        dataset_file.write_text(json.dumps([
            {"query": "Example", "positive": "Evidence", "negative": None},
        ]), encoding="utf-8")
        try:
            read_test(dataset_file)
        except ValueError:
            pass
        else:
            raise AssertionError("Non-string negative should fail validation")
    print("[PASS] Pairwise metrics and original-loader-compatible TEST validation")


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return
    if args.batch_size <= 0 or args.max_length < 8 or args.limit < 0:
        raise ValueError("Require batch-size >0, max-length >=8, limit >=0")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError("Seeds must be unique nonnegative integers")

    root = args.project.expanduser().resolve()
    all_test, test_sha = read_test(root / "data" / "processed" / "cross_encoder_TEST.json")
    if len(all_test) != 1136:
        print(f"[WARNING] Expected 1136 TEST pairs, received {len(all_test)}")
    if args.limit and args.limit > len(all_test):
        raise ValueError("--limit exceeds TEST size; use --limit 0 for all rows")
    rows = all_test[:args.limit] if args.limit else all_test
    query_counts = Counter(r["query"] for r in all_test)
    doc_counts = Counter(str(r.get("doc_id", "")) for r in all_test)
    print(f"TEST: {len(all_test)} rows | used: {len(rows)} | "
          f"distinct queries: {len(query_counts)} | distinct doc_ids: {len(doc_counts)}")
    print(f"TEST SHA256: {test_sha}")
    if args.limit:
        print("[WARNING] PILOT ONLY: metrics are from a subset, not the full held-out TEST")
    if len(query_counts) < len(all_test):
        print("[NOTE] TEST contains repeated queries: pairwise rows are not independent questions")

    checkpoints = {(regime, seed): checkpoint_path(root, regime, seed)
                   for seed in args.seeds for regime in REGIMES}
    for (condition, seed), path in checkpoints.items():
        print(f"[{condition} seed={seed}] {path}")
    if args.check_only:
        print("[PASS] TEST schema and requested checkpoint config/tokenizer/weights layout")
        print("[NOTE] File existence is not proof that checkpoint weights produce expected scores")
        return

    import torch
    from transformers import __version__ as transformers_version

    dev = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    device = torch.device(dev)
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but PyTorch does not detect a CUDA device")
    if args.amp != "fp32":
        if dev != "cuda":
            raise ValueError("bf16/fp16 autocast requires CUDA. Use --amp fp32 on CPU")
        if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA GPU does not support BF16; use --amp fp32")
    print(f"Environment: torch={torch.__version__}, transformers={transformers_version}, "
          f"device={device}, amp={args.amp}, padding={args.padding}, max_length={args.max_length}")
    if args.amp == "fp32":
        print("[NOTE] Training TEST inference used BF16 autocast; FP32 may differ near ties")
    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    reports = []
    detail = []
    mask_notes = {}
    for seed in args.seeds:
        for condition in REGIMES:
            path = checkpoints[(condition, seed)]
            print(f"\n===== CONDITION {condition} | SEED {seed} =====", flush=True)
            scores = predict(rows, path, device, args.batch_size,
                             args.max_length, args.padding, args.amp)
            diffs = [float(p) - float(n) for p, n in scores]
            metrics = pair_metrics(diffs)
            reference, note = original_mask(root, condition, seed, len(all_test), args.limit)
            if reference is not None:
                mismatched = [i for i, (v, delta) in enumerate(zip(reference, diffs))
                              if bool(v) != (delta > 0)]
                note.update({"mismatch_count": len(mismatched),
                             "mismatch_sample_indices": mismatched[:30],
                             "checked_rows": len(rows)})
                print(f"  Original TEST mask comparison: {len(mismatched)} mismatches")
            else:
                print(f"  Original TEST mask: {note['status']}")
            mask_notes[f"{condition}_{seed}"] = note
            reports.append({**metrics, "condition": condition, "seed": seed,
                            "checkpoint": checkpoint_identity(path, args.hash_weights),
                            "original_mask": note})
            for i, (row, (positive, negative), diff) in enumerate(zip(rows, scores, diffs)):
                detail.append({"condition": condition, "seed": seed, "sample_index": i,
                               "doc_id": str(row.get("doc_id", "")),
                               "positive_score": positive, "negative_score": negative,
                               "score_gap": diff, "correct": int(diff > 0),
                               "tie": int(diff == 0)})
            print(f"  accuracy={metrics['pairwise_accuracy']:.6f}, "
                  f"MRR={metrics['mrr']:.6f}, NDCG@2={metrics['ndcg_at_2']:.6f}")

    aggregate = {}
    for condition in REGIMES:
        selected = [r for r in reports if r["condition"] == condition]
        aggregate[condition] = {}
        for key in ("pairwise_accuracy", "precision_at_1", "mrr", "ndcg_at_2"):
            values = [r[key] for r in selected]
            aggregate[condition][key] = {
                "mean": statistics.mean(values),
                # Original fine-tuning script prints np.std() == population SD.
                "population_std_for_training_comparison": statistics.pstdev(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                "n_seeds": len(values),
            }
    out_dir = (args.out_dir or (root / "results" / "checkpoint_diagnostic")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    postfix = "full" if args.limit == 0 else f"pilot_first_{args.limit}"
    # Do not overwrite scores generated with another numeric precision/padding regime.
    postfix += (f"_{args.amp}_{args.padding}_length{args.max_length}"
                f"_seeds{'-'.join(str(seed) for seed in args.seeds)}"
                f"_test{test_sha[:10]}")
    details_path = out_dir / f"pairwise_scores_{postfix}.csv"
    summary_path = out_dir / f"pairwise_summary_{postfix}.json"
    manifest = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "test_sha256": test_sha,
        "test_total_rows": len(all_test), "test_evaluated_rows": len(rows),
        "is_full_test": args.limit == 0,
        "test_unique_queries": len(query_counts),
        "test_unique_doc_ids": len(doc_counts),
        "seeds": args.seeds,
        "hash_weights": args.hash_weights,
        "amp": args.amp, "device": str(device),
        "padding": args.padding, "max_length": args.max_length,
        "batch_size": args.batch_size,
        "torch_version": torch.__version__, "transformers_version": transformers_version,
        "metric_definition": "One positive vs one negative per TEST row; strict ties wrong; NDCG@2=1/log2(3) if incorrect",
        "rows": reports,
        "aggregate_by_regime": aggregate,
        "output_scores": str(details_path),
    }
    # Successful completion only: failures in loading/scoring won't replace old outputs.
    save_atomic_csv(details_path, detail)
    save_atomic_json(summary_path, manifest)
    print("\n=== DIAGNOSTIC COMPLETE ===")
    for cond, metrics in aggregate.items():
        print(f"{cond}: Accuracy={metrics['pairwise_accuracy']['mean']:.6f} "
              f"± {metrics['pairwise_accuracy']['population_std_for_training_comparison']:.6f} "
              "(population SD over seeds)")
    print(f"Scores: {details_path}\nSummary: {summary_path}")
    print("Inference only; checkpoint and TEST files were not modified")


if __name__ == "__main__":
    main()
