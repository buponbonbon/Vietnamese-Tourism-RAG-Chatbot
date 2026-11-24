"""Replay the tourism fixed-budget experiment with symmetric Unicode BM25."""

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREV = ROOT.parent / "tourism_retrieval_budget_20261120_001"
EXTERNAL = ROOT.parent / "tourism_external_qa_20261121_001"
sys.dont_write_bytecode = True


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior = load_script("tourism_fixed_budget", PREV / "evaluate_fixed_budget.py")
external = load_script("mlqa_experiment", EXTERNAL / "run_external.py")

M0 = "M0_fixed_1p0_weighted_matched_seed_42"
BGE = "BGE_pretrained"


def retrieve():
    _, questions, docs, legacy, known, _, _ = prior.load()
    past = prior.read(PREV / "retrieval_rankings.json")
    assert legacy == past["pools"]["bm25"] and len(known) == 903
    ids = list(docs)
    query_rows = [{"id": q["benchmark_id"], "text": q["query"]} for q in questions]
    start = time.monotonic()
    new, short = external.bm25(list(docs.values()), query_rows, "unicode")
    unicode_ranks = {q: [ids[d] for d in row] for q, row in new.items()}
    pools = {
        "legacy": legacy,
        "rrf_legacy": past["pools"]["rrf"],
        "dense": past["dense"],
        "unicode": unicode_ranks,
        "rrf_unicode": {q: prior.rrf(unicode_ranks[q], past["dense"][q]) for q in legacy},
    }
    assert all(set(pool) == set(legacy) for pool in pools.values())
    for pool in pools.values():
        assert all(len(row) == len(set(row)) == 50 for row in pool.values())
    prior.save(ROOT / "RANKS.json", pools)
    prior.save(ROOT / "RETRIEVAL_RUN.json",
               {"protocol_sha256": prior.sha(ROOT / "PROTOCOL.json"),
                "old_ranks_sha256": prior.sha(PREV / "retrieval_rankings.json"),
                "mlqa_script_sha256": prior.sha(EXTERNAL / "run_external.py"),
                "below_50_positive_BM25_scores": int(short),
                "retrieval_seconds": time.monotonic() - start})
    print("Candidate comparisons at K=5", prior.eval_retrievers(
        questions, pools, known, {"availability_only_depths": [5]}), flush=True)


def score():
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    frozen, questions, docs, _, _, original, _ = prior.load()
    qtext = {q["benchmark_id"]: q["query"] for q in questions}
    original = {(r["query_id"], r["chunk_id"]): r for r in original}
    pools = prior.read(ROOT / "RANKS.json")
    keys = sorted({(q, d) for ranks in pools.values() for q, row in ranks.items() for d in row[:5]})
    inputs = [(original[k]["query"], original[k]["text"]) if k in original else (qtext[k[0]], docs[k[1]])
              for k in keys]
    completion = prior.read(prior.ARC / "inference_completion.json")
    old_baseline = prior.read(PREV / "SELECTED_TOP1.json")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    for name in (M0, BGE):
        ref = next(r for r in frozen["models"] if r["name"] == name)
        model_dir = Path(ref["path"])
        for file, sha256 in ref["file_sha256"].items():
            assert prior.sha(model_dir / file) == sha256, file
        old = prior.read(PREV / (name + "_scores.json"))
        assert old["model_weight_sha256"] == ref["file_sha256"]["model.safetensors"]
        cache_file = prior.ARC / "inference" / name / "scores.json"
        assert prior.sha(cache_file) == next(r["sha256"] for r in completion["models"] if r["model"] == name)
        cache_archive = prior.read(cache_file)["scores"]
        cached = {k: cache_archive[row["row_id"]] for k, row in original.items()}
        for row in old["scores"]:
            k = row["q"], row["d"]
            assert k not in cached or cached[k] == row["value"], k
            cached[k] = row["value"]
        dense_cache_checks = []
        if name == M0:
            for row in prior.read(PREV / "DENSE_ONLY_SCORES.json"):
                k = row["q"], row["d"]
                if k in cached:
                    dense_cache_checks.append(abs(cached[k] - row["value"]))
                cached.setdefault(k, row["value"])
        dense_delta = max(dense_cache_checks, default=0)
        assert dense_delta <= 1e-4, dense_delta

        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False, dtype=torch.float32,
            attn_implementation="sdpa"
        ).eval()
        left, right = map(list, zip(*inputs))
        enc = tokenizer(left, right, max_length=512, truncation=True, padding=False)
        lengths = list(map(len, enc["input_ids"]))

        def forward(indices):
            batch = tokenizer.pad([{f: enc[f][i] for f in enc} for i in indices], return_tensors="pt")
            with torch.inference_mode():
                scores = model(**batch).logits.float().reshape(-1).tolist()
            assert len(indices) == len(scores) and all(math.isfinite(x) for x in scores)
            return scores

        probe = [i for i, k in enumerate(keys) if k in cached]
        probe = probe[::max(1, len(probe) // 8)][:8]
        error = max(abs(cached[keys[i]] - x) for start in (0, 4)
                    for i, x in zip(probe[start:start + 4], forward(probe[start:start + 4]), strict=True))
        assert error <= 1e-4, error
        new = sorted([i for i, k in enumerate(keys) if k not in cached], key=lambda i: (lengths[i], i))
        print(name, "cached pairs", len(keys) - len(new), "new pairs", len(new), flush=True)
        start = time.monotonic()
        output = {k: cached[k] for k in keys if k in cached}
        for i in range(0, len(new), 4):
            batch = new[i:i + 4]
            output.update((keys[j], x) for j, x in zip(batch, forward(batch), strict=True))
            if (i + 4) % 112 == 0 or i + 4 >= len(new):
                print(name, min(i+4, len(new)), "/", len(new), round(time.monotonic() - start, 1),
                      "seconds", flush=True)
        assert len(output) == len(keys)
        for which, old_arm in (("legacy", "bm25"), ("rrf_legacy", "rrf")):
            for q, row in pools[which].items():
                assert max(row[:5], key=lambda d: output[q, d]) == old_baseline[f"{name}_{old_arm}"][q]
        if name == M0:
            control = prior.read(PREV / "DENSE_ONLY_SELECTED.json")
            assert all(max(row[:5], key=lambda d: output[q, d]) == control[q]
                       for q, row in pools["dense"].items())
        prior.save(ROOT / (name + "_scores.json"),
                   {"model_weight_sha256": ref["file_sha256"]["model.safetensors"],
                    "cache_probe_max_abs_error": error, "new_pairs": len(new),
                    "dense_cache_overlap_max_abs_error": dense_delta,
                    "ranks_sha256": prior.sha(ROOT / "RANKS.json"),
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "scores": [{"q": q, "d": d, "s": output[q, d], "tokens": lengths[i],
                                "cached": (q, d) in cached} for i, (q, d) in enumerate(keys)]})
        del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("retrieve", "score"))
    {"retrieve": retrieve, "score": score}[parser.parse_args().stage]()
