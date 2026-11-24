"""Score the frozen candidate sets with archived checkpoints (CPU, no training)."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time

from build_benchmark import ROOT, digest, save

MANIFEST = Path("/workspace/Vietnamese-Tourism-RAG-Chatbot/data/oracle_recovery_20261113_001/"
                "evidence_selection_reinference_001/protocol.json")
MODELS = {
    "BGE": "BGE_pretrained",
    "M0": "M0_fixed_1p0_weighted_matched_seed_42",
    "M1": "M1_fixed_1p0_weighted_matched_seed_42",
}
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def score(benchmark, archive, model_name, batch_size):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import transformers

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    meta = next(m for m in archive["models"] if m["name"] == MODELS[model_name])
    checkpoint = Path(meta["path"])
    for name, expected in meta["file_sha256"].items():
        assert digest(checkpoint / name) == expected, name
    tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    net = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, torch_dtype=torch.float32, local_files_only=True, trust_remote_code=False,
        attn_implementation="sdpa").eval()
    docs, pairs = benchmark["docs"], benchmark["pairs"]
    styles = ["plain", "explicit"] if model_name == "M0" else ["plain"]
    result = {"benchmark_sha256": digest(ROOT / "benchmark.json"), "checkpoint": meta["name"],
              "checkpoint_files": meta["file_sha256"], "torch": torch.__version__,
              "transformers": transformers.__version__, "batch_size": batch_size, "by_style": {}}
    for fmt in styles:
        records = [(p["id"], i, d, q, docs[d][fmt]) for p in pairs
                   for i, q in enumerate(p["queries"]) for d in p["pool"]]
        keys = [(k, i, d) for k, i, d, _, _ in records]
        assert len(keys) == len(set(keys))
        encoded = tok([x[3] for x in records], [x[4] for x in records],
                      truncation=False, padding=False)
        lens = [len(seq) for seq in encoded["input_ids"]]
        assert min(lens) > 0 and max(lens) <= 512
        order = sorted(range(len(records)), key=lambda x: (lens[x], x))
        values = [None] * len(records)
        tick = time.monotonic()

        def run(ix):
            data = {key: [encoded[key][i] for i in ix] for key in encoded}
            batch = tok.pad(data, padding=True, return_tensors="pt")
            with torch.inference_mode():
                out = net(**batch).logits.flatten().tolist()
            assert len(out) == len(ix) and all(math.isfinite(val) for val in out)
            return out

        for j in range(0, len(order), batch_size):
            selection = order[j:j+batch_size]
            for i, val in zip(selection, run(selection)):
                values[i] = val
            done = min(j + batch_size, len(order))
            if done % (batch_size * 10) == 0 or done == len(order):
                print(f"{model_name}/{fmt} {done}/{len(order)} ({time.monotonic()-tick:.1f}s)", flush=True)
        check_points = order[::max(1, len(order)//8)][:8]
        delta = max(abs(values[i] - run([i])[0]) for i in check_points)
        assert delta < 1e-3, delta
        result["by_style"][fmt] = {
            "max_input_tokens": max(lens), "pair_count": len(keys), "unpadded_tokens": sum(lens),
            "batch_consistency_max_abs_delta": delta,
            "scores": [[*k, val] for k, val in zip(keys, values)],
        }
    del net
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, default=MANIFEST)
    parser.add_argument("--batch-size", type=int, default=16)
    options = parser.parse_args()
    bm = json.loads((ROOT / "benchmark.json").read_text())
    assert bm["protocol_sha256"] == digest(ROOT / "protocol.json")
    assert not (ROOT / f"scores_{options.model}.json").exists()
    archive = json.loads(options.checkpoint_manifest.read_text())
    save(ROOT / f"scores_{options.model}.json", score(bm, archive, options.model, options.batch_size))


if __name__ == "__main__":
    main()
