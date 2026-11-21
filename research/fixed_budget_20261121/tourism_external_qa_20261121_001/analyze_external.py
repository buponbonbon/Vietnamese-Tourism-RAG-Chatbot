"""Summarize frozen external rankings and score files with paired intervals."""

import csv
import importlib.metadata
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

from run_external import ARMS, ROOT, combine, digest, load, load_source, save

DEPTH = 5
SEED = 20261121


def contrast(a, b, sources):
    """Return A − B in the matched cohort, bootstrapping article clusters."""
    changes = np.asarray(a, dtype=int) - np.asarray(b, dtype=int)
    new_hits, lost_hits = int(sum(changes == 1)), int(sum(changes == -1))
    groups = defaultdict(list)
    for change, source in zip(changes, sources, strict=True):
        groups[source].append(change)
    counts = np.asarray([len(v) for v in groups.values()])
    totals = np.asarray([sum(v) for v in groups.values()])
    n = len(groups)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(40):
        indices = rng.integers(0, n, size=(500, n))
        draws.extend(totals[indices].sum(axis=1) / counts[indices].sum(axis=1))
    result = {"diff": float(changes.mean()), "net_hits": new_hits - lost_hits,
              "a_only": new_hits, "b_only": lost_hits,
              "paired_bootstrap_95": np.quantile(draws, [.025, .975]).tolist()}
    if n == len(a):
        result["exact_mcnemar_p"] = (binomtest(new_hits, new_hits + lost_hits).pvalue
                                      if new_hits + lost_hits else 1.0)
    return result


def make_report():
    files = ["PROTOCOL.json", "COHORT.json", "RANKS.json", "BM25_UNICODE.json",
             "RETRIEVAL_RUN.json", "run_external.py", "analyze_external.py"]
    protocol = load("PROTOCOL.json")
    cohort = load("COHORT.json")
    ranks = load("RANKS.json")
    baseline = load("BM25_UNICODE.json")
    infrastructure = load("RETRIEVAL_RUN.json")
    docs, qs = load_source()
    ids = cohort["ids"]
    all_ids = [q["id"] for q in qs]
    by_id = {q["id"]: q for q in qs}
    sources = [by_id[q]["article"] for q in all_ids]
    cohort_sources = [by_id[q]["article"] for q in ids]
    assert len(ids) == len(set(cohort_sources)) == 140
    assert all(by_id[q]["valid_offsets"] for q in ids)
    assert set(all_ids) == set(ranks) == set(baseline)
    for q in all_ids:
        row = ranks[q]
        assert row["gold"] == by_id[q]["doc"] and row["article"] == by_id[q]["article"]
        assert row["rrf"] == combine(row["bm25"], row["dense"])
        for ordering in [row[k] for k in ARMS] + [baseline[q]]:
            assert len(ordering) == len(set(ordering)) == 50
            assert all(0 <= j < len(docs) for j in ordering)

    per_query = {q: {"qid": q, "article": by_id[q]["article"], "gold_id": by_id[q]["doc"]}
                 for q in ids}
    expected = {(q, d) for q in ids for a in ARMS for d in ranks[q][a][:DEPTH]}
    score_runs = {}
    model_names = dict(zip(("BGE", "M0"), protocol["reranker"], strict=True))
    for kind, name in model_names.items():
        filename = name + "_scores.json"
        files.append(filename)
        val = load(filename)
        assert val["cohort_sha256"] == digest(ROOT / "COHORT.json")
        assert val["ranks_sha256"] == digest(ROOT / "RANKS.json")
        scores = {(row["q"], row["d"]): (row["score"], row["tokens"]) for row in val.pop("scores")}
        assert set(scores) == expected
        assert all(np.isfinite(s) and 1 <= t <= 512 for s, t in scores.values())
        score_runs[kind] = {**val, "scores": scores}

    all_availability = {}
    group_summary = {}
    hit_masks = {}
    available_masks = {}
    summary_rows = []
    for arm in [*ARMS, "bm25_unicode"]:
        pools = {q: (baseline[q] if arm == "bm25_unicode" else ranks[q][arm]) for q in all_ids}
        all_availability[arm] = {
            f"at_{k}": sum(by_id[q]["doc"] in pools[q][:k] for q in all_ids)
            for k in (1, DEPTH, 10, 20, 50)
        }
        available_masks[arm] = [by_id[q]["doc"] in pools[q][:DEPTH] for q in all_ids]
        if arm == "bm25_unicode":
            continue

        golds = {q: by_id[q]["doc"] for q in ids}
        pool = {q: pools[q][:DEPTH] for q in ids}
        available = sum(golds[q] in pool[q] for q in ids)
        group_summary[arm] = {"available": available, "models": {}}
        for kind in ["Raw", *model_names]:
            score = score_runs[kind]["scores"] if kind in score_runs else {}
            selected = {}
            cost = 0
            for q in ids:
                winner = max(pool[q], key=lambda d: score[q, d][0]) if score else pool[q][0]
                selected[q] = winner == golds[q]
                per_query[q][f"{arm}_available"] = int(golds[q] in pool[q])
                per_query[q][f"{arm}_{kind}_winner"] = winner
                per_query[q][f"{arm}_{kind}_hit"] = int(selected[q])
                if score:
                    cost += sum(score[q, d][1] for d in pool[q])
            hit_masks[arm, kind] = list(selected.values())
            hit = sum(selected.values())
            assert hit <= available
            row = {"retriever": arm, "reranker": kind, "questions": len(ids),
                   "available": available, "gold_hit1": hit, "absent": len(ids) - available,
                   "present_not_selected": available - hit,
                   "conditional_gold_hit1": hit / available if available else None,
                   "inference_pairs": len(ids) * DEPTH if score else 0,
                   "inference_unpadded_tokens": cost}
            group_summary[arm]["models"][kind] = row
            summary_rows.append(row)

    differences = {}
    for k in ["Raw", *model_names]:
        for a, b in (("rrf", "bm25"), ("dense", "bm25"), ("rrf", "dense")):
            differences[f"{a}_minus_{b}_{k}"] = contrast(hit_masks[a, k], hit_masks[b, k],
                                                        cohort_sources)
    for a in ARMS:
        for target in ("BGE", "Raw"):
            differences[f"M0_minus_{target}_{a}"] = contrast(hit_masks[a, "M0"],
                                                             hit_masks[a, target], cohort_sources)
    availability_diff = {
        f"{a}_minus_{b}": contrast(available_masks[a], available_masks[b], sources)
        for a, b in (("rrf", "bm25"), ("dense", "bm25"), ("rrf", "dense"),
                     ("bm25_unicode", "bm25"), ("rrf", "bm25_unicode"), ("dense", "bm25_unicode"))
    }
    decomposition = {}
    for a, b in (("rrf", "bm25"), ("dense", "bm25"), ("rrf", "dense")):
        partitions = defaultdict(list)
        for i, q in enumerate(ids):
            status = (by_id[q]["doc"] in ranks[q][a][:DEPTH],
                      by_id[q]["doc"] in ranks[q][b][:DEPTH])
            partitions[status].append(i)
        decomposition[f"{a}_vs_{b}"] = {}
        for status, label in [((True, True), "both"), ((True, False), "a_only"),
                              ((False, True), "b_only"), ((False, False), "neither")]:
            indices = partitions[status]
            decomposition[f"{a}_vs_{b}"][label] = {
                "queries": len(indices),
                "rank1_counts": {
                    kind: [sum(hit_masks[arm, kind][i] for i in indices) for arm in (a, b)]
                    for kind in ("Raw", "BGE", "M0")
                }
            }
    score_details = {k: {a: v for a, v in run.items() if a != "scores"}
                     for k, run in score_runs.items()}
    output = {"study": protocol["study"], "cohort_audit": cohort["audit"],
              "all_query_availability": all_availability, "all_query_hit5_contrasts": availability_diff,
              "reranked_cohort": group_summary,
              "cohort_bm25_unicode": {
                  "at_1": sum(by_id[q]["doc"] == baseline[q][0] for q in ids),
                  "at_5": sum(by_id[q]["doc"] in baseline[q][:5] for q in ids)
              },
              "contrast": differences,
              "availability_partition_diagnostic": decomposition,
              "primary": differences["rrf_minus_bm25_M0"],
              "distinct_scored_pairs_per_model": len(expected),
              "inputs_sha256": {name: digest(ROOT / name) for name in files},
              "retrieval_runtime": infrastructure, "reranker_runtime": score_details,
              "scipy_version": importlib.metadata.version("scipy")}
    save("EXTERNAL_RESULTS.json", output)
    for path, rows in (("EXTERNAL_SUMMARY.csv", summary_rows),
                       ("EXTERNAL_QUERY_OUTCOMES.csv", list(per_query.values()))):
        with (ROOT / path).open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("Primary", output["primary"])
    print("Availability", all_availability)
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    make_report()
