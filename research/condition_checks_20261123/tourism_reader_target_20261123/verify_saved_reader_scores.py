"""Recompute Table S22 from archived logits and the frozen pool projection."""

from collections import Counter
import csv
from hashlib import sha256
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
S20_COHORT = "tourism_reader_target_20261123"
MODEL_NAMES = {
    "BGE": "BGE_pretrained",
    **{
        f"{family}_{seed}": f"{family}_fixed_1p0_weighted_matched_seed_{seed}"
        for family in ("M0", "M1")
        for seed in (42, 123, 999)
    },
}


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def check_pools(questions):
    require(len(questions) == len({q["id"] for q in questions}) == 24, "Reader cohort size")
    for q in questions:
        case = q["id"]
        native, oracle = q["native"], q["oracle"]
        native_gold, oracle_gold = q["native_sufficient"], q["oracle_sufficient"]
        require(len(native) == len(set(native)) == len(oracle) == len(set(oracle)) == 5, case)
        require(set(native) == set(q["rating"]) and set(q["rating"].values()) <= {0, 1, 2}, case)
        require(native_gold == [doc for doc in native if q["rating"][doc] == 2], case)
        if native_gold:
            require(native == oracle and native_gold == oracle_gold, case)
        else:
            require(native[:4] == oracle[:4] and oracle[-1] not in native, case)
            require(oracle_gold == oracle[-1:], case)


def recompute(model, questions, scores):
    counts = dict(native_hit=0, available_native_fail=0, replacement_hit=0)
    trace = []
    for q in questions:
        case = q["id"]
        available = bool(q["native_sufficient"])
        for pool in (("native",) if available else ("native", "oracle")):
            candidates = q[pool]
            ranking = sorted(candidates, key=lambda doc: (-scores[case, doc], doc)) if scores else candidates
            winner = ranking[0]
            hit = winner in q[f"{pool}_sufficient"]
            if pool == "native":
                counts["native_hit"] += hit
                counts["available_native_fail"] += available and not hit
            else:
                counts["replacement_hit"] += hit
            trace.append(dict(case=case, model=model, pool=pool, selected=winner,
                              hit=str(hit), available=str(bool(q[f"{pool}_sufficient"])), rank=" ".join(ranking)))
    return counts, trace


def main():
    for filename, expected in read("reader_public_sha256.json").items():
        require(sha256((ROOT / filename).read_bytes()).hexdigest() == expected, filename)

    questions = read("reader_pools_s20.json")
    expected = read("reader_results.json")
    check_pools(questions)
    available = sum(bool(q["native_sufficient"]) for q in questions)
    require((available, len(questions) - available) == (expected["native_available"], expected["native_missing"]),
            "Availability counts")
    require(expected["n_new"] == len(questions), "New cohort size")

    ledger = read("reader_cohort_index.json")
    require(len({row["case"] for row in ledger}) == len(ledger) == expected["n_cumulative"], "Cumulative cohort")
    require(Counter(row["cohort"] for row in ledger) == expected["source_question_archives"], "Cohort membership")
    require(sum(row["mode"] == "native top five" for row in ledger) == expected["n_cumulative_native"],
            "Cumulative native cohort")
    require({q["id"] for q in questions} == {row["case"] for row in ledger if row["cohort"] == S20_COHORT},
            "S20 cohort membership")

    pairs = {(q["id"], doc) for q in questions for pool in ("native", "oracle") for doc in q[pool]}
    require(len(pairs) == expected["n_scored_pairs"], "Scored pair coverage")
    require(len(pairs) * len(MODEL_NAMES) == expected["n_checkpoint_logits"], "Logit counts")
    recomputed_trace = []
    for alias in ("BM25", *MODEL_NAMES):
        recorded = expected["models"][alias]
        scores = {}
        if alias != "BM25":
            archived = read(f"reader_scores_{alias}.json")
            values = archived.pop("scores")
            require(archived == recorded["metadata"], f"{alias} runtime metadata")
            require(archived["checkpoint"] == MODEL_NAMES[alias], f"{alias} checkpoint identity")
            require(archived["benchmark_sha256"] == expected["benchmark_sha256"], f"{alias} benchmark identity")
            scores = {(case, doc): value for case, doc, value in values}
            require(len(values) == len(scores) and scores.keys() == pairs, f"{alias} logit coverage")
            require(all(math.isfinite(value) for value in scores.values()), f"{alias} non-finite logits")
        counts, trace = recompute(alias, questions, scores)
        require(all(recorded[key] == value for key, value in counts.items()), f"{alias} Table S22")
        recomputed_trace.extend(trace)
        print(alias, counts)

    with (ROOT / "reader_selection_trace.csv").open(encoding="utf-8-sig", newline="") as stream:
        require(list(csv.DictReader(stream)) == recomputed_trace, "Saved per-case rankings")
    print(f"PASS S20: {len(questions)} queries, {len(pairs)} pairs, {len(recomputed_trace)} ranking rows")


if __name__ == "__main__":
    main()
