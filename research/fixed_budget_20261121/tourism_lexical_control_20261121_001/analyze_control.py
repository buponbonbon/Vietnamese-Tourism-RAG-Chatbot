"""Summarize rank and selection contrasts under all completions of missing grades."""

import csv
import itertools
import math

from run_control import BGE, EXTERNAL, M0, PREV, ROOT, prior

MODES = ("Raw", "BGE", "M0")
ARMS = ("legacy", "rrf_legacy", "dense", "unicode", "rrf_unicode")


def outcomes(pool, winner, grades):
    known_sufficient = any(grades.get(d) == 2 for d in pool)
    uncertain = [d for d in pool if d not in grades]
    y_low, y_high = prior.grade_bits(grades.get(winner))
    y = list(range(y_low, y_high + 1))
    states = {(int(known_sufficient or y0 or other), y0) for y0 in y
              for other in range(1 + int(any(d != winner for d in uncertain)))}
    return sorted(states)


def availability_bounds(a, b, grades):
    """Use shared unknowns once for an exact A-minus-B availability contrast."""
    possible = {d for d in set(a) | set(b) if d not in grades}
    shared = possible & set(a) & set(b)
    only_first = possible & (set(a) - set(b))
    only_second = possible & (set(b) - set(a))
    known_a = any(grades.get(d) == 2 for d in a)
    known_b = any(grades.get(d) == 2 for d in b)
    diffs = []
    for joint, first, second in itertools.product(range(1 + bool(shared)),
                                                   range(1 + bool(only_first)),
                                                   range(1 + bool(only_second))):
        diffs.append(int(known_a or joint or first) - int(known_b or joint or second))
    return min(diffs), max(diffs)


def sharp_sums(items):
    items = list(items)
    return [sum(min(values) for values in items), sum(max(values) for values in items)]


def summarize():
    _, questions, _, old, known, _, label_hashes = prior.load()
    r = prior.read(ROOT / "RANKS.json")
    assert r["legacy"] == old
    pools = {arm: {q: d[:5] for q, d in byq.items()} for arm, byq in r.items()}
    ids = [q["benchmark_id"] for q in questions]
    n = len(ids)
    per_q = {q: {d: grade for (qid, d), grade in known.items() if qid == q} for q in ids}
    candidates = {(q, d) for arm in ARMS for q, pool in pools[arm].items() for d in pool}
    logits = {}
    scores_meta = {}
    for short, name in [("BGE", BGE), ("M0", M0)]:
        x = prior.read(ROOT / (name + "_scores.json"))
        assert x["ranks_sha256"] == prior.sha(ROOT / "RANKS.json")
        rows = x.pop("scores")
        logits[short] = {(r["q"], r["d"]): (r["s"], r["tokens"]) for r in rows}
        assert len(rows) == len(logits[short]) and set(logits[short]) == candidates
        assert all(math.isfinite(s) and 1 <= t <= 512 for s, t in logits[short].values())
        scores_meta[short] = x

    rows = []
    cells = {}
    winner_maps = {}
    for arm in ARMS:
        for mode in MODES:
            scores = logits.get(mode)
            winners = {q: (max(p, key=lambda d: scores[q, d][0]) if scores else p[0])
                       for q, p in pools[arm].items()}
            winner_maps[arm, mode] = winners
            feasible = {q: outcomes(pools[arm][q], d, per_q[q]) for q, d in winners.items()}
            cell = {
                "availability": sharp_sums([a for a, y in state] for state in feasible.values()),
                "hit1": sharp_sums([y for a, y in state] for state in feasible.values()),
                "absence": sharp_sums([1 - a for a, y in state] for state in feasible.values()),
                "available_but_missed": sharp_sums([a - y for a, y in state] for state in feasible.values()),
                "known_pairs": sum((q, d) in known for q in ids for d in pools[arm][q]),
                "unknown_winners": sum((q, d) not in known for q, d in winners.items()),
                "scored_pairs": n * 5 if scores else 0,
                "tokens": sum(scores[q, d][1] for q in ids for d in pools[arm][q]) if scores else 0
            }
            cells[f"{arm}_{mode}"] = cell
            for q in ids:
                d = winners[q]
                rows.append({"qid": q, "arm": arm, "model": mode, "winner": d,
                             "winner_grade": per_q[q].get(d, "unjudged"),
                             "available_min": min(a for a, _ in feasible[q]),
                             "available_max": max(a for a, _ in feasible[q])})

    selection = {}
    availability = {}
    contrasts = (("unicode", "legacy"), ("rrf_unicode", "unicode"), ("rrf_legacy", "unicode"),
                 ("dense", "unicode"), ("rrf_unicode", "rrf_legacy"))
    for a, b in contrasts:
        availability[f"{a}_minus_{b}"] = sharp_sums(
            availability_bounds(pools[a][q], pools[b][q], per_q[q]) for q in ids)
        for model in MODES:
            label = f"{a}_minus_{b}_{model}"
            selection[label] = prior.paired_hits(winner_maps[a, model], winner_maps[b, model], known)
    for arm in ARMS:
        selection[f"M0_minus_BGE_{arm}"] = prior.paired_hits(
            winner_maps[arm, "M0"], winner_maps[arm, "BGE"], known)
    for a in ("Raw", "BGE"):
        selection[f"unicode_{a}_minus_legacy_M0"] = prior.paired_hits(
            winner_maps["unicode", a], winner_maps["legacy", "M0"], known)

    anchor = {row["benchmark_id"]: row["original_anchor_chunk_id"] for row in questions}
    gaps = [q["benchmark_id"] for q in questions if not q["has_grade2_in_native_pool"]]
    assert len(gaps) == 95 and all(known[q, anchor[q]] == 2 for q in gaps)
    cohort_by_id = {row["benchmark_id"]: row for row in questions}

    def summarized_subset(subset):
        return {
            arm: {"availability": sharp_sums(
                prior.availability(q, pools[arm][q], known) for q in subset),
                "hit1_M0": sharp_sums(
                    prior.grade_bits(known.get((q, winner_maps[arm, "M0"][q]))) for q in subset),
                "original_anchor_in_top5": sum(anchor[q] in pools[arm][q] for q in subset),
                "known_grade2_outside_anchor_in_top5": sum(any(known.get((q, d)) == 2 and d != anchor[q]
                                                               for d in pools[arm][q]) for q in subset)}
            for arm in ARMS
        }

    split = {k: [q for q in ids if cohort_by_id[q]["recorded_split"] == k]
             for k in sorted({row["recorded_split"] for row in questions})}
    behavior = {k: [q for q in ids if cohort_by_id[q]["behavior"] == k]
                for k in sorted({row["behavior"] for row in questions})}
    legacy_ceiling = cells["legacy_M0"]["availability"][0]
    ceiling_comparisons = {
        mode: {"new_Hit1_lower_count_minus_legacy_ceiling":
               cells["unicode_" + mode]["hit1"][0] - legacy_ceiling}
        for mode in MODES
    }
    input_paths = [ROOT / x for x in ("PROTOCOL.json", "RETRIEVAL_RUN.json", "RANKS.json", "run_control.py",
                                     M0 + "_scores.json", BGE + "_scores.json")]
    input_paths += [PREV / x for x in ("evaluate_fixed_budget.py", "retrieval_rankings.json",
                                      "DENSE_ONLY_SCORES.json", "DENSE_ONLY_SELECTED.json",
                                      M0 + "_scores.json", BGE + "_scores.json")]
    input_paths.append(EXTERNAL / "run_external.py")
    result = {
        "queries": n,
        "source_groups": len({row["source_group_id"] for row in questions}),
        "baseline_type": "fixed legacy exact by the archived labels",
        "cells": cells,
        "paired_availability_count_bounds": availability,
        "paired_hit1_count_bounds": selection,
        "historical_native_missing_subset": {"n": len(gaps), "cells": summarized_subset(gaps)},
        "recorded_splits": {k: {"n": len(subset), "cells": summarized_subset(subset)}
                            for k, subset in split.items()},
        "new_vs_legacy_relative_gain_percent": {
            "availability": [100 * x / cells["legacy_M0"]["availability"][0]
                             for x in availability["unicode_minus_legacy"]],
            "hit1_M0": [100 * x / cells["legacy_M0"]["hit1"][0]
                        for x in selection["unicode_minus_legacy_M0"]["bounds_count"]]
        },
        "behavior_subgroups_descriptive": {k: {"n": len(subset), "cells": summarized_subset(subset)}
                                           for k, subset in behavior.items()},
        "legacy_top5_any_selector_ceiling": legacy_ceiling,
        "unicode_exceeds_ceiling": ceiling_comparisons,
        "label_source_sha256": label_hashes,
        "score_metadata": scores_meta,
        "input_files": {str(p.relative_to(ROOT.parent)): prior.sha(p) for p in input_paths}
    }
    prior.save(ROOT / "RESULTS.json", result)
    with (ROOT / "PER_QUERY.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Cells", {k: cells[k] for k in cells if k.endswith("_M0")})
    print("Availability", availability)
    print("Paired M0", {k: d for k, d in selection.items() if k.endswith("_M0")})
    print("Percent relative to legacy", result["new_vs_legacy_relative_gain_percent"])


if __name__ == "__main__":
    summarize()
