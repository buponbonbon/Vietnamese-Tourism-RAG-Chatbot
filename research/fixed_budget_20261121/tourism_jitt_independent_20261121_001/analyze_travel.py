"""Compute fixed-budget reference retrieval and selection outcomes."""

import csv
import itertools
from collections import defaultdict
from fractions import Fraction
import math

import numpy as np
from scipy.stats import binomtest

from prepare_travel import ROOT, load, parse, save, sha
from run_travel import ARMS


def compare(a, b):
    x = np.asarray(a, dtype=int) - np.asarray(b, dtype=int)
    rng = np.random.default_rng(20261121)
    boot = x[rng.integers(0, len(x), size=(20_000, len(x)))].mean(axis=1)
    gain = int(sum(x == 1))
    loss = int(sum(x == -1))
    return {"difference_pp": 100 * x.mean(),
            "bootstrap_95_pp": (100 * np.quantile(boot, [.025,.975])).tolist(),
            "first_only": gain, "second_only": loss,
            "mcnemar_p": binomtest(gain, gain + loss).pvalue if gain + loss else 1.0}


def exact_rank(a, b):
    x = {d: i for i,d in enumerate(a,1)}
    y = {d: i for i,d in enumerate(b,1)}
    values = {d: sum(Fraction(1,60+z[d]) for z in (x, y) if d in z) for d in x.keys() | y.keys()}
    return sorted(values, key=lambda d: (-values[d], x.get(d,51), y.get(d,51), d))[:50]


def main():
    answers, cohort, audit = parse()
    assert audit == load("CORPUS_AUDIT.json")
    cohort_ids = load("COHORT_IDS.json")
    assert [r["qid"] for r in cohort_ids] == [q["qid"] for q in cohort]
    ids = [str(q["qid"]) for q in cohort]
    n = len(ids)
    ranks = load("RANKS.json")
    files = ["PROTOCOL.json", "COHORT_IDS.json", "CORPUS_AUDIT.json", "RANK_RUN.json",
             "RANKS.json", "BGE_scores.json", "M0_scores.json", "prepare_travel.py", "run_travel.py",
             "analyze_travel.py"]
    rank_metrics = {}
    for field, rr in ranks.items():
        assert set(rr) == set(ids)
        for q in cohort:
            row = rr[str(q["qid"])]
            assert row["gold"] == q["gold"] and answers[q["gold"]][0] == q["qid"]
            assert row["rrf"] == exact_rank(row["bm25"], row["dense"])
            for arm in ARMS:
                p = row[arm]
                assert len(p) == len(set(p)) == 50 and set(p) <= answers.keys()
        rank_metrics[field] = {arm: {str(k): sum(r["gold"] in r[arm][:k] for r in rr.values())
                                     for k in (1,5,10,20,50)} for arm in ARMS}
    rr = ranks["text"]
    expected = {(q,d) for q in ids for arm in ARMS for d in rr[q][arm][:5]}
    ce = {}
    ce_run = {}
    for m in ("BGE","M0"):
        trace = load(m+"_scores.json")
        assert trace["ranks_sha256"] == sha(ROOT / "RANKS.json")
        rows = trace.pop("rows")
        ce[m] = {(str(row["q"]),row["d"]): row for row in rows}
        assert set(ce[m]) == expected and len(ce[m]) == len(rows)
        assert all(math.isfinite(r["logit"]) and 1 <= r["tokens"] <= 512 for r in rows)
        ce_run[m] = trace
    rows = {q: {"qid": q, "gold": rr[q]["gold"]} for q in ids}
    hits = {}
    avails = {}
    results = {}
    for arm in ARMS:
        pool = {q: rr[q][arm][:5] for q in ids}
        avail = [rr[q]["gold"] in pool[q] for q in ids]
        avails[arm] = avail
        results[arm] = {}
        for m in ("raw", "BGE", "M0"):
            winner = {q: (min(enumerate(pool[q]),
                              key=lambda item: (-ce[m][q,item[1]]["logit"],item[0]))[1]
                          if m in ce else pool[q][0]) for q in ids}
            hit = [winner[q] == rr[q]["gold"] for q in ids]
            hits[arm,m] = hit
            assert all(y <= a for y,a in zip(hit,avail,strict=True))
            results[arm][m] = {
                "hit1": sum(hit), "availability5": sum(avail),
                "accepted_absent": n - sum(avail), "accepted_present_not_selected": sum(avail) - sum(hit),
                "pairs": n * 5 if m in ce else 0,
                "tokens": sum(ce[m][q,d]["tokens"] for q in ids for d in pool[q]) if m in ce else 0,
                "truncated_pairs": sum(ce[m][q,d]["truncated"] for q in ids for d in pool[q]) if m in ce else 0
            }
            for q, a, y in zip(ids,avail,hit,strict=True):
                rows[q][f"{arm}_available"] = int(a)
                rows[q][f"{arm}_{m}_winner"] = winner[q]
                rows[q][f"{arm}_{m}_hit"] = int(y)
    contrasts = {f"{a}_minus_{b}_{m}": compare(hits[a,m],hits[b,m])
                 for a,b in itertools.permutations(ARMS,2) for m in ("raw","BGE","M0")}
    contrasts.update({f"M0_minus_BGE_{a}": compare(hits[a,"M0"],hits[a,"BGE"]) for a in ARMS})
    diag = defaultdict(dict)
    for a,b in [("rrf","bm25"),("dense","bm25")]:
        for x,y in itertools.product((0,1), repeat=2):
            selected = [i for i,(aa,bb) in enumerate(zip(avails[a],avails[b],strict=True)) if (aa,bb)==(x,y)]
            diag[a+"_vs_"+b][f"{x}{y}"] = {"n":len(selected),
                                          "M0_hits":[sum(hits[z,"M0"][i] for i in selected) for z in (a,b)]}
    result = {"cohort": audit, "retrieval": rank_metrics, "results": results, "per_arm_paired": contrasts,
              "same_query_availability_partition": diag, "unique_pairs_per_model": len(expected),
              "primary": contrasts["rrf_minus_bm25_M0"], "reranking_meta": ce_run,
              "inputs_sha256": {name: sha(ROOT / name) for name in files}}
    save("TRAVEL_RESULTS.json", result)
    with (ROOT / "TRAVEL_QUERY_OUTCOMES.csv").open("x",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=list(next(iter(rows.values()))))
        writer.writeheader()
        writer.writerows(rows.values())
    print("Primary",result["primary"])
    print("Results",results)
    print("Same-query decompositions",dict(diag))


if __name__ == "__main__":
    main()
