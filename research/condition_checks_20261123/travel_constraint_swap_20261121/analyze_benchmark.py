"""Evaluate strict direction swaps and paired changes without tuning the benchmark."""

from collections import Counter, defaultdict
import csv
import json
import math
import random
from statistics import mean

from build_benchmark import BM25, ROOT, digest, save

CHECKPOINTS = ("BGE", "M0", "M1")


def lexical_scores(pairs, docs, n):
    by_network = {
        network: BM25({d: row for d, row in docs.items() if row["network"] == network}, n=n)
        for network in {p["network"] for p in pairs}
    }
    scores = {}
    for f in pairs:
        for i, query in enumerate(f["queries"]):
            values = by_network[f["network"]].rank(query)
            for d in f["pool"]:
                scores[f["id"], i, d] = values.get(d, 0.)
    return scores


def family_outcomes(pairs, scores, name):
    observed = []
    for f in pairs:
        fid, pool, gold = f["id"], f["pool"], f["gold"]
        info = []
        for i, d in enumerate(gold):
            s = {candidate: scores[(fid, i, candidate)] for candidate in pool}
            assert all(math.isfinite(x) for x in s.values())
            margin = s[d] - s[gold[1-i]]
            top = [candidate for candidate in pool if s[candidate] == max(s.values())]
            case = "gold" if top == [d] else (
                "tie" if len(top) > 1 else ("reverse" if top == [gold[1-i]] else "other"))
            info.append((int(margin > 0), int(top == [d]), margin, top, case))
        observed.append({"system": name, "family_id": fid, "network": f["network"],
                         "route_group": f["route_group"],
                         "both_strict_pair": info[0][0] * info[1][0],
                         "both_unique_top1": info[0][1] * info[1][1],
                         "single_queries_correct": sum(q[1] for q in info),
                         "same_unique_winner": int(len(info[0][3]) == 1 and info[0][3] == info[1][3]),
                         "pair_margin_a": info[0][2], "pair_margin_b": info[1][2],
                         "query_A": info[0][4], "query_B": info[1][4]})
    return observed


def aggregate(rows):
    keys = ("both_strict_pair", "both_unique_top1", "single_queries_correct", "same_unique_winner")
    out = {key: sum(row[key] for row in rows) for key in keys}
    out["families"] = len(rows)
    out["query_outcomes"] = dict(Counter(row[side] for row in rows for side in ("query_A","query_B")))
    out["exact_pair_margin_ties"] = sum(row[side] == 0 for row in rows for side in ("pair_margin_a","pair_margin_b"))
    out["nonzero_pair_margins_below_1e_3"] = sum(
        0 < abs(row[side]) < .001 for row in rows for side in ("pair_margin_a","pair_margin_b"))
    out["pair_both_failed_full_pool"] = out["both_strict_pair"] - out["both_unique_top1"]
    assert out["pair_both_failed_full_pool"] >= 0
    return out


def compare(first, second, key="both_unique_top1", replications=20000):
    """Stratified by operator, with route-group clusters for this frozen panel."""
    by = {x["family_id"]: x for x in second}
    assert len(first) == len(by)
    strata = defaultdict(lambda: defaultdict(list))
    delta = []
    for x in first:
        diff = x[key] - by[x["family_id"]][key]
        strata[x["network"]][x["route_group"]].append(diff)
        delta.append(diff)
    grp = [[(sum(v),len(v)) for v in groups.values()] for groups in strata.values()]
    rng = random.Random(20261121)
    samples = []
    for _ in range(replications):
        draw = [rng.choice(src) for src in grp for _ in src]
        samples.append(sum(x for x,_ in draw) / sum(n for _,n in draw))
    samples.sort()
    q = [samples[int((len(samples)-1)*.025)], samples[int((len(samples)-1)*.975)]]
    return {"pp": 100*mean(delta), "interval_pp": [100*x for x in q],
            "gained": sum(x > 0 for x in delta), "lost": sum(x < 0 for x in delta),
            "bootstrap_groups": sum(map(len, grp)), "replications": replications}


def main():
    bench_path = ROOT / "benchmark.json"
    benchmark = json.loads(bench_path.read_text())
    pairs, docs = benchmark["pairs"], benchmark["docs"]
    scores = {"BM25-unigram": lexical_scores(pairs, docs, 1),
              "BM25-bigram": lexical_scores(pairs, docs, 2)}
    inspection = {}
    benchmark_hash = digest(bench_path)
    assert benchmark["protocol_sha256"] == digest(ROOT / "protocol.json")
    required = {(f["id"],i,d) for f in pairs for i in (0,1) for d in f["pool"]}
    for model in CHECKPOINTS:
        m = json.loads((ROOT/f"scores_{model}.json").read_text())
        assert m["benchmark_sha256"] == benchmark_hash
        inspection[model] = {}
        for style, payload in m["by_style"].items():
            s = {(fid,i,d): val for fid,i,d,val in payload["scores"]}
            assert set(s) == required and len(s) == payload["pair_count"]
            scores[f"{model}-{style}"] = s
            inspection[model][style] = {key:value for key,value in payload.items() if key != "scores"}
            if style == "plain":
                scores[f"{model}-opposite-score"] = {
                    (f,i,d): v - s[f,1-i,d] for (f,i,d),v in s.items()
                }
    outcomes = {name: family_outcomes(pairs,s,name) for name,s in scores.items()}
    sections = {}
    networks = sorted({f["network"] for f in pairs})
    for name, rows in outcomes.items():
        sections[name] = {"all":aggregate(rows), "by_network": {
            n:aggregate([x for x in rows if x["network"]==n]) for n in networks}}
    assert sections["BM25-unigram"]["all"]["exact_pair_margin_ties"] == 2*len(pairs)
    comparisons = {
        "M0_vs_BGE_plain": compare(outcomes["M0-plain"],outcomes["BGE-plain"]),
        "M1_vs_M0_plain": compare(outcomes["M1-plain"],outcomes["M0-plain"]),
        "M0_explicit_vs_plain": compare(outcomes["M0-explicit"],outcomes["M0-plain"]),
    }
    for m in CHECKPOINTS:
        comparisons[m + "_opposite_vs_plain"] = compare(outcomes[m+"-opposite-score"],outcomes[m+"-plain"])
    natural = {
        "queries": 2*len(pairs), "families": len(pairs),
        "queries_available_top5": sum(r["available_at_5"] for f in pairs for r in f["natural_bm25"]),
        "queries_hit_top1": sum(r["hits_at_1"] for f in pairs for r in f["natural_bm25"]),
        "families_both_available_top5": sum(all(r["available_at_5"] for r in f["natural_bm25"]) for f in pairs),
        "families_both_hit_top1": sum(all(r["hits_at_1"] for r in f["natural_bm25"]) for f in pairs),
    }
    report = {"benchmark_sha256": benchmark_hash, "secondary_protocol_sha256":digest(ROOT/"secondary_protocol.json"),
              "data": benchmark["audit"], "metrics": sections, "comparisons_full_pool_both": comparisons,
              "retrieval_only_natural_bm25": natural,
              "score_integrity": inspection}
    save(ROOT/"results.json",report)
    rows = [row for portion in outcomes.values() for row in portion]
    with (ROOT/"per_family.csv").open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"models": {k:v["all"] for k,v in sections.items()},
                      "paired":comparisons,"natural":natural}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
