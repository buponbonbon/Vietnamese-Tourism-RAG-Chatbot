"""Independently reconstruct lexical ranks, winners and missing-grade bounds."""

import csv
import itertools
import math
import re
from collections import Counter, defaultdict
from fractions import Fraction

from run_control import BGE, M0, ROOT, external, prior


def check_ranks(questions, docs, ranks):
    query_terms = {}
    for q in questions:
        query_terms[q["benchmark_id"]] = [
            t for t in re.findall(r"[^\W_]+", q["query"].lower()) if t not in external.STOP
        ]
    vocabulary = set(itertools.chain.from_iterable(query_terms.values()))
    doc_names = list(docs)
    lengths = [len(text.split()) for text in docs.values()]
    average = sum(lengths) / len(doc_names)
    term_rows = defaultdict(dict)
    for i, text in enumerate(docs.values()):
        for term, tf in Counter(re.findall(r"[^\W_]+", text.lower())).items():
            if term in vocabulary:
                term_rows[term][i] = tf
    for q, terms in query_terms.items():
        scores = defaultdict(float)
        for term in terms:
            docs_for_term = term_rows.get(term, {})
            df = len(docs_for_term)
            idf = math.log((len(doc_names) - df + .5) / (df + .5) + 1.)
            for i, tf in docs_for_term.items():
                scores[i] += idf * (tf * 2.5 / (tf + 1.5 * (.25 + .75 * lengths[i] / average)))
        order = sorted(range(len(doc_names)), key=lambda i: (-scores.get(i, 0.0), i))[:50]
        assert [doc_names[i] for i in order] == ranks["unicode"][q], q
        for lexical, fusion in [("legacy", "rrf_legacy"), ("unicode", "rrf_unicode")]:
            a = {d: i for i, d in enumerate(ranks[lexical][q], 1)}
            b = {d: i for i, d in enumerate(ranks["dense"][q], 1)}
            fused = {d: sum(Fraction(1, 60+r[d]) for r in (a, b) if d in r) for d in a.keys() | b.keys()}
            order = sorted(fused, key=lambda d: (-fused[d], a.get(d,51), b.get(d,51), d))
            assert order[:50] == ranks[fusion][q], (q, fusion)


def enumerate_states(pools, winners, grades):
    joined = set().union(*pools)
    known_grade2 = {d for d in joined if grades.get(d) == 2}
    unjudged = sorted(joined - grades.keys())
    possible = set()
    for bits in itertools.product((False, True), repeat=len(unjudged)):
        grade2 = known_grade2 | {d for d, on in zip(unjudged, bits, strict=True) if on}
        possible.add(tuple((int(bool(p & grade2)), int(w in grade2))
                           for p, w in zip(pools, winners, strict=True)))
    return possible


def main():
    _, questions, docs, legacy, known, _, _ = prior.load()
    result = prior.read(ROOT / "RESULTS.json")
    for name, sha256 in result["input_files"].items():
        assert prior.sha(ROOT.parent / name) == sha256
    ranks = prior.read(ROOT / "RANKS.json")
    assert ranks["legacy"] == legacy
    check_ranks(questions, docs, ranks)
    qids = list(legacy)
    scores = {}
    for short, name in [("BGE", BGE), ("M0", M0)]:
        raw = prior.read(ROOT / (name + "_scores.json"))
        scores[short] = {(r["q"], r["d"]): r["s"] for r in raw["scores"]}
        assert raw["cache_probe_max_abs_error"] <= 1e-4
        assert raw["dense_cache_overlap_max_abs_error"] <= 1e-4

    with (ROOT / "PER_QUERY.csv").open(newline="") as f:
        records = {(r["qid"], r["arm"], r["model"]):r for r in csv.DictReader(f)}
    per_query_grades = defaultdict(dict)
    for (q, d), v in known.items():
        per_query_grades[q][d] = v
    for arm, q_ranks in ranks.items():
        for mode in ("Raw", "BGE", "M0"):
            minmax = defaultdict(lambda: [0,0])
            for q, p in q_ranks.items():
                pool = p[:5]
                want = (sorted(enumerate(pool), key=lambda r: (-scores[mode][q, r[1]], r[0]))[0][1]
                        if mode in scores else pool[0])
                row = records[q, arm, mode]
                assert row["winner"] == want
                res = enumerate_states([set(pool)], [want], per_query_grades[q])
                a, y, absent, failed = zip(*[(a, y, 1-a, a-y) for ((a,y),) in res])
                for key, values in [("availability",a),("hit1",y),("absence",absent),("available_but_missed",failed)]:
                    minmax[key][0] += min(values)
                    minmax[key][1] += max(values)
                assert int(row["available_min"]) == min(a) and int(row["available_max"]) == max(a)
            for key, v in minmax.items():
                assert v == result["cells"][arm + "_" + mode][key], (arm, mode, key, v)

    for contrast, bounds in result["paired_availability_count_bounds"].items():
        a, b = contrast.split("_minus_")
        for mode in ("M0", "BGE", "Raw"):
            a_bounds, y_bounds = [0,0], [0,0]
            for q in qids:
                pools = [set(ranks[arm][q][:5]) for arm in (a,b)]
                winners = [records[q,arm,mode]["winner"] for arm in (a,b)]
                states = enumerate_states(pools, winners, per_query_grades[q])
                new_a, new_y = zip(*[(x-t, y-z) for ((x,y),(t,z)) in states])
                for bb, vals in [(a_bounds,new_a),(y_bounds,new_y)]:
                    bb[0] += min(vals)
                    bb[1] += max(vals)
            assert a_bounds == bounds
            assert y_bounds == result["paired_hit1_count_bounds"][contrast+"_"+mode]["bounds_count"]
    prior.save(ROOT / "VERIFIED.json", {"status": "PASS",
              "checks": ["all 140 Unicode BM25 ranks from independent sparse accumulation",
                         "both RRF ranks via exact fractions", "all 2100 winner and error-bound rows",
                         "paired contrasts via exhaustive missing-label assignments",
                         "manifest hashes and cache agreement"],
              "results_sha256": prior.sha(ROOT / "RESULTS.json")})
    print("PASS: lexical retrieval, RRF, all winners, missing-label and paired bounds")


if __name__ == "__main__":
    main()
