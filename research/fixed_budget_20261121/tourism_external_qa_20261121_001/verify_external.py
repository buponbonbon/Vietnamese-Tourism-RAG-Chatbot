"""Check official gold mapping, the sample, rank fusion and logit outcomes."""

import csv
import hashlib
import json
import math
import zipfile
from fractions import Fraction
from pathlib import Path

from run_external import ROOT, choose_cohort, digest, load, load_source, save

protocol = load("PROTOCOL.json")
result = load("EXTERNAL_RESULTS.json")
cohort = load("COHORT.json")
ranks = load("RANKS.json")
for name, expected in result["inputs_sha256"].items():
    assert digest(ROOT / name) == expected, name
assert cohort["protocol_sha256"] == digest(ROOT / "PROTOCOL.json")
docs, questions = load_source()
ids, audit = choose_cohort(docs, questions)
assert ids == cohort["ids"] and audit == cohort["audit"]
assert len(ids) == 140

official = {}
with zipfile.ZipFile(ROOT / "MLQA_V1.zip") as f:
    article_data = json.load(f.open(protocol["member"]))["data"]
    idx = 0
    for article in article_data:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                assert qa["id"] not in official
                assert any(paragraph["context"].startswith(a["text"], a["answer_start"])
                           for a in qa["answers"] if a["text"])
                official[qa["id"]] = idx
            idx += 1
assert idx == len(docs)
assert set(official) == set(ranks)
for q in ranks:
    record = ranks[q]
    assert official[q] == record["gold"]
    bm = {d: i for i, d in enumerate(record["bm25"], 1)}
    dense = {d: i for i, d in enumerate(record["dense"], 1)}
    points = dict.fromkeys(bm.keys() | dense.keys(), Fraction(0))
    for ordering in (bm, dense):
        for d, rank in ordering.items():
            points[d] += Fraction(1, 60 + rank)
    reranked = sorted(points, key=lambda d: (-points[d], bm.get(d, 51), dense.get(d, 51), d))
    assert reranked[:50] == record["rrf"]

unicode_bm25 = load("BM25_UNICODE.json")
for run in (*("bm25", "dense", "rrf"), "bm25_unicode"):
    for k in (1, 5, 10, 20, 50):
        hits = sum(official[q] in (unicode_bm25[q] if run == "bm25_unicode" else item[run])[:k]
                   for q, item in ranks.items())
        assert hits == result["all_query_availability"][run][f"at_{k}"]

with (ROOT / "EXTERNAL_QUERY_OUTCOMES.csv").open(newline="") as f:
    outcomes = {row["qid"]: row for row in csv.DictReader(f)}
assert set(outcomes) == set(ids)
for kind, name in zip(("BGE", "M0"), protocol["reranker"], strict=True):
    rows = load(f"{name}_scores.json")["scores"]
    assert len(rows) == result["distinct_scored_pairs_per_model"]
    assert all(math.isfinite(row["score"]) for row in rows)
    scores = {(row["q"], row["d"]): row["score"] for row in rows}
    tokens = {(row["q"], row["d"]): row["tokens"] for row in rows}
    for arm in ("bm25", "dense", "rrf"):
        selected = 0
        available = 0
        token_sum = 0
        for q in ids:
            pool = ranks[q][arm][:5]
            s = [scores[q, d] for d in pool]
            answer = pool[s.index(sorted(s)[-1])]
            assert answer == int(outcomes[q][f"{arm}_{kind}_winner"])
            assert (answer == official[q]) == bool(int(outcomes[q][f"{arm}_{kind}_hit"]))
            selected += answer == official[q]
            available += official[q] in pool
            token_sum += sum(tokens[q, d] for d in pool)
        summary = result["reranked_cohort"][arm]["models"][kind]
        assert summary["available"] == available
        assert summary["gold_hit1"] == selected
        assert summary["absent"] + summary["present_not_selected"] + selected == len(ids)
        assert summary["inference_pairs"] == 5 * len(ids)
        assert summary["inference_unpadded_tokens"] == token_sum

deltas = [int(outcomes[q]["rrf_M0_hit"]) - int(outcomes[q]["bm25_M0_hit"]) for q in ids]
assert sum(deltas) == result["primary"]["net_hits"]
assert sum(v == 1 for v in deltas) == result["primary"]["a_only"]
assert sum(v == -1 for v in deltas) == result["primary"]["b_only"]
save("VERIFIED.json", {
    "status": "PASS",
    "all_questions": len(official),
    "reranked_questions": len(ids),
    "checks": ["source SHA256 and official spans", "cohort reconstruction and exact overlap",
               "all RRF ranks by exact fractions", "rank metric counts", "all CE winners, tokens and error counts",
               "paired primary counts"],
    "results_sha256": digest(ROOT / "EXTERNAL_RESULTS.json")
})
print("PASS: official source, ranks, fusion, reranking, costs, paired counts")
