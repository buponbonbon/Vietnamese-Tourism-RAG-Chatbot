"""Cross-check reference IDs and winners without the analysis routine."""

import csv
from pathlib import Path
import xml.etree.ElementTree as ET

from prepare_travel import ROOT, load, save, sha

r = load("TRAVEL_RESULTS.json")
n = r["cohort"]["eligible"]
rank = load("RANKS.json")["text"]
for name,h in r["inputs_sha256"].items():
    assert sha(ROOT/name) == h
assert sha(ROOT/"Posts.xml") == r["cohort"]["posts_sha256"]
questions = {}
parents = {}
for _, row in ET.iterparse(ROOT/"Posts.xml", events=("end",)):
    if row.tag != "row":
        continue
    kind = row.get("PostTypeId")
    key = row.get("Id")
    if kind == "1" and key in rank:
        assert "|vietnam|" in row.get("Tags", "")
        assert row.get("ClosedDate") is None
        questions[key] = int(row.get("AcceptedAnswerId"))
    if kind == "2":
        parents[int(key)] = row.get("ParentId")
    row.clear()
assert len(questions) == len(rank) == n
assert all(r["gold"] == questions[q] and parents[r["gold"]] == q for q,r in rank.items())
with (ROOT/"TRAVEL_QUERY_OUTCOMES.csv").open(newline="") as f:
    records = {row["qid"]:row for row in csv.DictReader(f)}
assert set(records) == set(questions)
logits = {}
for kind in ("BGE", "M0"):
    logits[kind] = {(str(x["q"]),x["d"]):x["logit"] for x in load(kind+"_scores.json")["rows"]}
for method in ("bm25","dense","rrf"):
    count = {k:0 for k in ("raw","BGE","M0")}
    hits5 = 0
    for q, g in questions.items():
        pool = rank[q][method][:5]
        hits5 += g in pool
        assert int(records[q][method+"_available"]) == (g in pool)
        for name in count:
            if name == "raw":
                expected = pool[0]
            else:
                s = [logits[name][q,d] for d in pool]
                expected = pool[s.index(sorted(s)[-1])]
            assert expected == int(records[q][f"{method}_{name}_winner"])
            assert (expected == g) == bool(int(records[q][f"{method}_{name}_hit"]))
            count[name] += expected == g
    assert r["retrieval"]["text"][method]["5"] == hits5
    for kind,c in count.items():
        cell = r["results"][method][kind]
        assert cell["hit1"] == c and cell["availability5"] == hits5
        assert cell["accepted_absent"] + cell["accepted_present_not_selected"] + c == n
assert r["primary"]["first_only"] == sum(x["rrf_M0_hit"]>x["bm25_M0_hit"] for x in records.values())
assert r["primary"]["second_only"] == sum(x["rrf_M0_hit"]<x["bm25_M0_hit"] for x in records.values())
save("VERIFIED.json", {"status":"PASS", "n":n, "checks":["archive posts hash",
      "independent question-tag accepted-answer parent checks","all candidate-budget outcomes",
      "independent accepted-answer counts and matched primary discordants"],
      "results_sha256":sha(ROOT/"TRAVEL_RESULTS.json")})
print("PASS", n, "reference IDs, parent threads, reranked selections and paired discordants")
