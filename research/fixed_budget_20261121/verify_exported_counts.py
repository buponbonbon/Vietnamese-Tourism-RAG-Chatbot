"""Verify headline tables from the de-identified artifact ZIP using stdlib only."""

import hashlib
import json
from pathlib import Path
import sys


BASE = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(__file__).resolve().parent


def read(name):
    return json.loads((BASE / name).read_text())


def check_hashes():
    for name, sha in read("MANIFEST.json").items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == sha, name


def tourism():
    root = "tourism_lexical/"
    r = read(root + "RESULTS.json")
    ranks = read(root + "RANKS.json")
    g = {(x["query_id"],x["chunk_id"]):x["grade"] == 2
         for x in read(root + "KNOWN_GRADES.json")}
    n = len(ranks["legacy"])
    m0 = read(root + "M0_fixed_1p0_weighted_matched_seed_42_scores.json")["scores"]
    bge = read(root + "BGE_pretrained_scores.json")["scores"]
    for arm in ranks:
        pools = {q: r[:5] for q,r in ranks[arm].items()}
        availability = [
            sum(any(g.get((q,d),fill) for d in p) for q,p in pools.items()) for fill in (False,True)]
        for model,rows in [("M0",m0),("BGE",bge),("Raw",None)]:
            s = {(x["q"],x["d"]):x["s"] for x in rows} if rows else {}
            winners = {q:max(p,key=lambda d:s[q,d]) if s else p[0] for q,p in pools.items()}
            hit1 = [sum(g.get((q,d),fill) for q,d in winners.items()) for fill in (False,True)]
            assert r["cells"][arm+"_"+model]["availability"] == availability
            assert r["cells"][arm+"_"+model]["hit1"] == hit1
    print("PASS tourism fixed-budget tables", n)


def reference(name, ranks, reference_result, model_rows, columns):
    qkey,dkey,skey = columns
    score = {m:{(str(x[qkey]),x[dkey]):x[skey] for x in rows}
             for m,rows in model_rows.items()}
    cells = {}
    for arm in ("bm25","dense","rrf"):
        hv = sum(row["gold"] in row[arm][:5] for row in ranks.values())
        cells[arm] = {}
        for model in score:
            hits = sum(max(r[arm][:5],key=lambda d:score[model][q,d])==r["gold"] for q,r in ranks.items())
            assert reference_result[arm][model] == [hv,hits]
            cells[arm][model] = hits
    print("PASS",name,"reference availability and rank 1",cells)


def travel():
    r=read("travel_forum/TRAVEL_RESULTS.json")
    ranks=read("travel_forum/RANKS.json")["text"]
    ref={a:{m:[c["availability5"],c["hit1"]] for m,c in models.items()} for a,models in r["results"].items()}
    scores={m:read("travel_forum/"+m+"_scores.json")["rows"] for m in ("BGE","M0")}
    reference("travel forum",ranks,ref,scores,("q","d","logit"))
    oracle=read("travel_forum/ORACLE_RESULTS.json")
    for m,c in oracle["models"].items():
        assert len(c["detail"])==oracle["n"]
        for x in c["detail"]:
            p=ranks[str(x["qid"])]
            assert x["gold"]==p["gold"] and x["gold"] not in p["bm25"][:5]
            assert x["pool"]==p["bm25"][:4]+[p["gold"]]
            assert x["winner"]==x["pool"][x["score"].index(max(x["score"]))]
        assert sum(x["winner"]==x["gold"] for x in c["detail"])==c["selected"]
    print("PASS travel forum fixed-budget reference substitution")


def mlqa():
    r=read("mlqa/EXTERNAL_RESULTS.json")
    ranks=read("mlqa/RANKS.json")
    ids=read("mlqa/COHORT.json")["ids"]
    for arm in ("bm25","dense","rrf"):
        for k in (1,5,10,20,50):
            assert sum(x["gold"] in x[arm][:k] for x in ranks.values())==r["all_query_availability"][arm][f"at_{k}"]
    ref={a:{m:[c["available"],c["gold_hit1"]] for m,c in x["models"].items()}
         for a,x in r["reranked_cohort"].items()}
    files={"BGE":"BGE_pretrained","M0":"M0_fixed_1p0_weighted_matched_seed_42"}
    scores={m:read("mlqa/"+filename+"_scores.json")["scores"] for m,filename in files.items()}
    reference("Vietnamese MLQA", {q:ranks[q] for q in ids}, ref,scores,("q","d","score"))


if __name__=="__main__":
    check_hashes()
    tourism()
    travel()
    mlqa()
