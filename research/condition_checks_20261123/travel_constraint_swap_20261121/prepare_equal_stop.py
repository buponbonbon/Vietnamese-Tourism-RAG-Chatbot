"""Create follow-up pools at equal passage stop count, without using any logits."""

import json

from build_benchmark import BM25, ROOT, digest, key, save


def main():
    b = json.loads((ROOT / "benchmark.json").read_text())
    groups = {}
    for network in sorted({p["network"] for p in b["pairs"]}):
        d = {ident: doc for ident,doc in b["docs"].items() if doc["network"] == network}
        groups[network] = d, BM25(d)
    fresh = []
    excluded = []
    changed = 0
    new_counts = 0
    for p in b["pairs"]:
        docs, bm25 = groups[p["network"]]
        stops = len(docs[p["gold"][0]]["stops"])
        assert len(docs[p["gold"][1]]["stops"]) == stops
        ends = set(p["directed_endpoints"][0])
        candidates = [d for d,doc in docs.items() if len(doc["stops"])==stops
                      and (doc["route"]!=p["route"] or not ends.issubset(doc["stops"]))]
        values = bm25.rank(p["queries"][0])
        ranked = sorted(candidates,key=lambda d:(-values.get(d,0.),d))
        if len(ranked)<3:
            excluded.append(p["id"])
            continue
        pool = sorted([*p["gold"],*ranked[:3]],key=lambda d:key("equal-stop-order",p["id"],d))
        assert len(pool)==len(set(pool))==5
        assert all(len(docs[d]["stops"])==stops for d in pool)
        changed += set(pool) != set(p["pool"])
        new_counts += 2*len(set(pool)-set(p["pool"]))
        fresh.append({**p,"pool":pool})
    save(ROOT/"equal_stop.json", {**b,"pairs":fresh, "original_benchmark_sha256":digest(ROOT/"benchmark.json"),
                                  "followup_sha256":digest(ROOT/"followup_protocol.json"),
                                  "excluded_family_ids":excluded,
                                  "families_changed":changed, "new_query_passage_scores":new_counts})
    print("Equal-stop pools",len(fresh),"changed",changed,"excluded",len(excluded),"new scores per model",new_counts)


if __name__=="__main__":
    main()
