"""Recalculate the main and robustness result counts directly from frozen logits."""

from pathlib import Path
import json
from hashlib import file_digest

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT/name).read_text())


def digest(name):
    with (ROOT/name).open("rb") as f:
        return file_digest(f,"sha256").hexdigest()


def counts(pairs, score_rows, signed=False):
    values = {(f,d,direction):score for f,direction,d,score in score_rows}
    need = {(f["id"],d,i) for f in pairs for d in f["pool"] for i in (0,1)}
    assert len(values)==len(score_rows) and values.keys()==need
    hit_mirror = hit_pool = 0
    for f in pairs:
        fid = f["id"]
        subsets = []
        for i in (0,1):
            s = {d:(values[fid,d,i]-values[fid,d,1-i] if signed else values[fid,d,i])
                 for d in f["pool"]}
            ranked = sorted(f["pool"],key=lambda d:s[d],reverse=True)
            gold = f["gold"][i]
            subsets.append((s[gold]>s[f["gold"][1-i]],ranked[0]==gold and s[ranked[0]]>s[ranked[1]]))
        hit_mirror+=all(x[0] for x in subsets)
        hit_pool+=all(x[1] for x in subsets)
    return hit_mirror,hit_pool


def verify(pairs, raw, summary, signed=False):
    assert counts(pairs,raw,signed) == (summary["both_strict_pair"],summary["both_unique_top1"])


def main():
    b=read("benchmark.json")
    r=read("results.json")
    ext=read("equal_stop.json")
    out=read("followup_results.json")
    assert r["benchmark_sha256"]==out["original_sha256"]==digest("benchmark.json")
    assert out["equal_stop_sha256"]==digest("equal_stop.json")
    assert out["followup_protocol_sha256"]==digest("followup_protocol.json")
    for name in ("BGE","M0","M1"):
        a=read(f"scores_{name}.json")
        for fmt,frozen in a["by_style"].items():
            verify(b["pairs"],frozen["scores"],r["metrics"][name+"-"+fmt]["all"])
        verify(b["pairs"],a["by_style"]["plain"]["scores"],r["metrics"][name+"-opposite-score"]["all"],True)
        verify(ext["pairs"],read(f"scores_equal_{name}.json")["scores"],out["equal_stop_runs"][name])
        for seed,stat in out["seed_runs"][name].items():
            points=(a["by_style"]["plain"]["scores"] if seed=="42"
                    else read(f"scores_seed_{name}_{seed}.json")["scores"])
            verify(b["pairs"],points,stat)
    print("PASS all 14 reranker result tables from raw logits and frozen candidates.")


if __name__=="__main__":
    main()
