"""Summarize all prespecified follow-up seeds and equal-stop pools."""

import csv
import json
from pathlib import Path
from statistics import mean

from analyze_benchmark import aggregate, compare, family_outcomes
from build_benchmark import ROOT, digest, save


def load(path):
    return json.loads(path.read_text())


def verified_scores(dataset, score_path):
    src = load(score_path)
    assert src["source_sha256"] == digest(dataset)
    s = {(f,i,d):v for f,i,d,v in src["scores"]}
    assert len(s) == len(src["scores"])
    return s


def main():
    primary = ROOT/"benchmark.json"
    b = load(primary)
    alt = ROOT/"equal_stop.json"
    equal = load(alt)
    assert equal["original_benchmark_sha256"]==digest(primary)
    assert equal["followup_sha256"]==digest(ROOT/"followup_protocol.json")
    summary = {"original_sha256":digest(primary), "equal_stop_sha256":digest(alt),
               "followup_protocol_sha256":digest(ROOT/"followup_protocol.json"),
               "equal_stop_families_changed":equal["families_changed"],
               "equal_stop_excluded":len(equal["excluded_family_ids"]),
               "seed_runs":{}, "equal_stop_runs":{}, "comparisons":{}}
    primary_rows = {}
    seed_rows = {}
    for model in ("BGE","M0","M1"):
        m = load(ROOT/f"scores_{model}.json")
        assert m["benchmark_sha256"] == digest(primary)
        original = {(f,i,d):v for f,i,d,v in m["by_style"]["plain"]["scores"]}
        primary_rows[model] = family_outcomes(b["pairs"],original,f"{model}-42")
        altered = verified_scores(alt,ROOT/f"scores_equal_{model}.json")
        overlap=original.keys() & altered.keys()
        assert all(original[k]==altered[k] for k in overlap)
        trial = family_outcomes(equal["pairs"], altered, f"{model}-equal-stop")
        summary["equal_stop_runs"][model]=aggregate(trial)
        seed_rows[model] = {42:primary_rows[model]}
        if model in ("M0","M1"):
            for seed in (123,999):
                s=verified_scores(primary,ROOT/f"scores_seed_{model}_{seed}.json")
                outcomes=family_outcomes(b["pairs"],s,f"{model}-{seed}")
                seed_rows[model][seed]=outcomes
        summary["seed_runs"][model]={str(seed):aggregate(rows) for seed,rows in seed_rows[model].items()}
        same_ids={p["id"] for p in equal["pairs"]}
        original_subset=[row for row in primary_rows[model] if row["family_id"] in same_ids]
        summary["comparisons"][f"{model}_equal_stop_minus_original"] = compare(trial,original_subset)

    for model in ("M0","M1"):
        assert all([x["family_id"] for x in rows] == [x["family_id"] for x in seed_rows[model][42]]
                   for rows in seed_rows[model].values())
    for seed in (42,123,999):
        summary["comparisons"][f"M1_minus_M0_seed_{seed}"]=compare(seed_rows["M1"][seed],seed_rows["M0"][seed])
    per_model = {}
    for m in ("M0","M1"):
        source = seed_rows[m]
        per_model[m]=[
            {**row,"both_unique_top1":mean(source[seed][i]["both_unique_top1"] for seed in (42,123,999))}
            for i,row in enumerate(source[42])
        ]
    summary["comparisons"]["M1_minus_M0_fixed_seed_averaged"] = compare(per_model["M1"],per_model["M0"])
    save(ROOT/"followup_results.json",summary)
    print(json.dumps({"equal_stop":{k:v["both_unique_top1"] for k,v in summary["equal_stop_runs"].items()},
                      "seeds":{k:{i:[s["both_strict_pair"],s["both_unique_top1"]] for i,s in x.items()}
                               for k,x in summary["seed_runs"].items()},
                      "span":summary["comparisons"]},indent=2))


if __name__=="__main__":
    main()
