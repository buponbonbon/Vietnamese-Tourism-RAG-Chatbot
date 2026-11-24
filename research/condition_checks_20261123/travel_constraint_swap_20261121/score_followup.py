"""Score prespecified seeds and equal-stop pools, reusing exact inputs when available."""

import argparse
import json
from pathlib import Path
import time

from build_benchmark import ROOT,digest,save
from score_benchmark import MANIFEST, MODELS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment",choices=("seed","equal"))
    ap.add_argument("model",choices=MODELS)
    ap.add_argument("--seed",type=int,choices=(123,999))
    options = ap.parse_args()
    if options.experiment == "seed":
        assert options.model in ("M0","M1") and options.seed
        name = MODELS[options.model].removesuffix("42") + str(options.seed)
        source = ROOT/"benchmark.json"
        outname = f"scores_seed_{options.model}_{options.seed}.json"
        cache = {}
    else:
        assert options.seed is None
        name = MODELS[options.model]
        source = ROOT/"equal_stop.json"
        outname = f"scores_equal_{options.model}.json"
        original = json.loads((ROOT/f"scores_{options.model}.json").read_text())
        assert original["benchmark_sha256"] == digest(ROOT/"benchmark.json")
        cache = {(f,i,d):s for f,i,d,s in original["by_style"]["plain"]["scores"]}
    assert not (ROOT/outname).exists()
    benchmark = json.loads(source.read_text())
    assert benchmark["protocol_sha256"]==digest(ROOT/"protocol.json")
    if options.experiment=="equal":
        assert benchmark["followup_sha256"]==digest(ROOT/"followup_protocol.json")
        assert benchmark["original_benchmark_sha256"]==digest(ROOT/"benchmark.json")
    docs = benchmark["docs"]
    keys = [(p["id"],i,d) for p in benchmark["pairs"] for i in (0,1) for d in p["pool"]]
    qtext = {(p["id"],i):q for p in benchmark["pairs"] for i,q in enumerate(p["queries"])}
    todo = [k for k in keys if k not in cache]
    if options.experiment=="equal":
        assert len(todo)==benchmark["new_query_passage_scores"]
    scores = {k:cache[k] for k in keys if k in cache}
    new_meta = {"pairs":len(todo),"reused":len(scores)}

    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    meta = next(m for m in json.loads(MANIFEST.read_text())["models"] if m["name"]==name)
    ckpt = Path(meta["path"])
    for fn,h in meta["file_sha256"].items():
        assert digest(ckpt/fn)==h, fn
    print(outname,len(scores),"reused",len(todo),"new",flush=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt,local_files_only=True,trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt,torch_dtype=torch.float32,attn_implementation="sdpa",local_files_only=True,trust_remote_code=False).eval()
    to_encode = todo + [k for k in keys if k in scores][:4]
    enc = tokenizer([qtext[f,i] for f,i,d in to_encode],[docs[d]["plain"] for f,i,d in to_encode],
                    padding=False,truncation=False)
    length = [len(x) for x in enc["input_ids"]]
    assert max(length)<=512

    def infer(ids):
        padded = tokenizer.pad({k:[v[i] for i in ids] for k,v in enc.items()},padding=True,return_tensors="pt")
        with torch.inference_mode():
            logits = model(**padded).logits.flatten()
        assert torch.isfinite(logits).all()
        return logits.tolist()

    order = sorted(range(len(todo)),key=length.__getitem__)
    tick = time.monotonic()
    pairs = [(todo[i],s) for start in range(0,len(order),16)
             for i,s in zip(order[start:start+16],infer(order[start:start+16]))]
    scores.update(pairs)
    cached_ids = list(range(len(todo),len(to_encode)))
    if cached_ids:
        max_error = max(abs(scores[to_encode[i]]-s) for i,s in zip(cached_ids,infer(cached_ids)))
        assert max_error<1e-3
        new_meta["cache_probe_max_abs_delta"] = max_error
    assert set(scores)==set(keys)
    save(ROOT/outname, {"source_sha256":digest(source),"checkpoint":name,"checkpoint_files":meta["file_sha256"],
                        "details":new_meta,"scores":[[*k,scores[k]] for k in keys]})
    print(outname, "complete",round(time.monotonic()-tick,2),"seconds for new scores",flush=True)


if __name__=="__main__":
    main()
