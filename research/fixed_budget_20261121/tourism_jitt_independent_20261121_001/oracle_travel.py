"""Inject the held-out forum reference at fixed candidate budget."""

import gc
import json
import math
from pathlib import Path
import time

from run_travel import ARCHIVE, MODELS, ROOT, load, parse, save, sha
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    docs, qs, audit = parse()
    ranks = load("RANKS.json")["text"]
    assert audit == load("CORPUS_AUDIT.json")
    absent = [q for q in qs if q["gold"] not in ranks[str(q["qid"])]["bm25"][:5]]
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    legacy = json.loads((ARCHIVE / "protocol.json").read_text())["models"]
    inputs = [ROOT/k for k in ("RANKS.json","PROTOCOL.json","ORACLE_PROTOCOL.json",
                              "BGE_scores.json","M0_scores.json","oracle_travel.py")]
    result = {"n":len(absent),"models":{},
              "inputs_sha256":{p.name:sha(p) for p in inputs}}
    for short,name in MODELS.items():
        raw = load(short+"_scores.json")
        scores = {(x["q"],x["d"]):x["logit"] for x in raw["rows"]}
        logit = dict(scores)
        new = [q for q in absent if (q["qid"],q["gold"]) not in logit]
        if new:
            info=next(x for x in legacy if x["name"]==name)
            d=Path(info["path"])
            assert info["file_sha256"]==raw["checkpoint"]
            assert all(sha(d/k)==v for k,v in info["file_sha256"].items())
            tok=AutoTokenizer.from_pretrained(d,local_files_only=True,trust_remote_code=False)
            model=AutoModelForSequenceClassification.from_pretrained(
                d,local_files_only=True,trust_remote_code=False,dtype=torch.float32,
                attn_implementation="sdpa").eval()
            for i in range(0,len(new),4):
                b=new[i:i+4]
                batch=tok([r["text"] for r in b],[docs[r["gold"]][2] for r in b],
                          truncation=True,max_length=512,padding=True,return_tensors="pt")
                with torch.inference_mode():
                    out=model(**batch).logits.float().reshape(-1).tolist()
                for r,x in zip(b,out,strict=True):
                    assert math.isfinite(x)
                    logit[r["qid"],r["gold"]]=x
            del model
            gc.collect()
        detail=[]
        for q in absent:
            qid,gold=q["qid"],q["gold"]
            pool=[*ranks[str(qid)]["bm25"][:4],gold]
            s=[logit[qid,d] for d in pool]
            win=pool[max(range(5),key=s.__getitem__)]
            assert len(set(pool))==5 and len(s)==5
            detail.append({"qid":qid,"gold":gold,"winner":win,"pool":pool,"score":s})
        n_hit=sum(x["winner"]==x["gold"] for x in detail)
        assert n_hit==sum(x["score"][-1]>max(x["score"][:-1]) for x in detail)
        result["models"][short]={"new_pairs":len(new),"selected":n_hit,"detail":detail}
        print(short,"selected",n_hit,"/",len(absent),"new pairs",len(new),flush=True)
    save("ORACLE_RESULTS.json",result)


if __name__=="__main__":
    main()
