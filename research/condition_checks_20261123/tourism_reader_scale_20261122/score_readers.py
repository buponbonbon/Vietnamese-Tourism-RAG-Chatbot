"""Rerank frozen reader candidate pools, adding verified evidence only for confirmed misses."""
from pathlib import Path
import json
import sys
import time
import gc

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'tourism_stage4_20261121'))
from score_stage4 import ARCHIVE, digest, infer, torch

MODELS={'BGE':'BGE_pretrained',**{f'{c}_{s}':f'{c}_fixed_1p0_weighted_matched_seed_{s}'
                                for c in ('M0','M1') for s in (42,123,999)}}


def prepare():
    frozen=ROOT/'reader_final.json'
    qs=json.loads((ROOT/'reader_worksheet.json').read_text())
    labels=json.loads((ROOT/'reader_judgments.json').read_text())
    assert set(labels)=={q['id'] for q in qs}
    docs={p['id']:p for p in json.loads((ROOT/'reader_corpus.json').read_text())}
    for q in qs:
        ratings=labels[q['id']]
        assert set(q['native'])==set(ratings)
        gold=[p for p in q['native'] if ratings[p]==2]
        q['native_sufficient']=gold
        if not gold:
            anchor=q['anchors'][0]
            assert anchor not in q['native'] and not any(v is None for v in ratings.values())
            q['oracle']=[*q['native'][:4],anchor]
            q['oracle_sufficient']=[anchor]
        else:
            q['oracle']=q['native']
            q['oracle_sufficient']=gold
        q['rating']=ratings
        q['candidate_ids']=list(dict.fromkeys(q['native']+q['oracle']))
    if not frozen.exists():
        with frozen.open('x') as out:
            json.dump(qs,out,ensure_ascii=False,indent=2)
        with (ROOT/'reader_freeze.sha256').open('x') as out:
            for name in ('reader_scale_protocol.md','reader_scale_plan.py','reader_corpus.json','reader_judgments.json','reader_final.json'):
                print(digest(ROOT/name),name,file=out)
    for line in (ROOT/'reader_freeze.sha256').read_text().splitlines():
        h,n=line.split()
        assert h==digest(ROOT/n)
    assert qs==json.loads(frozen.read_text())
    pairs=[(q['id'],p,q['query'],docs[p]['text']) for q in qs for p in q['candidate_ids']]
    return pairs,digest(frozen)


def main():
    pairs,checksum=prepare()
    print(f'{len(pairs)} query–passage pairs',flush=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    models={m['name']:m for m in json.loads(ARCHIVE.read_text())['models']}
    for alias,name in MODELS.items():
        out=ROOT/f'reader_scores_{alias}.json'
        if out.exists():
            old=json.loads(out.read_text())
            assert old['benchmark_sha256']==checksum and old['checkpoint']==name
            continue
        start=time.monotonic()
        r=infer(models[name],pairs)
        r['benchmark_sha256']=checksum
        with out.open('x') as f:
            json.dump(r,f,ensure_ascii=False,indent=2)
        print(alias,len(r['scores']),f'{time.monotonic()-start:.1f}s',flush=True)
        gc.collect()

if __name__=='__main__':main()
