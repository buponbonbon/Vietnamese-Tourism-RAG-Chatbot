"""Rerank the fixed follow-up reader questions on all preserved model variants."""
import gc
from pathlib import Path
import json
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'tourism_stage4_20261121'))
from score_stage4 import ARCHIVE, digest, infer, torch


def main():
    paths = [
        'stage4_extension_protocol.md','stage4_extension_plan.json',
        'stage4_extension_corpus.json','stage4_extension_worksheet.json','stage4_extension_ratings.json',
    ]
    with (ROOT/'stage4_extension_freeze.sha256').open('x') as manifest:
        for name in paths:
            print(digest(ROOT/name),name,file=manifest)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    qfile = ROOT/'stage4_extension_worksheet.json'
    questions = json.loads(qfile.read_text())['questions']
    corpus = {p['id']:p for p in json.loads((ROOT/'stage4_extension_corpus.json').read_text())}
    ratings = json.loads((ROOT/'stage4_extension_ratings.json').read_text())
    assert len(questions) == len(ratings)
    for q in questions:
        gold={pid for pid,grade in zip(q['native_ids'],ratings[q['id']],strict=True) if grade==2}
        assert gold == set(q['native_sufficient'])
    pairs=[(q['id'],p,q['query'],corpus[p]['text']) for q in questions for p in q['candidate_ids']]
    saved={r['name']:r for r in json.loads(ARCHIVE.read_text())['models']}
    checkpoints={'BGE':'BGE_pretrained'}
    for family in ('M0','M1'):
        for seed in (42,123,999):
            checkpoints[f'{family}_{seed}']=f'{family}_fixed_1p0_weighted_matched_seed_{seed}'
    for name, model in checkpoints.items():
        start=time.monotonic()
        scored=infer(saved[model],pairs)
        scored['benchmark_sha256']=digest(qfile)
        output=ROOT/f'stage4_extension_scores_{name}.json'
        with output.open('x') as dest:
            json.dump(scored,dest,indent=2,ensure_ascii=False)
        print(name,len(pairs),round(time.monotonic()-start,1),flush=True)
        gc.collect()


if __name__=='__main__':
    main()
