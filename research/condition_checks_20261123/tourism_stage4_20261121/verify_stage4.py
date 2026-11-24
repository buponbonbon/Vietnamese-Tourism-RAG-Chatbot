"""Independently verify score coverage, frozen labels and all reported rank-one counts."""
import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT/name).read_bytes())

def first_status(pool, grade, mapped):
    ranked=sorted(pool,key=lambda p:mapped[p],reverse=True)
    if mapped[ranked[0]]==mapped[ranked[1]]:
        return 'tie'
    return {1:'yes',0:'no',None:'unknown'}[grade[ranked[0]]]

def wins(rows):
    lo=rows.count('yes')
    hi=lo+rows.count('unknown')
    return (lo,hi,len(rows))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-rebuild',action='store_true')
    args=parser.parse_args()
    for expected, name in (x.split() for x in (ROOT/'freeze.sha256').read_text().splitlines()):
        assert sha256((ROOT/name).read_bytes()).hexdigest()==expected, name
    manifest = read('benchmark.json')
    result=read('results.json')
    assert result['benchmark_sha256']==sha256((ROOT/'benchmark.json').read_bytes()).hexdigest()
    qs={q['id']:q for q in manifest['questions']}
    pool={row['id'] for row in read('corpus.json')}
    planned={(q['id'],p) for q in qs.values() for p in q['candidate_ids']}
    with (ROOT/'adjudicated_native.tsv').open() as f:
        annotations={(x['id'],x['pid']): x['grade'] for x in csv.DictReader(f,delimiter='\t')}
    assert len(annotations)==5*len(qs)
    for q in qs.values():
        assert set(q['native']) | set(q['active']) == set(q['candidate_ids']) <= pool
        for p in q['native']:
            grade=q['native_grades'][p]
            assert annotations[q['id'],p]==('unknown' if grade is None else str(grade))
        if q['condition']=='oracle':
            assert list(q['native_grades'].values())==[0]*5
            assert q['active']==q['native'][:4]+[q['anchor']]
            assert q['active_grades'][q['anchor']]==1
        else:
            assert q['active']==q['native']
    for path in sorted(ROOT.glob('scores_*.json')):
        checkpoint=path.stem.removeprefix('scores_')
        entry=read(path.name)
        assert entry['benchmark_sha256']==result['benchmark_sha256']
        measured={(q,p):s for q,p,s in entry['scores']}
        assert set(measured)==planned and len(measured)==len(entry['scores'])
        assert all(math.isfinite(v) for v in measured.values())
        for cohort in ('new','bridge'):
            raw=[]
            known=[]
            oracles=[]
            pairs=0
            lost_low=0
            lost_high=0
            for q in qs.values():
                if q['cohort']!=cohort:
                    continue
                scores={p:measured[q['id'],p] for p in q['candidate_ids']}
                n=first_status(q['native'],q['native_grades'],scores)
                a=first_status(q['active'],q['active_grades'],scores)
                raw.append(n)
                if q['availability']=='present':
                    known.append(n)
                if q['condition']=='oracle':
                    oracles.append(a)
                if q['nested'] and scores[q['nested'][0]]>scores[q['nested'][1]]:
                    pairs+=1
                    lost_low+=a in ('no','tie')
                    lost_high+=a in ('no','tie','unknown')
            model=result['outcomes'][cohort]['models'][checkpoint]
            for name,values in [('native_hit1',raw),('native_hit1_on_confirmed_present',known),('oracle_hit1',oracles)]:
                stats=model[name]
                assert wins(values)==tuple(stats[k] for k in ('lower','upper','n'))
            nested=model['nested_comparison']
            assert (pairs,lost_low,lost_high)==tuple(nested[k] for k in ('k2_success','displaced_lower','displaced_upper'))
        print('PASS',checkpoint)
    if args.source_rebuild:
        from corpus_stage4 import documents
        from stage4_pools import build
        assert documents()==read('corpus.json')
        assert build()==read('native_pools.json')
        print('PASS corpus and native pool rebuild from archived sources')
    print('PASS frozen annotator decisions, substitution invariants and model outcome counts')

if __name__=='__main__':
    main()
