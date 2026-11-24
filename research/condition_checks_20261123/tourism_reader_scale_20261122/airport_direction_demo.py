"""Illustrate the preregistered direction distinction on a separately documented reader case."""
from pathlib import Path
import json
import re

ROOT=Path(__file__).resolve().parent
PATTERN=re.compile(r'\btừ (?:ga |nhà ga )?(T[123]) (?:sang|qua|đến) (?:ga |nhà ga )?(T[123])\b',re.I)


def destinations(text):
    return {(o.upper(),d.upper()) for o,d in PATTERN.findall(text)}


def main():
    reader=next(q for q in json.loads((ROOT/'reader_final.json').read_text()) if q['id']=='U03')
    corpus={p['id']:p for p in json.loads((ROOT/'reader_corpus.json').read_text())}
    assert 'quá cảnh ở TPHCM' in reader['query'] and 'T1-T2 qua T3' in reader['query']
    required={('T1','T3'),('T2','T3')}
    reverse={(right,left) for left,right in required}
    scope={}
    for pid in reader['oracle']:
        title=corpus[pid]['header']
        relations=destinations(title)
        scope[pid]={'relations':sorted(relations),'demoted':bool(relations & reverse and not relations & required)}
    outcomes={}
    for path in sorted(ROOT.glob('reader_scores_*.json')):
        scores={p:s for q,p,s in json.loads(path.read_text())['scores'] if q=='U03'}
        raw=sorted(reader['oracle'],key=lambda p:(-scores[p],p))
        gate=sorted(reader['oracle'],key=lambda p:(scope[p]['demoted'],-scores[p],p))
        outcomes[path.stem.removeprefix('reader_scores_')]={
            'raw_top':raw[0],'after_gate':gate[0],'sufficient':gate[0] in reader['oracle_sufficient']}
    result={'post_hoc':True,'case':'U03','pool':'oracle','candidate_flags':scope,'outcomes':outcomes}
    (ROOT/'airport_direction_demo.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)

if __name__=='__main__':main()
