"""Reproduce candidate and checkpoint counts and audit cumulative Vietnamese coverage."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import math
import re
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'tourism_stage4_20261121'))
from corpus_stage4 import BM25, normalize


def load(file):
    return json.loads(file.read_text())


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def main():
    frozen=ROOT/'reader_final.json'
    qs=load(frozen)
    docs=load(ROOT/'reader_corpus.json')
    for line in (ROOT/'reader_freeze.sha256').read_text().splitlines():
        h,n=line.split()
        assert h==digest(ROOT/n)
    bm25=BM25(docs)
    for q in qs:
        assert q['native']==[p for p,s in bm25.scores(q['query'])[:5]]
        assert all(q['rating'][p]==2 for p in q['native_sufficient'])
        assert len(q['native'])==len(set(q['oracle']))==5
        if not q['native_sufficient']:
            assert q['native'][:4]==q['oracle'][:4]
    prior={
        'reader_pilot':ROOT.parent/'vietnam_travel_natural_20261121'/'benchmark.json',
        'operators':ROOT.parent/'tourism_vietnamese_operators_20261121'/'benchmark.json',
        'stage4':ROOT.parent/'tourism_stage4_20261121'/'benchmark.json',
        'stage4_extension':ROOT.parent/'tourism_stage4_stage5_20261122'/'stage4_extension_worksheet.json',
    }
    queries=set()
    vi=[]
    early_vi_ids=[]
    for cohort,p in prior.items():
        for q in load(p)['questions']:
            if cohort=='stage4' and q['id'].startswith('P'):
                continue
            t=normalize(q['query'])
            assert t not in queries,t
            queries.add(t)
            if cohort=='reader_pilot':
                if q['evidence_language']!='vi':
                    continue
                early_vi_ids.append(q['id'])
            vi.append({'id':q['id'],'cohort':cohort})
    assert early_vi_ids==['p2_05','p3_11','p4_09']
    assert len(queries)==34 and len(vi)==28
    for q in qs:
        assert normalize(q['query']) not in queries
        queries.add(normalize(q['query']))
    pairs={(q['id'],pid) for q in qs for pid in q['candidate_ids']}
    details={}
    def check(method,logits=None):
        outcomes=[]
        for q in qs:
            for pool in ('native','oracle'):
                if pool=='oracle' and q['native_sufficient']:
                    continue
                ids=q[pool]
                if method=='BM25':
                    ranked=ids
                    tied=False
                else:
                    val={p:logits[q['id'],p] for p in ids}
                    ranked=sorted(ids,key=lambda p:(-val[p],p))
                    tied=val[ranked[0]]==val[ranked[1]]
                outcomes.append({'case':q['id'],'pool':pool,'top':ranked[0],
                                 'ranked':ranked,'sufficient':ranked[0] in q[f'{pool}_sufficient'] and not tied})
        details[method]=outcomes
    check('BM25')
    score_files=sorted(ROOT.glob('reader_scores_*.json'))
    assert len(score_files)==7
    for p in score_files:
        s=load(p)
        assert s['benchmark_sha256']==digest(frozen)
        values={(q,p):score for q,p,score in s['scores']}
        assert set(values)==pairs and len(values)==len(s['scores']) and all(math.isfinite(x) for x in values.values())
        check(p.stem.removeprefix('reader_scores_'),values)
    D={d['id']:d for d in docs}
    src=[{'case':q['id'],'decision':q['scope'],'question_url':q['question_url'],
          'evidence_urls':list(dict.fromkeys(D[i]['url'] for i in (*q['anchors'],*q['native_sufficient'])))}
         for q in qs]
    details_totals={method:{k:sum(row['sufficient'] for row in rows if row['pool']==k)
                            for k in ('native','oracle')} for method,rows in details.items()}
    result={'resolved_new':len(qs),'count_prior_vi':len(vi),'count_supported_vi':len(vi)+len(qs),
            'corrected_prior_pilot_ids':early_vi_ids,
            'new_native_available':sum(bool(q['native_sufficient']) for q in qs),
            'new_native_top1':{m:x['native'] for m,x in details_totals.items()},
            'new_oracle_top1_on_missing':{m:x['oracle'] for m,x in details_totals.items()},
            'oracle_question_count':sum(not q['native_sufficient'] for q in qs),
            'n_library':len(docs),'n_pair_logits':len(pairs)*len(score_files),
            'candidate_screen':dict(Counter(q['kind'] for q in load(ROOT/'candidate_questions.json'))),
            'selection_trace':details,'source_cases':src,'prior_cohort_count':dict(Counter(x['cohort'] for x in vi))}
    (ROOT/'reader_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ('selection_trace','source_cases')})
    for method,winners in details.items():
        print(method,[(r['case'],r['pool'],r['top'],r['sufficient']) for r in winners if r['case']=='U03'])


if __name__=='__main__':main()
