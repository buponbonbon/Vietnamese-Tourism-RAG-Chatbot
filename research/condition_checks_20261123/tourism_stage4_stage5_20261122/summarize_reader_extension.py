"""Verify the reader extension and recompute top-one selection from archived logits."""
from collections import Counter
from hashlib import file_digest
from pathlib import Path
import json
import math
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'tourism_stage4_20261121'))
from corpus_stage4 import BM25, normalize


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()


def load(name):
    return json.loads((ROOT/name).read_text())


def main():
    for line in (ROOT/'stage4_extension_freeze.sha256').read_text().splitlines():
        h,f=line.split()
        assert h==sha(ROOT/f),f
    qfile=ROOT/'stage4_extension_worksheet.json'
    queries=load(qfile.name)['questions']
    corpus=load('stage4_extension_corpus.json')
    ratings=load('stage4_extension_ratings.json')
    by_pid={d['id']:d for d in corpus}
    older={}
    for folder in ('vietnam_travel_natural_20261121','tourism_vietnamese_operators_20261121','tourism_stage4_20261121'):
        data=json.loads((ROOT.parent/folder/'benchmark.json').read_text())['questions']
        older[folder]=[q for q in data if not q['id'].startswith('P')]
    prior={normalize(r['query']) for data in older.values() for r in data}
    assert all(normalize(q['query']) not in prior for q in queries)
    assert len(prior)==27
    bm25=BM25(corpus)
    no_titles=BM25([{**d,'text':d['body']} for d in corpus])
    results={'BM25':{}}
    lower_body=[]
    for q in queries:
        restored=[p for p,_ in bm25.scores(q['query'])[:5]]
        assert restored==q['native_ids']
        keep={p for p,g in zip(restored,ratings[q['id']],strict=True) if g==2}
        assert keep==set(q['native_sufficient'])
        results['BM25'][q['id']]={'id':restored[0],'sufficient':restored[0] in keep}
        body_ids=[p for p,_ in no_titles.scores(q['query'])[:5]]
        lower_body.append({'qid':q['id'],'top5':body_ids,'known_anchor_in_top5':any(p in body_ids for p in q['proven_sufficient'])})
    pairs={(q['id'],p) for q in queries for p in q['candidate_ids']}
    for file in sorted(ROOT.glob('stage4_extension_scores_*.json')):
        condition=file.stem.removeprefix('stage4_extension_scores_')
        evidence=load(file.name)
        assert evidence['benchmark_sha256']==sha(qfile)
        model_scores={(q,p):s for q,p,s in evidence['scores'] if math.isfinite(s)}
        assert set(model_scores)==pairs and len(evidence['scores'])==len(pairs)
        results[condition]={}
        for q in queries:
            rank=sorted(q['native_ids'],key=lambda p:(-model_scores[q['id'],p],p))
            tied=model_scores[q['id'],rank[0]]==model_scores[q['id'],rank[1]]
            results[condition][q['id']]={'id':rank[0],'sufficient':rank[0] in set(q['native_sufficient']) and not tied,
                                       'strict_tie':tied}
    assert len(queries)==7 and sum(bool(q['native_sufficient']) for q in queries)==7
    assert not any(q['oracle_ids']!=q['native_ids'] for q in queries)
    assert len(results)==8, sorted(results)
    summary={k:sum(r['sufficient'] for r in v.values()) for k,v in results.items()}
    print(summary)
    print('body-only known anchor coverage',sum(r['known_anchor_in_top5'] for r in lower_body))
    output={
        'corpus':len(corpus),'queries':len(queries),'candidate_judgments':len(pairs),
        'n_available':sum(bool(q['native_sufficient']) for q in queries),
        'top1':summary,'outcomes':results,
        'body_only_sensitivity': lower_body,
        'prior_distinct_query_count':len(prior),'prior_all_Vietnamese':18,
        'distinct_reader_queries_combined':len(prior)+len(queries),
        'all_Vietnamese_combined':18+len(queries),
        'max_num_tokens':max(load(p.name)['max_tokens'] for p in ROOT.glob('stage4_extension_scores_*.json')),
        'new_sources':sorted({url for q in queries for url in (q['query_url'],*(by_pid[p]['url'] for p in q['proven_sufficient']))}),
    }
    (ROOT/'stage4_extension_results.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':
    main()
