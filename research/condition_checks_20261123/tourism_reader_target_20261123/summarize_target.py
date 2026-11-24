"""Independently reconstruct the cumulative ledger and score frozen outcomes."""
from collections import Counter
from csv import DictWriter
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
CONDITIONS = ['BM25','BGE',*(f'{family}_{seed}' for family in ('M0','M1') for seed in (42,123,999))]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def canonical(url):
    parsed = urlsplit(url)
    domain = parsed.hostname.removeprefix('www.')
    return f'{domain}{parsed.path.rstrip("/")}'


def ledger(new, corpus):
    prior = ROOT.parent
    sources = [
        ('vietnam_travel_natural_20261121','benchmark.json','older pilot',lambda r:r['id'] in {'p2_05','p3_11','p4_09'}),
        ('tourism_vietnamese_operators_20261121','benchmark.json','older pilot',lambda r:True),
        ('tourism_stage4_20261121','benchmark.json','native top five',lambda r:r['cohort'] != 'previous' and r['id'].startswith('N')),
        ('tourism_stage4_stage5_20261122','stage4_extension_worksheet.json','native top five',lambda r:True),
        ('tourism_reader_scale_20261122','reader_final.json','native top five',lambda r:True),
    ]
    data = []
    for folder, filename, protocol, predicate in sources:
        rel = Path(folder) / filename
        path = prior / rel
        rows = read(path)
        if isinstance(rows, dict):
            rows = rows['questions']
        selected = [r for r in rows if predicate(r)]
        for row in selected:
            query_url = row.get('query_url',row.get('question_url',row.get('article_url')))
            qproof = row.get('query_proof',row.get('query_source_sha256',
                row.get('story_sha256',row.get('source_sha256'))))
            anchor = row.get('anchor',row.get('proven_sufficient',row.get('anchors',row.get('sufficient'))))
            src = row.get('query_locator',row.get('query_selector',row.get('source_id',row.get('query_proof'))))
            data.append(dict(case=row['id'],question=row['query'],question_url=query_url,query_proof=qproof,
                             source_locator=src,evidence=anchor,cohort=folder,mode=protocol,
                             evidence_language='vi',archive=str(rel),archive_sha256=digest(path)))
    assert [Counter(x['cohort'] for x in data)[s[0]] for s in sources] == [3,7,11,7,8]
    assert len(data) == 36
    frozen = ROOT/'reader_final.json'
    for row in new:
        parts = corpus[row['anchors'][0]]['provenance']
        data.append(dict(case=row['id'],question=row['quote'],question_url=row['question_url'],
                         query_proof=row['question_sha256'],source_locator=row['source_id'],
                         evidence=[{'id':row['anchors'][0],'url':p['url'],'locator':p['locator'],
                                    'snapshot_sha256':p['snapshot_sha256']} for p in parts],
                         cohort=ROOT.name,mode='native top five',evidence_language='vi',
                         archive='reader_final.json',archive_sha256=digest(frozen)))
    assert all(c['question_url'] and c['evidence'] for c in data)
    assert len({re.sub(r'\W+', ' ',r['question'].casefold()) for r in data}) == len(data) == 60
    assert len({r['case'] for r in data}) == len(data)
    assert not ({canonical(r['question_url']) for r in data[:36]} &
                {canonical(r['question_url']) for r in data[36:]})
    assert Counter(r['mode'] for r in data) == {'native top five':50,'older pilot':10}
    return data


def main():
    for line in (ROOT/'reader_freeze.sha256').read_text().splitlines():
        checksum, name = line.split()
        assert digest(ROOT/name) == checksum
    frozen = ROOT/'reader_final.json'
    questions = read(frozen)
    all_docs = read(ROOT/'reader_corpus.json')
    corpus = {p['id']:p for p in all_docs}
    cases = ledger(questions, corpus)
    assert len(all_docs)==len(corpus)==437
    results, pair_trace = {}, []
    availability = {q['id']: bool(q['native_sufficient']) for q in questions}
    expected_pairs = {(q['id'], p) for q in questions for p in q['candidate_ids']}
    for q in questions:
        assert len(set(q['native'])) == len(set(q['oracle'])) == 5
        assert q['native_sufficient'] == [p for p in q['native'] if q['rating'][p] == 2]
        assert set(q['candidate_ids']) == set(q['native'] + q['oracle'])
        if availability[q['id']]:
            assert q['native'] == q['oracle']
        else:
            assert q['native'][:4] == q['oracle'][:4] and q['native'][4] != q['oracle'][4]
            assert q['oracle'][4] == q['oracle_sufficient'][0] == q['anchors'][0]
    for key in CONDITIONS:
        condition = Counter()
        inference_metadata = {}
        if key != 'BM25':
            scores = read(ROOT/f'reader_scores_{key}.json')
            assert scores['benchmark_sha256'] == digest(frozen)
            logits = {(case, doc): value for case,doc,value in scores['scores']}
            assert len(logits) == len(scores['scores']) and set(logits) == expected_pairs
            assert all(math.isfinite(s) for s in logits.values())
            inference_metadata = {k:v for k,v in scores.items() if k!='scores'}
        for q in questions:
            for pool in ('native','oracle'):
                if pool=='oracle' and availability[q['id']]:
                    continue
                ids = q[pool]
                gold = q[f'{pool}_sufficient']
                order = ids if key=='BM25' else sorted(ids,key=lambda p:(-logits[q['id'],p],p))
                selected = order[0]
                hit = selected in gold
                case = q['id']
                if pool=='native':
                    condition['native_hit'] += hit
                    if availability[case]:
                        condition['available_native_fail'] += not hit
                else:
                    condition['replacement_hit'] += hit
                pair_trace.append(dict(case=case,model=key,pool=pool,selected=selected,hit=hit,
                                       available=bool(gold),rank=' '.join(order)))
        results[key] = dict(condition, metadata=inference_metadata)
    result = dict(native_available=sum(availability.values()), native_missing=len(questions)-sum(availability.values()),
                  n_new=len(questions),n_corpus=len(corpus),n_scored_pairs=len(expected_pairs),
                  n_checkpoint_logits=len(expected_pairs)*(len(CONDITIONS)-1),models=results,
                  n_cumulative=len(cases),n_cumulative_native=50,
                  benchmark_sha256=digest(frozen),source_question_archives=Counter(x['cohort'] for x in cases))
    (ROOT/'reader_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'reader_ledger_60.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'reader_selection_trace.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = DictWriter(f, fieldnames=list(pair_trace[0]))
        writer.writeheader()
        writer.writerows(pair_trace)
    print({k:v for k,v in result.items() if k!='models'})
    for key, r in results.items():
        print(key,*(r[n] for n in ('native_hit','available_native_fail','replacement_hit')),
              'max_tokens',r['metadata'].get('max_tokens'))


if __name__=='__main__':
    main()
