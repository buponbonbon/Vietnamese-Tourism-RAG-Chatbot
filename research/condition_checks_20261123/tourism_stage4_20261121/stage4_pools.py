"""Resolve author wording, independent evidence, and all native top-five pools."""
from collections import Counter
import csv
from pathlib import Path
import json

from corpus_stage4 import ROOT, BM25, load, save, normalize, source_index, read_html, span, digest


def records(name, key='id'):
    return {x[key]: x for x in load(name)}


def body_text(source):
    doc = read_html(source)
    paragraphs = [normalize(' '.join(e.itertext())) for e in
                  doc.xpath('//article[contains(@class,"fck_detail")]//p')]
    dates = [normalize(x) for x in doc.xpath('//*[contains(@class,"date")]/text()')
             if 'GMT' in x]
    assert len(paragraphs) >= 2 and len(dates) == 1
    return normalize(' '.join(paragraphs[1:])), dates[0]


def collect_queries():
    manifests = {'reddit':records('reddit_screen.json'),'comment':records('comment_screen.json'),
                 'forum':records('forum_screen.json')}
    sources = source_index()
    existing = []
    for older in ('vietnam_travel_natural_20261121','tourism_vietnamese_operators_20261121'):
        existing.extend(x['query'] for x in
                        json.loads((ROOT.parent/older/'benchmark.json').read_text())['questions'])
    existing = set(map(normalize,existing))
    prior = json.loads((ROOT.parent/'tourism_vietnamese_operators_20261121'/'benchmark.json').read_text())
    pilot = {x['id']:x for x in prior['questions']}
    for q in load('case_plan.json'):
        kind, key = q['kind'], q['key']
        if kind == 'pilot':
            row = pilot[key]
            yield {**q, 'query':row['query'], 'query_url':row['article_url'],
                   'query_timestamp':row['publication'], 'query_proof':row['reader_source']}
            continue
        if kind == 'article':
            meta, file = sources[key]
            assert digest(file) == meta['sha256']
            original, date = body_text(file)
            url, body = meta['final_url'], original
            anchor = {'source_sha256': meta['sha256']}
        else:
            row = manifests[kind][key]
            date = row.get('date_epoch') or row.get('timestamp')
            url = row.get('source_url') or row.get('source')
            body = normalize(row.get('query') or row.get('body'))
            anchor = {'locator':key}
        if q.get('spans'):
            excerpt = '\n'.join(span(body, a, b) for a,b in q['spans'])
        else:
            excerpt = span(body, q.get('start'), q.get('end'))
        assert 16 <= len(excerpt) <= 890 and normalize(excerpt) not in existing, q['id']
        if q.get('title_context'):
            assert kind == 'comment'
            excerpt = row['title'] + '\n' + excerpt
        yield {**q, 'query':excerpt,'query_url':url,'query_timestamp':date,'query_proof':anchor}


def ids_for_rules(documents, rules):
    matches = set()
    for rule in rules:
        scope, markers = rule
        good = [p['id'] for p in documents if p['id'].rsplit('.',1)[0]==scope
                and all(normalize(m).casefold() in normalize(p['text']).casefold() for m in markers)]
        matches.update(good)
    return matches


def build():
    corpus = load('corpus.json')
    docs = {p['id']:p for p in corpus}
    bm25 = BM25(corpus)
    queries = list(collect_queries())
    used = set()
    pools = []
    for q in queries:
        assert q['query'] not in used, q['id']
        used.add(q['query'])
        gold = ids_for_rules(corpus,q['gold_rules'])
        assert gold, (q['id'],q['gold_rules'])
        ranked = bm25.scores(q['query'])
        native = [p[0] for p in ranked[:5]]
        anchor = next(x[0] for x in ranked if x[0] in gold)
        hits = [i for i,(p,_) in enumerate(ranked,1) if p in gold]
        assert all(q['query_url'].split('#')[0] != docs[p]['url'] for p in gold)
        pools.append({
            'id':q['id'],'query':q['query'],'service_group':q['group'],
            'cohort':'bridge' if q['kind']=='pilot' else 'new','reader_kind':q['kind'],
            'decision':q['decision'],'scope':q['scope'],'query_url':q['query_url'],
            'query_timestamp':q['query_timestamp'],'query_proof':q['query_proof'],
            'retrieval':[[pid,score] for pid,score in ranked[:5]],'native':native,
            'gold':sorted(gold),'gold_proof':q['gold_rules'],'anchor':anchor,
            'first_known_sufficient_rank':hits[0],
            'eligible_domains':q.get('domains',[]),
        })
    return pools


def review_sheet(pools, docs):
    sheet=[]
    for row in pools:
        for rank, pid in enumerate(row['native'],1):
            cand = docs[pid]
            if pid in row['gold']:
                grade, reason = '1','Exact source clause meets all declared conditions.'
            elif cand['service'] not in row['eligible_domains']:
                grade, reason = '0','Different service or jurisdiction for this decision.'
            else:
                grade, reason = 'unknown','Requires review against scoped question.'
            sheet.append({'id':row['id'],'rank':rank,'pid':pid,'grade':grade,'reason':reason,
                          'query':row['query'],'decision':row['decision'],'domain':cand['service'],
                          'passage':cand['text'],'source':cand['url']})
    with (ROOT/'review_native.tsv').open('w',newline='') as out:
        w=csv.DictWriter(out, delimiter='\t',fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    print('Native candidate rows requiring explicit human-readable review',sum(x['grade']=='unknown' for x in sheet))


if __name__ == '__main__':
    pools = build()
    save('native_pools.json',pools)
    review_sheet(pools,{p['id']:p for p in load('corpus.json')})
    print('Query groups',Counter(p['service_group'] for p in pools))
