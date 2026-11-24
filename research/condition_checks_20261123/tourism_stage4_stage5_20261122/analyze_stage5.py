"""Evaluate stage 5 priority using only frozen candidate scopes and model scores."""
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
import csv
import json
import math

from build_stage5 import digest, enough, key

ROOT = Path(__file__).resolve().parent
GATES = ('raw','route','unordered_stops','unordered_stops_mask25')


def incompatible(doc, question, gate):
    if gate == 'raw':
        return False
    if gate == 'unordered_stops_mask25' and int(key('mask25',question['id'],doc['id'])[:8],16)%4 == 0:
        return False

    for field in ('agency','route'):
        if doc.get(field) and question.get(field) and doc[field] != question[field]:
            return True
    if gate == 'route':
        return False
    return bool(doc.get('stops') and question.get('from') and question.get('to')
                and not {question['from'], question['to']}.issubset(doc['stops']))


def outcome(q, condition, docs, values, method, gate):
    pool = q['pools'][condition]
    ids = pool['candidates']
    gold = set(pool['sufficient'])
    bad = {p for p in ids if incompatible(docs[p],q,gate)}
    if method == 'direct_order_check':
        # Intentionally uses the scope and order fields from the label rule.
        scores = dict(zip(ids, pool['bm25'], strict=True))
        rank = sorted(ids, key=lambda p:(not enough(docs[p],(q['agency'],q['route']),q['from'],q['to']),
                                         -scores[p],p))
        replace = rank[0]
        return {'qid':q['id'],'agency':q['agency'],'route':q['route'],'family':q['family'],
                'available':bool(gold),'top':replace,'success':replace in gold,
                'strict_tie':False,'incompatible':0,'gold_excluded':0,
                'all_gold_excluded':False}
    if method == 'BM25':
        scores = dict(zip(ids,pool['bm25'],strict=True))
    else:
        scores = {p:values[q['id'],p] for p in ids}
    rank = sorted(ids,key=lambda p:(p in bad,-scores[p],p))
    first, second = rank[:2]
    strict_tie = method != 'BM25' and ((first in bad)==(second in bad)) and scores[first]==scores[second]
    return {'qid':q['id'],'agency':q['agency'],'route':q['route'],'family':q['family'],
            'available':bool(gold),'top':first,'success':first in gold and not strict_tie,
            'strict_tie':strict_tie,'incompatible':len(bad),'gold_excluded':len(gold&bad),
            'all_gold_excluded':bool(gold) and gold <= bad}


def summarize(rows, base):
    agg = {k:sum(r[k] for r in rows)
           for k in ('available','success','strict_tie','incompatible','gold_excluded','all_gold_excluded')}
    agg['N'] = len(rows)
    agg['repair'] = sum(r['success'] and not base[r['qid']]['success'] for r in rows)
    agg['regression'] = sum(not r['success'] and base[r['qid']]['success'] for r in rows)
    both = defaultdict(list)
    for r in rows:
        both[r['family']].append(r['success'])
    assert all(len(v) == 2 for v in both.values())
    agg['joint_directions'] = sum(all(x) for x in both.values())
    agg['families'] = len(both)
    return agg


def bootstrap(routes, draws=2000):
    rng = Random(20261122)
    groups = defaultdict(list)
    for agency, x in routes:
        groups[agency].append(x)
    samples = []
    for _ in range(draws):
        pairs = [rng.choice(v) for v in groups.values() for _ in v]
        num,den = map(sum,zip(*pairs))
        samples.append(num/den if den else 0)
    samples.sort()
    return [samples[int(draws*a)] for a in (.025,.975)]


def main():
    path = ROOT/'stage5_eval.json'
    data = json.loads(path.read_text())
    questions,docs = data['questions'],data['candidates']
    by_agency = sorted({q['agency'] for q in questions})
    all_pairs = {(q['id'],p) for q in questions for pool in q['pools'].values() for p in pool['candidates']}
    methods = {'BM25':{},'direct_order_check':{}}
    for p in sorted(ROOT.glob('stage5_scores_*.json')):
        j = json.loads(p.read_text())
        assert j['benchmark_sha256'] == digest(path),p.name
        methods[p.stem.removeprefix('stage5_scores_')] = {
            (q,d):s for q,d,s in j['scores'] if math.isfinite(s)
        }
        assert set(methods[p.stem.removeprefix('stage5_scores_')]) == all_pairs

    summary, wins = [], []
    for condition in ('natural','stress'):
        for model, scores in methods.items():
            raw = [outcome(q,condition,docs,scores,model,'raw') for q in questions]
            base = {r['qid']:r for r in raw}
            variants = GATES if model!='direct_order_check' else ('raw',)
            for gate in variants:
                rows = raw if gate=='raw' else [outcome(q,condition,docs,scores,model,gate) for q in questions]
                grouped = defaultdict(list)
                for r in rows:
                    grouped[r['agency'],r['route']].append(r)
                delta = [(agency, (sum(r['success']-base[r['qid']]['success'] for r in rr),
                                   sum(r['available'] for r in rr)))
                         for (agency,_),rr in grouped.items()]
                a = {'pool':condition,'method':model,'gate':gate,
                     'routes':len(grouped),**summarize(rows,base),
                     'paired_within_agency_route_bootstrap_conditional_95':bootstrap(delta)}
                a['agencies'] = {agency:summarize([r for r in rows if r['agency']==agency],base)
                                 for agency in by_agency}
                summary.append(a)
                wins.extend({'pool':condition,'method':model,'gate':gate,**row} for row in rows)
    assert all(r['gold_excluded']==0 for r in summary)
    assert all(sum(not incompatible(docs[p],q,'unordered_stops') for p in q['pools']['stress']['candidates'])==3
               for q in questions)
    result = {'benchmark_sha256':digest(path), 'pair_logits_per_model':len(all_pairs),
              'n_corpus':data['n_corpus'],'summary':summary}
    (ROOT/'stage5_results.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'stage5_outcomes.json').write_text(json.dumps(wins,indent=2)+'\n')
    with (ROOT/'stage5_results.csv').open('w',encoding='utf-8',newline='') as out:
        fields = ['pool','method','gate','routes','N','available','success','repair','regression',
                  'joint_directions','families','gold_excluded','strict_tie']
        writer = csv.DictWriter(out,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(summary)
    for a in summary:
        print(a['pool'],a['method'],a['gate'],f"{a['success']}/{a['available']}",
              f"{a['repair']}-{a['regression']}", a['paired_within_agency_route_bootstrap_conditional_95'])


if __name__ == '__main__':
    main()
