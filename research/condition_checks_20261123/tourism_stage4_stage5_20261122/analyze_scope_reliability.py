"""Post-hoc scope error audit and a reproducible metadata corruption sensitivity."""
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
import json

from analyze_stage5 import incompatible, outcome
from build_stage5 import key

ROOT = Path(__file__).resolve().parent
FRACTIONS = (0.01,0.05,0.10)
TRIALS = 30


def raw_failure(row, question, docs):
    if not row['available']:
        return 'unavailable'
    if row['strict_tie']:
        return 'tie'
    doc=docs[row['top']]
    if doc['agency'] != question['agency'] or doc['route'] != question['route']:
        return 'wrong agency or route'
    first, last = question['from'],question['to']
    if not {first,last}.issubset(doc['stops']):
        return 'required stop missing'
    if doc['stops'].index(first)>doc['stops'].index(last):
        return 'wrong direction'
    return 'pickup or dropoff restricted'


def corrupt(docs, probability, trial):
    # Alter membership, not the reranker input text. Each modified document
    # loses one recorded physical stop and gains a wrong populated identifier.
    altered={}
    for pid,doc in docs.items():
        rng=Random(int(key('noisy-stop-fields',pid,trial)[:16],16))
        if rng.random() >= probability:
            altered[pid]=doc
            continue
        stops=list(doc['stops'])
        i=rng.randrange(len(stops))
        stops[i]='artificial-wrong-stop'
        altered[pid]={**doc,'stops':stops}
    return altered


def mean(x):
    return sum(x)/len(x)


def main():
    benchmark=json.loads((ROOT/'stage5_eval.json').read_text())
    questions,docs=benchmark['questions'],benchmark['candidates']
    failure_rows=json.loads((ROOT/'stage5_outcomes.json').read_text())
    question_by_id={q['id']:q for q in questions}
    failures=defaultdict(Counter)
    for row in failure_rows:
        if row['gate'] in ('raw','unordered_stops') and not row['success']:
            failures[row['pool'],row['method'],row['gate']][raw_failure(
                row,question_by_id[row['qid']],docs
            )]+=1
    for q in questions:
        gold=set(q['pools']['stress']['sufficient'])
        retained=[p for p in q['pools']['stress']['candidates']
                  if p not in gold and not incompatible(docs[p],q,'unordered_stops')]
        assert len(retained)==2
        assert all(docs[p]['stops'].index(q['from'])>docs[p]['stops'].index(q['to']) for p in retained)
    scenarios={}
    for name in ('BGE','M0_42','M1_42'):
        data=json.loads((ROOT/f'stage5_scores_{name}.json').read_text())
        scenarios[name]={(q,p):val for q,p,val in data['scores']}
    sensitivity=[]
    for probability in FRACTIONS:
        by_method=defaultdict(list)
        for trial in range(TRIALS):
            altered=corrupt(docs,probability,trial)
            for pool in ('natural','stress'):
                for name, scores in scenarios.items():
                    rows=[outcome(q,pool,altered,scores,name,'unordered_stops') for q in questions]
                    by_method[pool,name].append({
                        'successful':sum(r['success'] for r in rows),
                        'gold_passages_demoted':sum(r['gold_excluded'] for r in rows),
                        'queries_with_all_sufficient_demoted':sum(r['all_gold_excluded'] for r in rows),
                    })
        for (pool,name),reps in by_method.items():
            sensitivity.append({
                'planned_noise_probability':probability,'pool':pool,'model':name,'trials':len(reps),
                **{k:mean([v[k] for v in reps]) for k in reps[0]},
            })
    result={'failure_counts':[{ 'pool':k[0],'model':k[1],'gate':k[2], **v}
                              for k,v in sorted(failures.items())],
            'synthetic_stop_metadata_corruption':sensitivity}
    (ROOT/'stage5_reliability.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in result['failure_counts']:
        if row['model']=='M1_42':
            print(row)
    for row in sensitivity:
        if row['planned_noise_probability']==.05:
            print(row)


if __name__=='__main__':
    main()
