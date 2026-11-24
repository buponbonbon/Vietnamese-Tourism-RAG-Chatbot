"""Analyze frozen native and same-five substitution results, preserving unknown grades."""
from collections import Counter
import csv
import json
from corpus_stage4 import ROOT, BM25, load, save, digest

FORMS = ('BM25','BGE','M0_42','M0_123','M0_999','M1_42','M1_123','M1_999')


def judged_rank1(pool, grades, values, tie_break):
    best = max(values[p] for p in pool)
    top = [p for p in pool if values[p]==best]
    if len(top)>1 and not tie_break:
        return 'tie',None
    winner = min(top)
    v = grades[winner]
    return ('yes' if v==1 else ('no' if v==0 else 'unknown')),winner


def check_case(q, model, vals):
    native, native_label=q['native'],q['native_grades']
    active, active_label=q['active'],q['active_grades']
    raw=model=='BM25'
    native_status, native_winner=judged_rank1(native,native_label,vals,raw)
    active_status, active_winner=judged_rank1(active,active_label,vals,raw)
    compare, difference=None,None
    if q['nested']:
        pos, neg = q['nested']
        difference=vals[pos]-vals[neg]
        compare=difference>0
    return {'model':model,'case':q['id'],'cohort':q['cohort'],'group':q['service_group'],
            'availability':q['availability'],'active_condition':q['condition'],
            'native_hit1':native_status,'native_first':native_winner,
            'active_hit1':active_status,'active_first':active_winner,
            'k2_success':compare,'k2_margin':difference,
            'k2_displaced':None if compare is None else (
                'unknown' if compare and active_status=='unknown'
                else (compare and active_status in ('no','tie')))}


def interval(rows, field):
    n=Counter(r[field] for r in rows)
    return {'lower':n['yes'],'upper':n['yes']+n['unknown'],'n':len(rows),
            'unknown_at_top':n['unknown'],'ties':n['tie']}


def summary(rows):
    known=[r for r in rows if r['availability']=='present']
    oracle=[r for r in rows if r['active_condition']=='oracle']
    comparable=[r for r in rows if r['k2_success'] is not None]
    return {
        'native_hit1':interval(rows,'native_hit1'),
        'native_hit1_on_confirmed_present':interval(known,'native_hit1'),
        'oracle_hit1':interval(oracle,'active_hit1'),
        'nested_comparison':{
            'k2_success':sum(r['k2_success'] for r in comparable),
            'k2_ties':sum(r['k2_margin']==0 for r in comparable),
            'k5':interval(comparable,'active_hit1'),
            'displaced_lower':sum(r['k2_displaced'] is True for r in comparable),
            'displaced_upper':sum(r['k2_displaced'] in (True,'unknown') for r in comparable),
            'n':len(comparable),
            'native_n':sum(r['active_condition']=='native' for r in comparable),
            'oracle_n':sum(r['active_condition']=='oracle' for r in comparable),
        },
    }


def run():
    batch=load('benchmark.json')
    assert batch['corpus_sha256']==digest(ROOT/'corpus.json')
    qs=batch['questions']
    expected={(q['id'],pid) for q in qs for pid in q['candidate_ids']}
    bm25=BM25(load('corpus.json'))
    lexical={}
    for q in qs:
        ranks=bm25.scores(q['query'])
        assert [(p,s) for p,s in ranks[:5]]==[tuple(a) for a in q['retrieval']]
        lexical.update({(q['id'],p):s for p,s in ranks if p in q['candidate_ids']})
    by_model={'BM25':lexical}
    for model in FORMS[1:]:
        score=load(f'scores_{model}.json')
        assert score['benchmark_sha256']==digest(ROOT/'benchmark.json')
        values={(q,p):s for q,p,s in score['scores']}
        assert len(values)==len(score['scores'])
        by_model[model]=values

    rows=[]
    for model, scores in by_model.items():
        assert set(scores)==expected, model
        rows.extend(check_case(q,model,{p:scores[q['id'],p] for p in q['candidate_ids']})
                    for q in qs)

    output={}
    for cohort in ('new','bridge'):
        cohort_qs=[q for q in qs if q['cohort']==cohort]
        count=Counter(q['availability'] for q in cohort_qs)
        by_group={group:{m:summary([r for r in rows if r['cohort']==cohort and
                                    r['group']==group and r['model']==m]) for m in FORMS}
                  for group in sorted({q['service_group'] for q in cohort_qs})}
        output[cohort]={
            'n':len(cohort_qs),'groups':dict(Counter(q['service_group'] for q in cohort_qs)),
            'availability':dict(count),'available_lower':count['present'],
            'available_upper':count['present']+count['uncertain'],
            'models':{m:summary([r for r in rows if r['cohort']==cohort and r['model']==m])
                      for m in FORMS},'by_group':by_group
        }
    with (ROOT/'per_case.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    save('results.json',{'benchmark_sha256':digest(ROOT/'benchmark.json'),
                         'scored_pairs_per_model':len(expected),'outcomes':output})
    return output


if __name__=='__main__':
    result=run()
    for cohort, table in result.items():
        print(cohort, table['availability'])
        for m,row in table['models'].items():
            keys=('native_hit1','native_hit1_on_confirmed_present','oracle_hit1')
            line=[f'{row[k]["lower"]}–{row[k]["upper"]}/{row[k]["n"]}' for k in keys]
            nested=row['nested_comparison']
            print(m,' '.join(line),'K2',nested['k2_success'],'lost',nested['displaced_lower'],nested['displaced_upper'])
