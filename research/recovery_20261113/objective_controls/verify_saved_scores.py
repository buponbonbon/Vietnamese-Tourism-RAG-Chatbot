"""Recheck nine saved objective score/log records without private TEST text.

The original full-source audit is retained separately. This check validates
stored row identities, logits, paired arithmetic and update counts. Mechanical
defect indices are taken from the earlier source audit, not rediscovered here.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def bootstrap(delta,groups):
    names=sorted(set(groups));index={g:i for i,g in enumerate(names)}
    totals=np.zeros(len(names));sizes=np.zeros(len(names))
    for value,group in zip(delta,groups):
        totals[index[group]]+=value;sizes[index[group]]+=1
    draws=np.random.default_rng(20261112).integers(len(names),size=(20000,len(names)))
    frequency=np.zeros((20000,len(names)),dtype=np.int16)
    np.add.at(frequency,(np.arange(20000)[:,None],draws),1)
    samples=(frequency@totals)/(frequency@sizes)
    return np.percentile(samples,[2.5,97.5])


def main():
    reference=json.loads((ROOT/'objective_score_audit.json').read_text())
    arrays={};ties={};groups=None;maximum=0.
    def close(a,b):
        nonlocal maximum
        error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        maximum=max(maximum,error);assert error<1e-12,(a,b)
    for path in sorted((ROOT/'remote_rerun_metadata').glob('*/*/test_scores.json')):
        raw=json.loads(path.read_text());manifest=json.loads((path.parent/'manifest.json').read_text())
        key=(manifest['run']['case'],manifest['run']['seed']);assert key not in arrays
        assert raw['test_sha256']==reference['test_sha256']
        assert raw['weight_sha256']==manifest['weight_sha256']
        records=raw['scores'];assert len(records)==1136
        assert [r['index'] for r in records]==list(range(1136))
        ids=[str(r['doc_id']) for r in records]
        if groups is None:groups=ids
        assert groups==ids
        values=[];equal=[]
        for r in records:
            a,b=r['positive_logit'],r['negative_logit']
            assert math.isfinite(a) and math.isfinite(b)
            assert bool(r['correct'])==(a>b) and bool(r['tie'])==(a==b)
            values.append(a>b);equal.append(a==b)
        arrays[key]=np.array(values);ties[key]=np.array(equal)
        rows=[json.loads(line) for line in (path.parent/'progress.jsonl').read_text().splitlines() if line]
        assert [r['update'] for r in rows]==list(range(1,409))
        assert sum(r.get('pairs',r.get('samples',0)) for r in rows)==26004
        assert manifest['optimizer_updates']==408 and manifest['training_pairs_seen']==26004
        source=next(r for r in reference['runs'] if (r['condition'],r['seed'])==key)
        assert sha(path)==source['test_scores_sha256']
        assert int(arrays[key].sum())==source['correct'] and int(ties[key].sum())==source['ties']
    assert len(arrays)==9 and len(set(groups))==247
    keep=np.ones(1136,dtype=bool);keep[reference['invalid_test_rows_zero_based']]=False
    for row in reference['conditions']:
        condition=row['condition'];a=np.array([arrays[condition,s] for s in [42,123,999]])
        t=np.array([ties[condition,s] for s in [42,123,999]])
        close(a.mean(axis=1),np.array(row['correct_by_seed'])/1136)
        close(a.mean(),row['mean_accuracy']);close(a.mean(axis=1).std(ddof=0),row['population_sd'])
        close(a[:,keep].mean(),row['filtered_mean_accuracy']);close((a+.5*t).mean(),row['half_tie_mean_accuracy'])
    tests=[]
    for other in ['R05','R08']:
        for seed in [42,123,999]:
            a,b=arrays['R03',seed],arrays[other,seed]
            x=int(np.sum(a&~b));y=int(np.sum(~a&b));n=x+y
            p=min(1.,2*sum(math.comb(n,k) for k in range(min(x,y)+1))/2**n) if n else 1.
            tests.append(dict(reference='R03',other=other,seed=seed,reference_only=x,other_only=y,p_exact_two_sided=p))
    last=0.
    for j,i in enumerate(sorted(range(6),key=lambda i:tests[i]['p_exact_two_sided'])):
        last=max(last,min(1.,(6-j)*tests[i]['p_exact_two_sided']));tests[i]['p_holm']=last
    for a,b in zip(tests,reference['paired_tests']):
        assert a.keys()==b.keys()
        for key in a:
            if isinstance(a[key],(float,int)):close(a[key],b[key])
            else:assert a[key]==b[key]
    for row in reference['exploratory_source_group_bootstrap']:
        other=row['other_minus_R03']
        delta=np.mean([arrays[other,s].astype(float)-arrays['R03',s] for s in [42,123,999]],axis=0)
        close(delta.mean(),row['mean_difference'])
        close(bootstrap(delta,groups),row['exploratory_group_bootstrap_95_percentile_interval'])
    report=dict(status='PASS',saved_runs=9,rows_per_run=1136,paired_tests=6,source_groups=247,max_arithmetic_discrepancy=maximum,original_TEST_text_revalidated=False,scope='Saved-score/update-log arithmetic only. Defect indices and source TEST hash come from the preserved earlier full-source audit.')
    (ROOT/'public_score_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
