"""Recheck saved-score arithmetic without private reviewer files.

This adaptation uses the original independent outranker-count, analytic K and
bootstrap-frequency checks. It does NOT revalidate original human judgments.
The original full source verifier and its earlier report remain unchanged.
"""
from pathlib import Path
import json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parent
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def bootstrap(delta, groups):
    names = sorted(set(groups)); index = {g:i for i,g in enumerate(names)}
    totals = np.zeros(len(names)); sizes = np.zeros(len(names))
    for d,g in zip(delta,groups):
        totals[index[g]] += d; sizes[index[g]] += 1
    draws = np.random.default_rng(20261113).integers(len(names), size=(20000,len(names)))
    frequency = np.zeros((20000,len(names)), dtype=np.int16)
    np.add.at(frequency, (np.arange(20000)[:,None], draws), 1)
    samples = (frequency @ totals)/(frequency @ sizes)
    return float(np.mean(delta)), np.percentile(samples,[2.5,97.5])


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert digest(ROOT/'cohort.json')==protocol['cohort_sha256']
    assert digest(ROOT/'scoring_pairs.json')==protocol['scoring_pairs_sha256']
    for name,expected in protocol['script_hashes'].items():assert digest(ROOT/name)==expected
    cohort=json.loads((ROOT/'cohort.json').read_text())
    pairs=json.loads((ROOT/'scoring_pairs.json').read_text())
    result=json.loads((ROOT/'results.json').read_text())
    completion=json.loads((ROOT/'inference_completion.json').read_text())
    available=np.array([q['has_grade2_in_native_pool'] for q in cohort]); missing=~available
    ki=[j for j,q in enumerate(cohort) if sum(pairs[i]['grade']==2 for i in q['native_row_ids'])==1]
    assert (sum(available),sum(missing),len(ki))==(45,95,41)
    outcomes={(r['model'],r['query_id']):r for r in json.loads((ROOT/'per_query_outcomes.json').read_text())}
    computed={}; all_token_hashes=set(); max_error=0.
    def close(a,b):
        nonlocal max_error
        error=float(np.max(np.abs(np.array(a)-np.array(b))))
        max_error=max(max_error,error); assert error<1e-12,(a,b)
    def rank_winner(ids,scores):
        # Rank from counts, without sorting or argmin/argmax.
        candidates=[i for i in ids if sum(scores[j]>scores[i] or (scores[j]==scores[i] and j<i) for j in ids)==0]
        assert len(candidates)==1; return candidates[0]
    for spec,cm,summary in zip(protocol['models'],completion['models'],result['per_model']):
        name=spec['name']; path=ROOT/'inference'/name/'scores.json'
        assert cm['model']==summary['model']==name and digest(path)==cm['sha256']
        raw=json.loads(path.read_text()); identity=raw['identity']; scores=raw['scores']
        assert len(scores)==795 and all(math.isfinite(s) for s in scores)
        assert identity['protocol_sha256']==digest(ROOT/'protocol.json')
        assert identity['weight_sha256']==spec['file_sha256']['model.safetensors']
        assert raw['batch_consistency_max_abs_difference']<=1e-4
        for key,value in protocol['runtime'].items(): assert identity['runtime'][key]==value
        all_token_hashes.add(identity['tokenization']['sha256'])
        n=[];e=[];k=[]
        for q in cohort:
            a=rank_winner(q['native_row_ids'],scores);b=rank_winner(q['expanded_row_ids'],scores)
            n.append(int(pairs[a]['grade']==2));e.append(int(pairs[b]['grade']==2))
            r=outcomes[name,q['benchmark_id']]
            assert (a,b,n[-1],e[-1])==(r['native_selected_row'],r['expanded_selected_row'],r['native_hit'],r['expanded_hit'])
        for j in ki:
            ids=cohort[j]['native_row_ids']; pos=next(i for i in ids if pairs[i]['grade']==2)
            wins=sum(scores[pos]>scores[i] or (scores[pos]==scores[i] and pos<i) for i in ids if i!=pos)
            k.append([math.comb(wins,z)/math.comb(4,z) if wins>=z else 0. for z in (1,2,3,4)])
        n=np.array(n);e=np.array(e);k=np.array(k)
        assert (int(n.sum()),int(e[missing].sum()),int(e.sum()))==(summary['native_correct'],summary['oracle95_correct'],summary['expanded140_correct'])
        close(k.mean(axis=0),[summary['k_selection_rates'][str(z)] for z in (2,3,4,5)])
        computed[name]=dict(native=n,expanded=e,K=k)
    assert len(all_token_hashes)==1 and len(computed)==13 and len(outcomes)==1820
    conditions={}
    for summary in result['conditions']:
        cond=summary['condition']; names=[s['name'] for s in protocol['models'] if s['condition']==cond]
        a={field:np.array([computed[n][field] for n in names]) for field in ('native','expanded','K')}; conditions[cond]=a
        seed_values=dict(native_hit=a['native'].mean(axis=1),selection_given_available=a['native'][:,available].mean(axis=1),oracle95_hit=a['expanded'][:,missing].mean(axis=1),expanded140_hit=a['expanded'].mean(axis=1))
        for field,values in seed_values.items():
            close(values.mean(),summary['mean'][field]);close(values.std(ddof=0),summary['population_sd'][field])
        close(a['K'].mean(axis=(0,1)),[summary['K_mean'][str(z)] for z in (2,3,4,5)])
    def diff(a,b,field,mask): return (conditions[a][field]-conditions[b][field]).mean(axis=0)[mask]
    full=np.ones(140,dtype=bool)
    values={
      'primary_M1_minus_M0_native':(diff('M1','M0','native',full),full),
      'diagnostic_M1_minus_M0_oracle95':(diff('M1','M0','expanded',missing),missing),
      'diagnostic_M1_minus_M0_expanded140':(diff('M1','M0','expanded',full),full),
      'diagnostic_Q_minus_R_native':(diff('Q','R','native',full),full),
      'diagnostic_Q_minus_R_oracle95':(diff('Q','R','expanded',missing),missing)}
    gain=((conditions['M1']['expanded']-conditions['M1']['native'])-(conditions['M0']['expanded']-conditions['M0']['native'])).mean(axis=0)
    values['diagnostic_M1_minus_M0_oracle_gain_difference_full140']=(gain,full)
    for key,(delta,mask) in values.items():
        point,ci=bootstrap(delta,[q['source_group_id'] for q,ok in zip(cohort,mask) if ok]);expected=result['paired_contrasts'][key]
        close(point,expected['mean_difference']);close(ci,expected['source_group_95_percentile_interval'])
    delta=(conditions['M1']['K']-conditions['M0']['K']).mean(axis=0)[:,0]
    point,ci=bootstrap(delta,[cohort[i]['source_group_id'] for i in ki]);expected=result['paired_contrasts']['diagnostic_K2_M1_minus_M0']
    close(point,expected['mean_difference']);close(ci,expected['source_group_95_percentile_interval'])
    report=dict(status='PASS',models=13,model_query_checks=1820,source_workbook_validation_performed=False,cohort_queries=140,oracle_queries=95,K_queries=41,bootstrap_contrasts=7,max_absolute_metric_or_interval_discrepancy=max_error,protocol_sha256=digest(ROOT/'protocol.json'),results_sha256=digest(ROOT/'results.json'),verifier_sha256=digest(Path(__file__)))
    (ROOT/'public_metric_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
