"""Grade-2 evidence selection on reconstructed native/oracle pools. No tuning."""
from pathlib import Path
from itertools import combinations
import json,hashlib,csv,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def selected(rows,scores):return min(rows,key=lambda i:(-scores[i],i))

def self_check():
 assert selected([2,5,9],[0,0,1,0,0,1,0,0,0,0])==2
 s=[5,4,3,2,1];assert [float(selected(list(ids),s)==0) for ids in combinations(range(5),2) if 0 in ids]==[1,1,1,1]
 scores=[3,5,4,2,1];rates=[]
 for k in [2,3,4,5]:
  rates.append(np.mean([selected(sorted([0,*sub]),scores)==0 for sub in combinations([1,2,3,4],k-1)]))
 assert np.allclose(rates,[.5,1/6,0,0])
 return 'PASS'

def bootstrap(delta,source_groups,seed=20261113):
 delta=np.asarray(delta,dtype=float);names=sorted(set(source_groups));members=[[i for i,g in enumerate(source_groups) if g==name] for name in names]
 sizes=np.asarray(list(map(len,members)));sums=np.asarray([delta[ids].sum() for ids in members])
 rng=np.random.default_rng(seed);draw=rng.integers(0,len(names),size=(20000,len(names)))
 samples=sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)
 return {'mean_difference':float(delta.mean()),'source_group_95_percentile_interval':np.percentile(samples,[2.5,97.5]).tolist(),'queries':len(delta),'source_groups':len(names),'replicates':20000,'seed':seed}

def main():
 assert self_check()=='PASS'
 p=json.loads((ROOT/'protocol.json').read_text());qs=json.loads((ROOT/'cohort.json').read_text());pairs=json.loads((ROOT/'scoring_pairs.json').read_text())
 assert sha(ROOT/'cohort.json')==p['cohort_sha256'] and sha(ROOT/'scoring_pairs.json')==p['scoring_pairs_sha256']
 c=json.loads((ROOT/'inference_completion.json').read_text());assert c['status']=='COMPLETE' and len(c['models'])==13
 primary=[];per_model=[];per_query=[];data={};kdata={};tokenization=[]
 native_available=np.asarray([q['has_grade2_in_native_pool'] for q in qs]);missing=~native_available
 k_indices=[i for i,q in enumerate(qs) if sum(pairs[j]['grade']==2 for j in q['native_row_ids'])==1]
 assert len(qs)==140 and sum(native_available)==45 and sum(missing)==95 and len(k_indices)==41
 for spec,cm in zip(p['models'],c['models']):
  assert spec['name']==cm['model'];f=ROOT/'inference'/spec['name']/'scores.json';assert sha(f)==cm['sha256']
  raw=json.loads(f.read_text());assert raw['status']=='COMPLETE' and raw['identity']['protocol_sha256']==sha(ROOT/'protocol.json')
  assert raw['batch_consistency_max_abs_difference']<=1e-4
  scores=raw['scores'];assert len(scores)==795
  native=[];expanded=[];krates=[];tokenization.append({'model':spec['name'],**raw['identity']['tokenization']})
  for q in qs:
   a=selected(q['native_row_ids'],scores);b=selected(q['expanded_row_ids'],scores)
   x=int(pairs[a]['grade']==2);y=int(pairs[b]['grade']==2);native.append(x);expanded.append(y)
   if q['has_grade2_in_native_pool']:assert a==b
   per_query.append({'model':spec['name'],'query_id':q['benchmark_id'],'source_group_id':q['source_group_id'],'native_available':q['has_grade2_in_native_pool'],'native_selected_row':a,'native_selected_grade':pairs[a]['grade'],'native_hit':x,'expanded_selected_row':b,'expanded_selected_grade':pairs[b]['grade'],'expanded_hit':y})
  for qi in k_indices:
   q=qs[qi];positive=[i for i in q['native_row_ids'] if pairs[i]['grade']==2][0];neg=[i for i in q['native_row_ids'] if i!=positive]
   rates=[]
   for k in [2,3,4,5]:
    outcomes=[int(selected(sorted([positive,*sub]),scores)==positive) for sub in combinations(neg,k-1)]
    rates.append(float(np.mean(outcomes)))
   krates.append(rates)
  native=np.asarray(native,dtype=float);expanded=np.asarray(expanded,dtype=float)
  data[spec['name']]={'native':native,'expanded':expanded};kdata[spec['name']]=np.asarray(krates)
  per_model.append({'model':spec['name'],'condition':spec['condition'],'seed':spec['seed'],'native_correct':int(native.sum()),'native_hit':float(native.mean()),'native_conditional_selection_given_available':float(native[native_available].mean()),'oracle95_correct':int(expanded[missing].sum()),'oracle95_hit':float(expanded[missing].mean()),'expanded140_correct':int(expanded.sum()),'expanded140_hit':float(expanded.mean()),'k_selection_rates':dict(zip(['2','3','4','5'],np.mean(krates,axis=0).tolist()))})
 conditions=[];arrays={};karrays={}
 for condition in ['BGE','M0','M1','Q','R']:
  names=[s['name'] for s in p['models'] if s['condition']==condition]
  native=np.asarray([data[n]['native'] for n in names]);expanded=np.asarray([data[n]['expanded'] for n in names]);k=np.asarray([kdata[n] for n in names]);arrays[condition]={'native':native,'expanded':expanded};karrays[condition]=k
  fields={'native_hit':native.mean(axis=1),'selection_given_available':native[:,native_available].mean(axis=1),'oracle95_hit':expanded[:,missing].mean(axis=1),'expanded140_hit':expanded.mean(axis=1)}
  conditions.append({'condition':condition,'seeds':len(names),'mean':{key:float(v.mean()) for key,v in fields.items()},'population_sd':{key:float(v.std(ddof=0)) for key,v in fields.items()},'K_mean':dict(zip(['2','3','4','5'],k.mean(axis=(0,1)).tolist())),'K_population_sd_of_seed_query_means':dict(zip(['2','3','4','5'],k.mean(axis=1).std(axis=0,ddof=0).tolist()))})
 def paired(a,b,field,mask=None):
  diff=(arrays[a][field]-arrays[b][field]).mean(axis=0)
  if mask is None:mask=np.ones(len(qs),dtype=bool)
  return bootstrap(diff[mask],[q['source_group_id'] for q,ok in zip(qs,mask) if ok])
 contrasts={'primary_M1_minus_M0_native':paired('M1','M0','native'),'diagnostic_M1_minus_M0_oracle95':paired('M1','M0','expanded',missing),'diagnostic_M1_minus_M0_expanded140':paired('M1','M0','expanded'),'diagnostic_Q_minus_R_native':paired('Q','R','native'),'diagnostic_Q_minus_R_oracle95':paired('Q','R','expanded',missing)}
 did=((arrays['M1']['expanded']-arrays['M1']['native'])-(arrays['M0']['expanded']-arrays['M0']['native'])).mean(axis=0)
 contrasts['diagnostic_M1_minus_M0_oracle_gain_difference_full140']=bootstrap(did,[q['source_group_id'] for q in qs])
 kdiff=(karrays['M1']-karrays['M0']).mean(axis=0)[:,0]
 contrasts['diagnostic_K2_M1_minus_M0']=bootstrap(kdiff,[qs[i]['source_group_id'] for i in k_indices])
 result={'status':'COMPLETE','protocol_sha256':sha(ROOT/'protocol.json'),'cohort_sha256':sha(ROOT/'cohort.json'),'queries':140,'native_candidate_rows':700,'unique_scored_pairs':795,'source_groups':81,'natural_available':45,'natural_unavailable':95,'native_availability_ceiling':45/140,'oracle_cohort_queries':95,'oracle_source_groups':57,'K_queries':41,'K_source_groups':29,'conditions':conditions,'per_model':per_model,'paired_contrasts':contrasts,'tokenization':tokenization,'scope':'Reconstructed development cohort from original exact-hash reviewer files and frozen BM25 results; fresh full inference, not recovery of missing old logits. Q/R and oracle analyses are diagnostics; all intervals are conditional on existing fixed seeds.'}
 (ROOT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 (ROOT/'per_query_outcomes.json').write_text(json.dumps(per_query,ensure_ascii=False,indent=2)+'\n')
 for fname,rows in [('per_model_summary.csv',[{k:v for k,v in r.items() if k!='k_selection_rates'} for r in per_model]),('per_query_outcomes.csv',per_query)]:
  with (ROOT/fname).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({k:result[k] for k in ['queries','natural_available','natural_unavailable','conditions','paired_contrasts']},indent=2))

if __name__=='__main__':
 if '--self-check-only' in sys.argv:print('METRIC_SELF_CHECK',self_check())
 else:main()
