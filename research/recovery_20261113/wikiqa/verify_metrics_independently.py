"""Independent rank-count arithmetic, native-data integrity, and bootstrap check."""
from pathlib import Path
from collections import defaultdict
import json,csv,hashlib,math
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
METRICS=('MRR','MAP','Hit@1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((ROOT/'evaluation_protocol.json').read_text())
qs=json.loads((ROOT/'evaluation_inputs.json').read_text())
result=json.loads((ROOT/'results.json').read_text())
assert sha(ROOT/'evaluation_inputs.json')==protocol['input_sha256']==result['inputs_sha256']
assert sha(ROOT/'evaluation_protocol.json')==result['protocol_sha256']
raw=pq.read_table(ROOT/'wikiqa_test.parquet').to_pylist()
assert sha(ROOT/'wikiqa_test.parquet')==protocol['dataset']['test_parquet_sha256']
native=defaultdict(list)
for i,r in enumerate(raw):native[r['question_id']].append((i,r))
expected=[qid for qid,rows in native.items() if {r['label'] for _,r in rows}=={0,1}]
assert expected==[q['query_id'] for q in qs]
for q in qs:
 rows=native[q['query_id']];assert len(rows)==len(q['candidates'])
 for order,((i,r),c) in enumerate(zip(rows,q['candidates'])):
  assert c=={'test_row_index':i,'candidate_order':order,'text':r['answer'],'label':r['label']}
  assert q['question']==r['question'] and q['source_title']==r['document_title']
with (ROOT/'per_query_metrics.csv').open() as f:exported={(r['model'],r['query_id']):[float(r[m]) for m in METRICS] for r in csv.DictReader(f)}
assert len(exported)==len(qs)*len(protocol['models'])
arrays={};maximum_metric_error=0.;tokens=set();runtime_versions=set();checks=[]
for spec in protocol['models']:
 data=json.loads((ROOT/'inference'/spec['name']/'scores.json').read_text())
 assert data['status']=='COMPLETE'
 assert data['identity']['weight_sha256']==spec['file_sha256']['model.safetensors']
 assert data['identity']['protocol_sha256']==sha(ROOT/'evaluation_protocol.json')
 assert data['identity']['input_sha256']==sha(ROOT/'evaluation_inputs.json')
 assert data['identity']['scorer_sha256']==protocol['pre_inference_script_sha256']['score_wikiqa_cpu.py']
 assert data['batch_consistency_check']['status']=='PASS' and data['batch_consistency_check']['maximum_absolute_difference']<=1e-4
 assert data['identity']['tokenization']['truncated_rows']==0
 tokens.add(data['identity']['tokenization']['sha256'])
 runtime_versions.add(json.dumps(data['identity']['runtime'],sort_keys=True))
 all_scores=data['scores'];assert len(all_scores)==2341 and all(math.isfinite(s) for s in all_scores)
 values=[];offset=0
 for q in qs:
  cs=q['candidates'];s=all_scores[offset:offset+len(cs)];offset+=len(cs)
  ranks=[]
  for i,c in enumerate(cs):
   ranks.append(1+sum(s[j]>s[i] or (s[j]==s[i] and cs[j]['test_row_index']<c['test_row_index']) for j in range(len(cs))))
  positive_ranks=[ranks[i] for i,c in enumerate(cs) if c['label']==1]
  mrr=1/min(positive_ranks)
  ap=sum(sum(other<=rank for other in positive_ranks)/rank for rank in positive_ranks)/len(positive_ranks)
  hit=float(1 in positive_ranks)
  v=[mrr,ap,hit];values.append(v)
  err=float(np.max(np.abs(np.asarray(v)-exported[(spec['name'],q['query_id'])])))
  maximum_metric_error=max(maximum_metric_error,err);assert err<=1e-12
 arrays[spec['name']]=np.asarray(values)
 checks.append({'model':spec['name'],'queries':len(values),'batch_consistency_check':data['batch_consistency_check']})
assert len(tokens)==len(runtime_versions)==1
for row in result['per_model']:
 for j,m in enumerate(METRICS):assert abs(arrays[row['model']][:,j].mean()-row[m])<=1e-12
for condition in result['conditions']:
 a=np.asarray([arrays[s['name']].mean(axis=0) for s in protocol['models'] if s['condition']==condition['condition']])
 for j,m in enumerate(METRICS):
  assert abs(a[:,j].mean()-condition['mean'][m])<=1e-12
  assert abs(a[:,j].std(ddof=0)-condition['population_sd_across_fixed_seeds'][m])<=1e-12
left=np.asarray([arrays[s['name']] for s in protocol['models'] if s['condition']=='M0'])
right=np.asarray([arrays[s['name']] for s in protocol['models'] if s['condition']=='M1'])
differences=np.mean(right-left,axis=0)
names=sorted({q['source_title'] for q in qs});gids=np.asarray([names.index(q['source_title']) for q in qs])
gsizes=np.bincount(gids,minlength=len(names))
gsums=np.asarray([differences[gids==g].sum(axis=0) for g in range(len(names))])
rng=np.random.default_rng(protocol['primary']['bootstrap_seed'])
draws=rng.integers(len(names),size=(protocol['primary']['bootstrap_replicates'],len(names)))
weights=np.zeros((len(draws),len(names)),dtype=np.int16)
np.add.at(weights,(np.arange(len(draws))[:,None],draws),1)
boot=(weights@gsums)/(weights@gsizes)[:,None]
ci=np.quantile(boot,[.025,.975],axis=0)
ci_error=0.
for j,m in enumerate(METRICS):
 target=result['paired_contrast'][m]
 assert abs(float(differences[:,j].mean())-target['mean_M1_minus_M0'])<=1e-12
 err=float(np.max(np.abs(ci[:,j]-target['source_bootstrap_95_percentile_interval'])))
 ci_error=max(ci_error,err);assert err<=1e-12
report={'status':'PASS','verification_method':'Independent candidate outranker counts and positive-rank counts, then source-resampling frequency matrix; no imports from scoring/analysis scripts','native_cohort_order_text_label_integrity':'PASS','query_model_combinations_checked':len(exported),'maximum_exported_metric_absolute_error':maximum_metric_error,'maximum_bootstrap_interval_absolute_error':ci_error,'tokenization_common_across_models':True,'runtime_common_across_models':True,'no_truncation':True,'models':checks,'scope':'Recomputes results from the exported raw score files and native Parquet; does not regenerate logits without model weights.'}
(ROOT/'independent_metric_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='models'},ensure_ascii=False,indent=2))
