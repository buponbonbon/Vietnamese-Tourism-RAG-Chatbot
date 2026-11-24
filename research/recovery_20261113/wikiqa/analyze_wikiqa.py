"""Prespecified MRR contrast; MAP and Hit@1 remain secondary."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,csv,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
METRICS=('MRR','MAP','Hit@1')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def metrics(labels,scores,row_order):
 order=sorted(range(len(labels)),key=lambda i:(-scores[i],row_order[i]))
 y=np.asarray([labels[i] for i in order],dtype=int)
 relevant=np.flatnonzero(y)+1
 assert len(relevant)>0
 return [float(1/relevant[0]),float(np.mean(np.arange(1,len(relevant)+1)/relevant)),float(y[0])]

def self_check():
 assert np.allclose(metrics([0,1,0,1],[3,2,1,0],[0,1,2,3]),[.5,.5,0])
 assert np.allclose(metrics([0,1,1],[0,0,0],[0,1,2]),[.5,7/12,0])
 assert metrics([1,0],[1,0],[0,1])==[1,1,1]
 assert metrics([0,1],[0,0],[2,1])==[1,1,1]
 return 'PASS'

def write_csv(path,fields,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
 assert self_check()=='PASS'
 protocol=json.loads((ROOT/'evaluation_protocol.json').read_text())
 inputs=json.loads((ROOT/'evaluation_inputs.json').read_text())
 assert sha(ROOT/'evaluation_inputs.json')==protocol['input_sha256']
 completion=json.loads((ROOT/'inference_completion.json').read_text());assert completion['status']=='COMPLETE' and len(completion['models'])==7
 model_values={};per_model=[];query_rows=[]
 n=sum(len(q['candidates']) for q in inputs)
 for spec,cm in zip(protocol['models'],completion['models']):
  name=spec['name'];assert name==cm['model_name']
  p=ROOT/'inference'/name/'scores.json';assert sha(p)==cm['scores_sha256']
  data=json.loads(p.read_text());assert data['status']=='COMPLETE' and len(data['scores'])==n
  assert data['identity']['protocol_sha256']==sha(ROOT/'evaluation_protocol.json')
  assert data['batch_consistency_check']['status']=='PASS'
  values=[];offset=0
  for q in inputs:
   cs=q['candidates'];v=metrics([c['label'] for c in cs],data['scores'][offset:offset+len(cs)],[c['test_row_index'] for c in cs]);offset+=len(cs)
   values.append(v)
   query_rows.append({'model':name,'query_id':q['query_id'],'source_title':q['source_title'],**dict(zip(METRICS,v))})
  model_values[name]=np.asarray(values)
  per_model.append({'model':name,'condition':spec['condition'],'seed':spec['seed'],**dict(zip(METRICS,np.mean(values,axis=0).tolist()))})
 by_condition={};condition_rows=[]
 for condition in ('BGE','M0','M1'):
  names=[m['name'] for m in protocol['models'] if m['condition']==condition]
  arr=np.asarray([model_values[name] for name in names])
  by_condition[condition]=arr
  seed_means=arr.mean(axis=1)
  cond={'condition':condition,'seeds':len(names),'mean':dict(zip(METRICS,seed_means.mean(axis=0).tolist())),'population_sd_across_fixed_seeds':dict(zip(METRICS,seed_means.std(axis=0,ddof=0).tolist()))}
  condition_rows.append(cond)
 delta=(by_condition['M1']-by_condition['M0']).mean(axis=0)
 source_titles=sorted({q['source_title'] for q in inputs})
 members=[[i for i,q in enumerate(inputs) if q['source_title']==title] for title in source_titles]
 counts=np.asarray(list(map(len,members)),dtype=int)
 group_sums=np.asarray([delta[ids].sum(axis=0) for ids in members])
 rng=np.random.default_rng(protocol['primary']['bootstrap_seed'])
 draws=rng.integers(0,len(members),size=(protocol['primary']['bootstrap_replicates'],len(members)))
 boot=group_sums[draws].sum(axis=1)/counts[draws].sum(axis=1)[:,None]
 contrast={metric:{'mean_M1_minus_M0':float(delta[:,j].mean()),'source_bootstrap_95_percentile_interval':np.percentile(boot[:,j],[2.5,97.5]).tolist(),'role':'primary' if j==0 else 'secondary_unadjusted'} for j,metric in enumerate(METRICS)}
 result={'status':'COMPLETE','protocol_sha256':sha(ROOT/'evaluation_protocol.json'),'inputs_sha256':sha(ROOT/'evaluation_inputs.json'),'queries':len(inputs),'candidate_rows':n,'source_title_groups':len(members),'metric_self_checks':'PASS','conditions':condition_rows,'per_model':per_model,'paired_contrast':contrast,'primary_metric':'MRR','bootstrap_replicates':len(boot),'bootstrap_seed':protocol['primary']['bootstrap_seed'],'standard_deviation':'population across the fixed existing seeds (ddof=0); not future-seed uncertainty'}
 (ROOT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 write_csv(ROOT/'per_model_summary.csv',['model','condition','seed',*METRICS],per_model)
 write_csv(ROOT/'per_query_metrics.csv',['model','query_id','source_title',*METRICS],query_rows)
 flat_conditions=[{'condition':c['condition'],'seeds':c['seeds'],**{metric:c['mean'][metric] for metric in METRICS},**{metric+'_sd':c['population_sd_across_fixed_seeds'][metric] for metric in METRICS}} for c in condition_rows]
 write_csv(ROOT/'condition_summary.csv',['condition','seeds',*METRICS,*[m+'_sd' for m in METRICS]],flat_conditions)
 print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
 if '--self-check-only' in sys.argv:print('METRIC_SELF_CHECK',self_check())
 else:main()
