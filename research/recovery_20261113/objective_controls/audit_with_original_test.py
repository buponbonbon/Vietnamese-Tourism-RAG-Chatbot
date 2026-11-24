"""Re-audit nine returned objective runs from original JSON logits; no inference."""
from pathlib import Path
from collections import Counter
import json,hashlib,math,unicodedata
import numpy as np
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
testp=PROJECT/'data/processed/cross_encoder_TEST.json';test=json.loads(testp.read_text())
assert sha(testp)=='473d43d8376ba6ab828dfe92b84af2995d4555ee4b47e53364df7a8c83089558'
invalid=[i for i,r in enumerate(test) if r['positive']==r['negative'] or not r['negative'].strip()]
assert invalid==[1008,1014,1018,1019,1047,1079,1087,1088]
keep=np.ones(len(test),dtype=bool);keep[invalid]=False
values={};ties={};runs=[];versions=set();plans=set();settings=[]
for folder in sorted((ROOT/'remote_rerun_metadata').iterdir()):
 d=folder/folder.name;m=json.loads((d/'manifest.json').read_text());data=json.loads((d/'test_scores.json').read_text());cfg=json.loads((d/'final_model/config.json').read_text())
 assert m['status']=='COMPLETE' and m['optimizer_updates']==408 and m['training_pairs_seen']==26004
 assert data['test_sha256']==sha(testp) and data['weight_sha256']==m['weight_sha256']
 progress=[json.loads(x) for x in (d/'progress.jsonl').read_text().splitlines() if x.strip()]
 assert len(progress)==408 and [x['update'] for x in progress]==list(range(1,409))
 assert sum(x.get('pairs',x.get('samples',0)) for x in progress)==26004
 scores=data['scores'];assert len(scores)==1136
 correct=[];tied=[]
 for i,(r,s) in enumerate(zip(test,scores)):
  assert s['index']==i and str(s['doc_id'])==str(r['doc_id'])
  pos,neg=s['positive_logit'],s['negative_logit'];assert math.isfinite(pos) and math.isfinite(neg)
  c=pos>neg;t=pos==neg;assert bool(s['correct'])==c and bool(s['tie'])==t
  correct.append(c);tied.append(t)
 key=(m['run']['case'],m['run']['seed']);values[key]=np.asarray(correct,dtype=bool);ties[key]=np.asarray(tied,dtype=bool)
 versions.add(cfg['transformers_version']);plans.add(m['plan_sha256']);settings.append(m['settings'])
 header=json.loads((folder/'safetensors_header.json').read_text());ts=[v for k,v in header.items() if k!='__metadata__'];assert len(ts)==201 and {v['dtype'] for v in ts}=={'F32'}
 assert sum(math.prod(v['shape']) for v in ts)==278044417
 runs.append({'condition':key[0],'seed':key[1],'correct':sum(correct),'ties':sum(tied),'accuracy':sum(correct)/1136,'filtered_accuracy':float(values[key][keep].mean()),'half_tie_accuracy':float((values[key].astype(float)+.5*ties[key])[...].mean()),'test_scores_sha256':sha(d/'test_scores.json'),'declared_weight_sha256':m['weight_sha256'],'optimizer_updates':408,'pair_presentations':26004})
assert len(plans)==1 and all(x==settings[0] for x in settings)
expected={'R03':[1095,1103,1104],'R05':[1102,1102,1099],'R08':[1099,1100,1104]}
conditions=[]
for cond in ['R03','R05','R08']:
 rs=[r for seed in [42,123,999] for r in runs if r['condition']==cond and r['seed']==seed]
 assert [r['correct'] for r in rs]==expected[cond]
 conditions.append({'condition':cond,'correct_by_seed':expected[cond],'ties_by_seed':[r['ties'] for r in rs],'mean_accuracy':float(np.mean([r['accuracy'] for r in rs])),'population_sd':float(np.std([r['accuracy'] for r in rs],ddof=0)),'filtered_mean_accuracy':float(np.mean([r['filtered_accuracy'] for r in rs])),'half_tie_mean_accuracy':float(np.mean([r['half_tie_accuracy'] for r in rs]))})
tests=[]
for other in ['R05','R08']:
 for seed in [42,123,999]:
  a=values['R03',seed];b=values[other,seed];x=int(np.sum(a&~b));y=int(np.sum(b&~a));n=x+y
  p=min(1.,2*sum(math.comb(n,k) for k in range(min(x,y)+1))/2**n) if n else 1.
  tests.append({'reference':'R03','other':other,'seed':seed,'reference_only':x,'other_only':y,'p_exact_two_sided':p})
last=0
for j,i in enumerate(sorted(range(len(tests)),key=lambda i:tests[i]['p_exact_two_sided'])):
 last=max(last,min(1.,(len(tests)-j)*tests[i]['p_exact_two_sided']));tests[i]['p_holm']=last
sources=sorted({str(r['doc_id']) for r in test});members=[np.asarray([i for i,r in enumerate(test) if str(r['doc_id'])==s]) for s in sources];assert len(sources)==247
rng=np.random.default_rng(20261112);draws=rng.integers(0,len(sources),size=(20000,len(sources)))
size=np.asarray([len(ids) for ids in members]);contrasts=[]
for other in ['R05','R08']:
 delta=np.mean([values[other,seed].astype(float)-values['R03',seed].astype(float) for seed in [42,123,999]],axis=0)
 sums=np.asarray([delta[ids].sum() for ids in members]);boot=sums[draws].sum(axis=1)/size[draws].sum(axis=1)
 contrasts.append({'other_minus_R03':other,'mean_difference':float(delta.mean()),'exploratory_group_bootstrap_95_percentile_interval':np.percentile(boot,[2.5,97.5]).tolist(),'replicates':20000,'seed':20261112})
report={'status':'PASS','scope':'Recalculation of exported logits and run logs, plus structural weight headers. No fresh R03/R05/R08 inference or full large-weight checksum recomputation.','test_sha256':sha(testp),'plan_sha256_recorded':list(plans)[0],'returned_config_transformers_version':sorted(versions),'runs':runs,'conditions':conditions,'paired_tests':tests,'exploratory_source_group_bootstrap':contrasts,'invalid_test_rows_zero_based':invalid,'source_groups':247,'bootstrap_note':'Fresh exploratory reanalysis on 12 November with explicitly recorded seed; replaces the unrecorded-seed rounded bootstrap intervals in the uploaded draft.'}
(ROOT/'objective_score_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['status','conditions','paired_tests','exploratory_source_group_bootstrap']},indent=2))
