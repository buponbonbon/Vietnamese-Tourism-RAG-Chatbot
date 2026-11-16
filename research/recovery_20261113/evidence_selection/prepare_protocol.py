from pathlib import Path
import json,hashlib,datetime,shutil
import torch,transformers,numpy as np
ROOT=Path(__file__).resolve().parent;REC=ROOT.parent;PROJECT=REC.parents[1]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

assert not (ROOT/'protocol.json').exists()
assert json.loads((REC/'quality_weight_recovery.json').read_text())['status']=='COMPLETE'
source=REC/'bound_reviews_and_eligibility_001/reconstructed_inputs.json';raw=json.loads(source.read_text())
qs=[];pairs=[]
for q in raw:
 rows=[]
 for c in q['candidates']:
  rid=len(pairs);pairs.append({'row_id':rid,'query_id':q['benchmark_id'],'query':q['query'],'chunk_id':c['chunk_id'],'text':c['text'],'grade':c['grade'],'role':'native','bm25_rank':c['rank']});rows.append(rid)
 qs.append({k:v for k,v in q.items() if k not in ['candidates','oracle_addition']}|{'native_row_ids':rows,'expanded_row_ids':list(rows)})
for q,r in zip(qs,raw):
 if r['oracle_addition']:
  c=r['oracle_addition'];rid=len(pairs);pairs.append({'row_id':rid,'query_id':q['benchmark_id'],'query':q['query'],'chunk_id':c['chunk_id'],'text':c['text'],'grade':2,'role':'oracle_anchor','bm25_rank':None});q['expanded_row_ids'].append(rid)
assert len(qs)==140 and len(pairs)==795
assert len({(r['query_id'],r['chunk_id']) for r in pairs})==795
(ROOT/'cohort.json').write_text(json.dumps(qs,ensure_ascii=False,indent=2)+'\n')
(ROOT/'scoring_pairs.json').write_text(json.dumps(pairs,ensure_ascii=False,indent=2)+'\n')
old=json.loads((PROJECT/'data/wikiqa_external_eval_20261112_recovery_001/evaluation_protocol.json').read_text());models=old['models']
manifest={k.replace('\\','/'):v for k,v in json.loads((REC/'quality_source_metadata/upload_manifest.json').read_text()).items()}
for condition in ['Q','R']:
 for seed in [42,123,999]:
  name=f'{condition}_quality_mask_seed_{seed}';prefix='runs/'+name+'/final_model/'
  files={k[len(prefix):]:v for k,v in manifest.items() if k.startswith(prefix)}
  models.append({'name':name,'condition':condition,'seed':seed,'path':str(REC/'quality_models'/name/'final_model'),'file_sha256':files})
assert len(models)==13
for m in models:
 for name,digest in m['file_sha256'].items():assert sha(Path(m['path'])/name)==digest,(m['name'],name)
protocol={'schema':'restored-tourism-evidence-selection-v1','frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_artifacts':json.loads((REC/'bound_reviews_and_eligibility_001/eligibility_summary.json').read_text()),'cohort_sha256':sha(ROOT/'cohort.json'),'scoring_pairs_sha256':sha(ROOT/'scoring_pairs.json'),'models':models,'runtime':{'device':'cpu','dtype':'float32','torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'threads':4,'batch_size':4,'attention':'sdpa','padding':'dynamic_longest','max_length':512},'primary':{'contrast':'M1 minus M0','outcome':'Native-pool grade-2 Hit@1','queries':140,'uncertainty':'20000 source_group_id bootstrap replicates, seed 20261113; average paired fixed seed differences within query first'},'oracle':{'cohort':'All 95 eligible queries without consensus grade-2 evidence in the native top five','intervention':'Append the original reviewed consensus-grade-2 anchor; retain native order for ties. Leave the other 45 pools unchanged.','outcomes':['M1 minus M0 on oracle95','M1 minus M0 on all140 after intervention','Paired difference in oracle gains on the SAME full140 queries'],'role':'development diagnostic; not an independent blind test'},'K_sweep':{'cohort':'All 41 eligible native pools with exactly one grade-2 and four lower-grade candidates','K':[2,3,4,5],'subsets':'ALL combinations of K-1 distractors, each with the one sufficient passage. Preserve native row order for ties.','aggregation':'Mean subset selection rate within query, then mean query and mean fixed seeds. Do not count combinations as independent queries.','role':'diagnostic; not a new algorithm or independent tests'},'quality_control':{'contrast':'Q minus R','role':'development/mechanism diagnostic per archived quality plan; zero-mask exposure patterns also differ'},'ties':'Highest pointwise score; tied scores retain native order. Oracle anchor appended last, without optimistic tie preference.','inference_reuse':'Score each of the 795 unique query/passage inputs once per model; derive all pool and subset outcomes from these pointwise scores.','uncertainty_limits':'Conditional on these 3 existing fixed seeds and these source-grouped development queries. No population-of-seeds or equivalence claim. Diagnostic intervals unadjusted.','provenance_limits':['Exact original workbook, corpus, benchmark and BM25 hashes are recovered; no byte identity with the missing previous cohort export is asserted.','Scores are freshly re-inferred; missing old logits are not reconstructed from aggregate numbers.','R2 COMPLETED is a post-feedback return; agreement is not a new blind reliability estimate.','No labels changed or filled; NEEDS_REVIEW, rewrites and nonconsensus native pools excluded by declared rule.','Unavailability means absence of grade-2 evidence in the fixed BM25 top five, not corpus-wide unanswerability.','No generation, hallucination, user utility or deployable oracle claim.'],'script_hashes':{p.name:sha(p) for p in ROOT.glob('*.py')}}
(ROOT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
(ROOT/'STATUS.json').write_text(json.dumps({'phase':'PREPARED','models_complete':0,'total_models':13,'protocol_sha256':sha(ROOT/'protocol.json')},indent=2))
print('PROTOCOL_FROZEN',sha(ROOT/'protocol.json'),'COHORT',sha(ROOT/'cohort.json'),'PAIRS',sha(ROOT/'scoring_pairs.json'))
