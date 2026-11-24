"""Freeze a replacement evaluation after the previous cloud runtime was lost."""
from pathlib import Path
from collections import defaultdict,Counter
import json,hashlib,shutil,re,unicodedata,datetime
import pyarrow.parquet as pq
import torch,transformers,numpy as np,platform
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
REC=PROJECT/'data/recovery_20261112_001'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def write(name,obj):
 p=ROOT/name
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

assert not (ROOT/'evaluation_protocol.json').exists()
recovery=json.loads((REC/'checkpoint_recovery_completion.json').read_text());assert recovery['status']=='COMPLETE'
base=json.loads((REC/'base_model_recovery.json').read_text());assert base['status']=='PASS'
source=REC/'wikiqa_sources/test.parquet'
assert sha(source)=='eb309da359eda13c4a68fcd0dbff5cc7fc4b73d8fda2585d162606dc035b9c48'
shutil.copyfile(source,ROOT/'wikiqa_test.parquet')
rows=pq.read_table(source).to_pylist()
groups=defaultdict(list)
for i,row in enumerate(rows):
 assert row['label'] in (0,1)
 groups[row['question_id']].append({'test_row_index':i,**row})
classes=Counter();queries=[]
for qid,rs in groups.items():
 labels={r['label'] for r in rs}
 category='mixed' if labels=={0,1} else ('no_positive' if labels=={0} else 'all_positive')
 classes[category]+=1
 if category!='mixed':continue
 assert len({r['question'] for r in rs})==len({r['document_title'] for r in rs})==1
 queries.append({'query_id':qid,'question':rs[0]['question'],'source_title':rs[0]['document_title'],'candidates':[{'test_row_index':r['test_row_index'],'candidate_order':j,'text':r['answer'],'label':r['label']} for j,r in enumerate(rs)]})
assert len(rows)==6165 and len(groups)==633
assert dict(classes)=={'no_positive':390,'mixed':237,'all_positive':6},classes
assert sum(len(q['candidates']) for q in queries)==2341
assert len({q['source_title'] for q in queries})==234
write('evaluation_inputs.json',queries)
norm=lambda s:' '.join(unicodedata.normalize('NFKC',s).casefold().split())
source_inputs=[];old_queries=set();old_passages=set()
expected={'train_moc_13.json':'cf28b8bf4fded1337b5e8e31bf310083e6535a73cf9d07219300ce217e7a0a16','cross_encoder_VAL.json':'ecb977cfae667f0654cfabc945d4da6a2ea869e996138bd75a359d18a093bab3','cross_encoder_TEST.json':'473d43d8376ba6ab828dfe92b84af2995d4555ee4b47e53364df7a8c83089558'}
for name,digest in expected.items():
 p=PROJECT/'data/processed'/name;assert sha(p)==digest
 ds=json.loads(p.read_text())
 old_queries.update(norm(r['query']) for r in ds)
 old_passages.update(norm(r[key]) for r in ds for key in ('positive','negative'))
 source_inputs.append({'file':name,'rows':len(ds),'sha256':digest})
q_overlap=sum(norm(q['question']) in old_queries for q in queries)
p_overlap=sum(norm(c['text']) in old_passages for q in queries for c in q['candidates'])
duplicate_rows=[]
for q in queries:
 seen={}
 for c in q['candidates']:
  if c['text'] in seen:
   duplicate_rows.append({'query_id':q['query_id'],'first_row':seen[c['text']]['test_row_index'],'duplicate_row':c['test_row_index'],'same_label':seen[c['text']]['label']==c['label']})
  else:seen[c['text']]=c
integrity={'status':'PASS','native_queries':len(groups),'native_rows':len(rows),'query_categories':dict(classes),'selected_queries':len(queries),'selected_rows':2341,'source_title_groups':234,'same_query_exact_duplicate_sentence_rows_retained':duplicate_rows,'overlap_checks':{'normalization':'Unicode NFKC, casefold, whitespace collapse','tourism_inputs':source_inputs,'exact_query_overlap':q_overlap,'exact_passage_overlap':p_overlap,'scope':'Only the three recovered hashed tourism pair files; no semantic or pretraining-exposure claim; later tourism benchmark exports were not recovered.'},'native_order_preserved':True,'per_candidate_filtering':False}
write('input_integrity_check.json',integrity)
manifest={k.replace('\\','/'):v for k,v in json.loads((REC/'matched_source_metadata/upload_manifest.json').read_text()).items()}
models=[]
base_path=Path(base['path'])
models.append({'name':'BGE_pretrained','condition':'BGE','seed':None,'path':str(base_path),'file_sha256':{p.name:sha(p) for p in base_path.iterdir() if p.is_file()}})
for condition in ('M0','M1'):
 for seed in (42,123,999):
  name=f'{condition}_fixed_1p0_weighted_matched_seed_{seed}'
  p=REC/'matched_recovered/runs'/name/'final_model'
  files={k.rsplit('/',1)[1]:digest for k,digest in manifest.items() if k.startswith('runs/'+name+'/final_model/')}
  assert len(files)==4
  for fn,digest in files.items():assert sha(p/fn)==digest,(name,fn)
  models.append({'name':name,'condition':condition,'seed':seed,'path':str(p),'file_sha256':files})
protocol={'schema':'wikiqa-external-cpu-recovery-v1','frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset':{'repository':'microsoft/wiki_qa','revision':'3f104672b5de699878fe7907afc486f0de325eb5','test_parquet_sha256':sha(source),'cohort':'All native test queries with at least one positive and one negative; all their native sentence candidates retained','queries':237,'candidate_rows':2341,'source_title_groups':234,'input':'question and candidate sentence; title only used for uncertainty grouping','tie_rule':'Equal logits retain original native test-row order'},'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'python':platform.python_version(),'device':'cpu','dtype':'float32','threads':4,'batch_size':4,'padding':'dynamic_longest','maximum_length':512,'attention':'sdpa','optimization':False},'models':models,'primary':{'contrast':'M1 minus M0','metric':'query MRR','aggregation':'Mean over paired seeds 42, 123, 999 within each query, then query mean','bootstrap_unit':'provided document_title','bootstrap_replicates':20000,'bootstrap_seed':20261111,'confidence':'95% percentile interval with group resampling; query-weighted within each sampled collection of source groups'},'secondary':{'metrics':['MAP','Hit@1'],'intervals':'Exploratory unadjusted; cannot replace primary MRR based on observed results'},'uncertainty_scope':'Conditional on the existing fixed three seeds, selected WikiQA cohort and these checkpoints; does not estimate future training-seed uncertainty.','input_sha256':sha(ROOT/'evaluation_inputs.json'),'pre_inference_script_sha256':{p.name:sha(p) for p in ROOT.glob('*.py')},'recovery':'This is a fresh full seven-model inference after restoration. The previous partial WikiQA score exports and scoring scripts were not found in the restored snapshot; no previous partial result is treated as completed or compared numerically. The primary metric, contrast, cohort, seeds and bootstrap settings follow the preserved discussion. This is not a preregistration.','interpretation_limits':['English answer-sentence ranking is different from Vietnamese tourism grade-2 evidence sufficiency.','Language, domain and passage granularity change jointly; their causal effects are not separated.','No new corpus training, generated-answer evaluation or hallucination reduction claim.']}
write('evaluation_protocol.json',protocol)
card=(REC/'wikiqa_sources/README.md').read_text()
shutil.copyfile(REC/'wikiqa_sources/README.md',ROOT/'WIKIQA_ORIGINAL_DATA_CARD.md')
license_text=card.split('### Licensing Information',1)[1].split('### Citation Information',1)[0]
(ROOT/'WIKIQA_LICENSE_AND_MODIFICATIONS.md').write_text('# WikiQA source license and modification notice\n\nAttribution: Microsoft Research; Yi Yang, Wen-tau Yih and Christopher Meek (EMNLP 2015).\n\nModified subset and JSON representation, 12 November 2026: retain all 237 mixed-label native test query pools (2,341 rows), preserve all their sentences, labels and row order, add row IDs and source grouping. This is a modified evaluation representation, not the original dataset. Original full test Parquet and original card are also preserved. The following original license terms apply to the dataset and derived data; this file does not grant additional rights.\n\n'+license_text.strip()+'\n')
citation=re.search(r'```\s*(@inproceedings\{[\s\S]*?)```',card.split('### Citation Information',1)[1]).group(1).strip()
(ROOT/'references_wikiqa.bib').write_text(citation+'\n')
write('STATUS.json',{'phase':'PREPARED','models_complete':0,'total_models':7,'protocol_sha256':sha(ROOT/'evaluation_protocol.json')})
print('PROTOCOL_FROZEN',sha(ROOT/'evaluation_protocol.json'),'INPUTS',sha(ROOT/'evaluation_inputs.json'))
print(json.dumps(integrity,ensure_ascii=False,indent=2))
