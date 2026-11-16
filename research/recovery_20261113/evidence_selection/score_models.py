"""Pointwise CPU FP32 scoring of reconstructed native+oracle pairs; never train."""
from pathlib import Path
import torch,transformers,numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import json,hashlib,os,time,gc,fcntl,math,platform
ROOT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def atomic(path,obj):
 temp=path.with_suffix(path.suffix+'.tmp')
 with temp.open('w') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 os.replace(temp,path)

def main():
 p=json.loads((ROOT/'protocol.json').read_text());pairs=json.loads((ROOT/'scoring_pairs.json').read_text())
 assert sha(ROOT/'scoring_pairs.json')==p['scoring_pairs_sha256']
 assert sha(__file__)==p['script_hashes']['score_models.py']
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
 runtime={'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'device':'cpu','dtype':'float32','attention':'sdpa','threads':4,'batch_size':4,'padding':'dynamic_longest','max_length':512,'deterministic_algorithms':True}
 common=None;finished=[];suite_start=time.monotonic()
 for mi,spec in enumerate(p['models'],1):
  name=spec['name'];modeldir=Path(spec['path']);dest=ROOT/'inference'/name;dest.mkdir(parents=True,exist_ok=True)
  print('VERIFY_MODEL',mi,len(p['models']),name,flush=True)
  for fn,digest in spec['file_sha256'].items():assert sha(modeldir/fn)==digest,(name,fn)
  tokenizer=AutoTokenizer.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False,use_fast=True)
  encoded=tokenizer([r['query'] for r in pairs],[r['text'] for r in pairs],padding=False,truncation=True,max_length=512)
  uncapped=tokenizer([r['query'] for r in pairs],[r['text'] for r in pairs],padding=False,truncation=False)
  token_hash=hashlib.sha256(json.dumps(dict(encoded),sort_keys=True,separators=(',',':')).encode()).hexdigest()
  if common is None:common=token_hash
  assert common==token_hash,(name,'Tokenization differs across models')
  lengths=list(map(len,uncapped['input_ids']));truncated=[i for i,(a,b) in enumerate(zip(encoded['input_ids'],uncapped['input_ids'])) if len(a)!=len(b)]
  tokenization={'sha256':token_hash,'uncapped_min_length':min(lengths),'uncapped_max_length':max(lengths),'truncated_row_ids':truncated,'truncated_grade2_rows':sum(pairs[i]['grade']==2 for i in truncated)}
  identity={'protocol_sha256':sha(ROOT/'protocol.json'),'pairs_sha256':sha(ROOT/'scoring_pairs.json'),'scorer_sha256':sha(__file__),'model':name,'weight_sha256':spec['file_sha256']['model.safetensors'],'runtime':runtime,'tokenization':tokenization}
  complete=dest/'scores.json';partial=dest/'partial_scores.json'
  if complete.exists():
   o=json.loads(complete.read_text());assert o['identity']==identity and o['status']=='COMPLETE' and len(o['scores'])==len(pairs)
   assert all(math.isfinite(x) for x in o['scores']);finished.append({'model':name,'sha256':sha(complete)});print('REUSED_COMPLETE_MODEL',mi,name,flush=True);continue
  scores=[]
  if partial.exists():
   o=json.loads(partial.read_text());assert o['identity']==identity;scores=o['scores'];assert len(scores)%4==0 or len(scores)==len(pairs)
  model=AutoModelForSequenceClassification.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='sdpa').to('cpu').eval()
  def forward(ids):
   batch=tokenizer.pad([{k:encoded[k][i] for k in encoded} for i in ids],padding=True,return_tensors='pt')
   with torch.inference_mode():v=model(**batch).logits.float().reshape(-1).tolist()
   assert len(v)==len(ids) and all(math.isfinite(x) for x in v)
   return v
  a=[forward([i])[0] for i in range(8)];b=[v for st in range(0,8,4) for v in forward(range(st,st+4))]
  difference=max(abs(x-y) for x,y in zip(a,b));assert difference<=1e-4,(name,difference)
  start=time.monotonic();initial=len(scores)
  for st in range(initial,len(pairs),4):
   scores.extend(forward(range(st,min(st+4,len(pairs)))))
   if len(scores)%128==0 or len(scores)==len(pairs):
    elapsed=time.monotonic()-start;done=len(scores)-initial;eta=elapsed*(len(pairs)-len(scores))/done if done else None
    atomic(partial,{'identity':identity,'status':'PARTIAL','scores':scores})
    atomic(ROOT/'STATUS.json',{'phase':'SCORING','models_complete':len(finished),'total_models':len(p['models']),'current_model':name,'rows_done':len(scores),'rows_total':len(pairs),'model_elapsed_seconds':elapsed,'estimated_remaining_model_seconds':eta})
    print('PROGRESS',mi,len(p['models']),name,len(scores),len(pairs),'elapsed_s',round(elapsed,1),'model_eta_s',round(eta,1) if eta is not None else None,flush=True)
  o={'status':'COMPLETE','identity':identity,'batch_consistency_max_abs_difference':difference,'elapsed_current_process_seconds':time.monotonic()-start,'resumed_rows':initial,'scores':scores}
  atomic(complete,o);finished.append({'model':name,'sha256':sha(complete)})
  print('MODEL_COMPLETE',mi,len(p['models']),name,'seconds',round(o['elapsed_current_process_seconds'],1),'batch_diff',difference,flush=True)
  atomic(ROOT/'STATUS.json',{'phase':'SCORING','models_complete':len(finished),'total_models':len(p['models']),'last_completed_model':name})
  del model,tokenizer,encoded,uncapped;gc.collect()
 atomic(ROOT/'inference_completion.json',{'status':'COMPLETE','models':finished,'protocol_sha256':sha(ROOT/'protocol.json'),'runtime':runtime,'elapsed_current_invocation_seconds':time.monotonic()-suite_start})
 atomic(ROOT/'STATUS.json',{'phase':'INFERENCE_COMPLETE','models_complete':len(finished),'total_models':len(p['models'])})
 print('ALL_THIRTEEN_MODEL_SCORES_COMPLETE',flush=True)

if __name__=='__main__':
 with (ROOT/'.scoring.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
