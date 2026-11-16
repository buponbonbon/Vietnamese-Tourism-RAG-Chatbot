"""CPU FP32 inference only. Resume matching partial caches; never train."""
from pathlib import Path
import json,hashlib,time,os,gc,fcntl,math,sys,platform
import torch,transformers,numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification
ROOT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def atomic(p,obj):
 temp=p.with_suffix(p.suffix+'.tmp')
 with temp.open('w') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 os.replace(temp,p)

def main():
 protocol=json.loads((ROOT/'evaluation_protocol.json').read_text())
 assert sha(ROOT/'evaluation_inputs.json')==protocol['input_sha256']
 assert sha(__file__)==protocol['pre_inference_script_sha256'][Path(__file__).name]
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 torch.manual_seed(0);torch.use_deterministic_algorithms(True)
 runtime={'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'python':platform.python_version(),'device':'cpu','dtype':'float32','threads':torch.get_num_threads(),'batch_size':4,'max_length':512,'attention':'sdpa','padding':'dynamic_longest','deterministic_algorithms':True}
 queries=json.loads((ROOT/'evaluation_inputs.json').read_text())
 flat=[{'query_id':q['query_id'],'question':q['question'],**c} for q in queries for c in q['candidates']]
 assert len(flat)==2341
 reference_tokens=None;finished=[]
 start_all=time.monotonic()
 for mi,spec in enumerate(protocol['models'],1):
  name=spec['name'];path=Path(spec['path']);dest=ROOT/'inference'/name;dest.mkdir(parents=True,exist_ok=True)
  for fn,digest in spec['file_sha256'].items():assert sha(path/fn)==digest,(name,fn)
  print('VERIFY',mi,7,name,flush=True)
  tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
  encoded=tok([r['question'] for r in flat],[r['text'] for r in flat],padding=False,truncation=True,max_length=512)
  uncapped=tok([r['question'] for r in flat],[r['text'] for r in flat],padding=False,truncation=False)
  token_hash=hashlib.sha256(json.dumps(dict(encoded),sort_keys=True,separators=(',',':')).encode()).hexdigest()
  if reference_tokens is None:reference_tokens=token_hash
  assert token_hash==reference_tokens,(name,'different tokenization')
  token_manifest={'sha256':token_hash,'minimum_tokens':min(map(len,uncapped['input_ids'])),'maximum_tokens':max(map(len,uncapped['input_ids'])),'truncated_rows':sum(len(a)!=len(b) for a,b in zip(encoded['input_ids'],uncapped['input_ids']))}
  assert token_manifest['truncated_rows']==0
  identity={'protocol_sha256':sha(ROOT/'evaluation_protocol.json'),'input_sha256':protocol['input_sha256'],'scorer_sha256':sha(__file__),'model_name':name,'weight_sha256':spec['file_sha256']['model.safetensors'],'tokenization':token_manifest,'runtime':runtime}
  output=dest/'scores.json';partial=dest/'partial_scores.json'
  if output.exists():
   cached=json.loads(output.read_text());assert cached['identity']==identity and cached['status']=='COMPLETE'
   assert len(cached['scores'])==len(flat) and all(math.isfinite(x) for x in cached['scores'])
   finished.append({'model_name':name,'scores_sha256':sha(output),'rows':len(flat)})
   print('REUSE_COMPLETE',mi,7,name,flush=True);continue
  scores=[]
  if partial.exists():
   cached=json.loads(partial.read_text());assert cached['identity']==identity
   scores=cached['scores'];assert 0<=len(scores)<=len(flat)
   assert len(scores)%4==0 or len(scores)==len(flat)
   assert all(math.isfinite(x) for x in scores)
  model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='sdpa').to('cpu').eval()
  def forward(indices):
   batch=tok.pad([{k:encoded[k][i] for k in encoded} for i in indices],padding=True,return_tensors='pt')
   with torch.inference_mode():vals=model(**batch).logits.float().reshape(-1).tolist()
   assert len(vals)==len(indices) and all(math.isfinite(x) for x in vals)
   return vals
  canary_n=16
  one=[forward([i])[0] for i in range(canary_n)]
  four=[x for st in range(0,canary_n,4) for x in forward(range(st,min(st+4,canary_n)))]
  maxdiff=max(abs(a-b) for a,b in zip(one,four));assert maxdiff<=1e-4,(name,maxdiff)
  batch_check={'kind':'Same restored model and first 16 WikiQA rows: batch 1 versus batch 4; not a comparison to missing previous score exports','maximum_absolute_difference':maxdiff,'tolerance':1e-4,'status':'PASS'}
  initial=len(scores);start=time.monotonic()
  for st in range(initial,len(flat),4):
   scores.extend(forward(range(st,min(st+4,len(flat)))))
   if len(scores)%128==0 or len(scores)==len(flat):
    elapsed=time.monotonic()-start;done=len(scores)-initial
    eta=elapsed*(len(flat)-len(scores))/done if done else None
    atomic(partial,{'identity':identity,'scores':scores,'rows_done':len(scores),'status':'PARTIAL'})
    atomic(ROOT/'STATUS.json',{'phase':'SCORING','model':name,'model_number':mi,'models_complete':len(finished),'total_models':7,'rows_done':len(scores),'rows_total':len(flat),'current_process_elapsed_seconds':elapsed,'estimated_remaining_model_seconds':eta})
    print('PROGRESS',mi,7,name,len(scores),len(flat),'elapsed_s',round(elapsed,1),'model_eta_s',round(eta,1) if eta is not None else None,flush=True)
  result={'status':'COMPLETE','identity':identity,'batch_consistency_check':batch_check,'elapsed_this_process_seconds':time.monotonic()-start,'rows_resumed':initial,'scores':scores}
  atomic(output,result)
  finished.append({'model_name':name,'scores_sha256':sha(output),'rows':len(flat)})
  atomic(ROOT/'STATUS.json',{'phase':'SCORING','models_complete':len(finished),'total_models':7,'last_completed_model':name})
  print('MODEL_COMPLETE',mi,7,name,'elapsed_s',round(result['elapsed_this_process_seconds'],1),'batch_check_maxdiff',maxdiff,flush=True)
  del model,tok,encoded,uncapped;gc.collect()
 atomic(ROOT/'inference_completion.json',{'status':'COMPLETE','models':finished,'protocol_sha256':sha(ROOT/'evaluation_protocol.json'),'runtime':runtime,'elapsed_current_invocation_seconds':time.monotonic()-start_all})
 atomic(ROOT/'STATUS.json',{'phase':'INFERENCE_COMPLETE','models_complete':7,'total_models':7,'analysis_complete':False})
 print('ALL_SEVEN_INFERENCE_COMPLETE',flush=True)

if __name__=='__main__':
 with (ROOT/'.scoring.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  main()
