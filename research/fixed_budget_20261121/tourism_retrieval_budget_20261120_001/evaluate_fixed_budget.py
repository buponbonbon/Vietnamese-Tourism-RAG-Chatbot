"""Full-corpus dense + frozen BM25 at the same fixed cross-encoder budget."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import unicodedata
from collections import Counter

os.environ.setdefault('RAYON_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import faiss
import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
REPO = Path('/workspace/Vietnamese-Tourism-RAG-Chatbot')
REC = REPO/'data/oracle_recovery_20261113_001'
ARC = REC/'evidence_selection_reinference_001'
POOL_FILE = ARC/'source_inputs/native_bm25_top50.csv'
CORPUS_FILE = ARC/'source_inputs/vietnam_tourism_rag_dataset.csv'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        while b := f.read(1<<22):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text())


def save(p,data):
    with p.open('x',encoding='utf-8') as f:
        json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')


def csv_rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:
        yield from csv.DictReader(f)


def norm(s):
    return ' '.join(unicodedata.normalize('NFC',s).split())


def load():
    frozen=read(ARC/'protocol.json')
    for f,key in ((POOL_FILE,'native_bm25_top50'),(CORPUS_FILE,'corpus')):
        assert sha(f)==frozen['source_artifacts']['input_hashes'][key]
    for key in ('cohort','scoring_pairs'):
        assert sha(ARC/(key+'.json'))==frozen[key+'_sha256']
    qs=read(ARC/'cohort.json')
    assert len(qs)==140 and len({q['source_group_id'] for q in qs})==81
    byq={q['benchmark_id']:q for q in qs}
    docs={}
    for doc in csv_rows(CORPUS_FILE):
        key=doc['chunk_id']
        assert key not in docs and doc['text_content'],key
        docs[key]=doc['text_content']
    assert len(docs)==56984
    lexical={k:[] for k in byq}
    for r in csv_rows(POOL_FILE):
        key=r['benchmark_id']
        if key in lexical:
            did=r['candidate_chunk_id']
            assert norm(r['query'])==norm(byq[key]['query'])
            assert norm(r['candidate_text'])==norm(docs[did])
            lexical[key].append((int(r['rank']),did))
    for q,rows in lexical.items():
        pairs=sorted(rows)
        assert [r for r,_ in pairs]==list(range(1,51))
        assert len({d for _,d in pairs})==50
        lexical[q]=[d for _,d in pairs]
    known={}
    source_label_hashes={}
    folder=REC/'bound_reviews_and_eligibility_001'
    reviews=[]
    for i in (1,2):
        f=folder/f'reviewer_{i:02d}_passages.csv'
        idx={}
        for r in csv_rows(f):
            if r['benchmark_id'] not in byq or r['review_status']!='DONE' or r['relevance_grade'] not in ('0','1','2'):
                continue
            k=(r['benchmark_id'],r['chunk_id'])
            assert k not in idx
            idx[k]=r
        reviews.append(idx)
        source_label_hashes[f.name]=sha(f)
    for k in reviews[0].keys() & reviews[1].keys():
        a,b=[m[k] for m in reviews]
        if a['relevance_grade']!=b['relevance_grade']:
            continue
        assert norm(a['query'])==norm(b['query'])==norm(byq[k[0]]['query'])
        assert norm(a['candidate_evidence'])==norm(b['candidate_evidence'])==norm(docs[k[1]])
        known[k]=int(a['relevance_grade'])
    original=read(ARC/'scoring_pairs.json')
    for r in original:
        k=(r['query_id'],r['chunk_id'])
        assert norm(r['query'])==norm(byq[k[0]]['query'])
        assert norm(r['text'])==norm(docs[k[1]])
        assert known.get(k,r['grade'])==r['grade']
        known[k]=r['grade']
    for q,rows in lexical.items():
        assert all((q,d) in known for d in rows[:5])
        assert any(known[q,d]==2 for d in rows[:5])==byq[q]['has_grade2_in_native_pool']
    return frozen,qs,docs,lexical,known,original,source_label_hashes


def dense_rank(qs,docs,settings):
    model=ROOT/'dense_model'
    manifest=read(ROOT/'dense_download_manifest.json')
    for file,r in manifest.items():
        assert sha(model/file)==r['sha256']
    assert read(model/'config.json')['normalize']
    tokenizer=Tokenizer.from_file(str(model/'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=512)
    tensors=load_file(str(model/'model.safetensors'))
    print('dense tensor shapes',[(k,v.shape,str(v.dtype)) for k,v in tensors.items()],flush=True)
    assert set(tensors)=={'embeddings'},set(tensors)
    embedding=tensors['embeddings']
    assert embedding.shape==(tokenizer.get_vocab_size(),256) and embedding.dtype==np.float32,embedding.shape
    unk=tokenizer.token_to_id('<unk>')
    def encode(seq):
        out=np.zeros((len(seq),256),dtype=np.float32)
        for i,item in enumerate(tokenizer.encode_batch_fast(seq,add_special_tokens=False)):
            ids=[x for x in item.ids if x!=unk]
            if ids:
                v=embedding[ids].mean(axis=0)
                out[i]=v/(np.linalg.norm(v)+1e-32)
        return out
    names=list(docs)
    index=faiss.IndexFlatIP(256)
    faiss.omp_set_num_threads(4)
    started=time.perf_counter()
    for start in range(0,len(names),512):
        vec=encode([docs[n] for n in names[start:start+512]])
        assert np.isfinite(vec).all()
        index.add(vec)
        done=min(start+512,len(names))
        if start%8192==0 or done==len(names):
            print('dense index',done,len(names),round(time.perf_counter()-started,2),flush=True)
    build_seconds=time.perf_counter()-started
    assert index.ntotal==56984
    query_start=time.perf_counter()
    qvec=encode([q['query'] for q in qs])
    assert np.all(np.linalg.norm(qvec,axis=1)>.99)
    ranked_scores,ranked_rows=index.search(qvec,50)
    seconds=time.perf_counter()-query_start
    assert np.isfinite(ranked_scores).all() and (ranked_rows>=0).all()
    out={}
    for q, scores,rows in zip(qs,ranked_scores,ranked_rows,strict=True):
        arr=sorted(zip(scores,rows,strict=True),key=lambda x:(-float(x[0]),int(x[1])))
        out[q['benchmark_id']]=[names[j] for _,j in arr]
        assert len(set(out[q['benchmark_id']]))==50
    faiss.write_index(index,str(ROOT/'potion_full_corpus.index'))
    save(ROOT/'index_mapping.json',{'corpus_sha256':sha(CORPUS_FILE),'chunk_ids':names,
         'embeddings_revision':settings['semantic_input']['revision'],
         'index_sha256':sha(ROOT/'potion_full_corpus.index')})
    return out,{'corpus_documents':index.ntotal,'build_seconds':build_seconds,'batched_140_queries_encode_search_seconds':seconds,
              'dimension':256,'model_version':settings['semantic_input']['revision']}


def rrf(a,b):
    ar={d:i for i,d in enumerate(a,1)}
    br={d:i for i,d in enumerate(b,1)}
    def order(d):
        return (-(1/(60+ar[d]) if d in ar else 0)-(1/(60+br[d]) if d in br else 0),
                ar.get(d,51),br.get(d,51),d)
    return sorted(ar.keys()|br.keys(),key=order)[:50]


def grade_bits(grade):
    return (0,1) if grade is None else (int(grade==2),)*2


def availability(qid,pool,known):
    vals=[grade_bits(known.get((qid,d))) for d in pool]
    return tuple(max(a) for a in zip(*vals))


def paired_hits(a,b,known):
    lower=upper=0
    new=set()
    equal=0
    for q in a:
        if a[q]==b[q]:
            equal+=1
            continue
        ka,kb=(q,a[q]),(q,b[q])
        x,y=grade_bits(known.get(ka)),grade_bits(known.get(kb))
        lower+=x[0]-y[1]
        upper+=x[1]-y[0]
        new.update(k for k in (ka,kb) if k not in known)
    return {'bounds_count':[lower,upper],'same_winner_count':equal,'unique_unknown_winners_needed_for_exact_difference':len(new)}


def eval_retrievers(qs,pools,known,settings):
    d={}
    qids=[q['benchmark_id'] for q in qs]
    for k in settings['availability_only_depths']:
        groups={}
        for p,rs in pools.items():
            bounds=[availability(q,rs[q][:k],known) for q in qids]
            groups[p]={'available_query_count_bounds':list(map(sum,zip(*bounds))),
                       'known_pairs':sum((q,x) in known for q in qids for x in rs[q][:k]),
                       'total_pairs':len(qids)*k}
        d[k]=groups
    return d


def rerank(qs,docs,pools,known,original,frozen,settings):
    import torch, transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    k=settings['pool_depth']
    qids=[q['benchmark_id'] for q in qs]
    qtext={q['benchmark_id']:q['query'] for q in qs}
    row_key=sorted({(q,d) for pool in pools.values() for q in qids for d in pool[q][:k]})
    previous={(r['query_id'],r['chunk_id']):r for r in original}
    new=[i for i,x in enumerate(row_key) if x not in previous]
    binary_tokenizer=None
    cell={}
    for retriever,pool in pools.items():
        cell[retriever+'_rank1']={q:pool[q][0] for q in qids}
    calc_by_model={}
    for name in settings['reranking']['models']:
        ref=next(r for r in frozen['models'] if r['name']==name)
        model_dir=Path(ref['path'])
        for file,digest in ref['file_sha256'].items():
            assert sha(model_dir/file)==digest,(name,file)
        cache_path=ARC/'inference'/name/'scores.json'
        completion=read(ARC/'inference_completion.json')
        assert sha(cache_path)==next(r['sha256'] for r in completion['models'] if r['model']==name)
        scores_cache=read(cache_path)['scores']
        tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
        # Preserve the exact archived input strings for cache hits.
        inputs=[(previous[q,d]['query'],previous[q,d]['text']) if (q,d) in previous else (qtext[q],docs[d]) for q,d in row_key]
        tx=[x[0] for x in inputs];dx=[x[1] for x in inputs]
        enc=tokenizer(tx,dx,truncation=True,max_length=512,padding=False)
        uncapped=tokenizer(tx,dx,truncation=False,padding=False)
        hash_tokens=hashlib.sha256(json.dumps(dict(enc),sort_keys=True).encode()).hexdigest()
        if binary_tokenizer is None:binary_tokenizer=hash_tokens
        assert binary_tokenizer==hash_tokens
        model=AutoModelForSequenceClassification.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='sdpa').eval()

        def forward(ix):
            batch=tokenizer.pad([{key:enc[key][i] for key in enc} for i in ix],return_tensors='pt')
            with torch.inference_mode():
                vals=model(**batch).logits.float().reshape(-1).tolist()
            assert len(vals)==len(ix) and all(math.isfinite(x) for x in vals)
            return vals
        probe=[i for i,r in enumerate(row_key) if r in previous][:8]
        check=max(abs(v-scores_cache[previous[row_key[i]]['row_id']])
                  for st in (0,4) for i,v in zip(probe[st:st+4],forward(probe[st:st+4]),strict=True))
        assert check<=1e-4,(name,check)
        score={key:scores_cache[previous[key]['row_id']] for key in row_key if key in previous}
        started=time.perf_counter()
        for st in range(0,len(new),4):
            for i,v in zip(new[st:st+4],forward(new[st:st+4]),strict=True):
                score[row_key[i]]=v
            if (st+4)%112==0 or st+4>=len(new):
                print(name,'new pairs',min(st+4,len(new)),len(new),'seconds',round(time.perf_counter()-started,1),flush=True)
        elapsed=time.perf_counter()-started
        assert set(score)==set(row_key)
        save(ROOT/(name+'_scores.json'),{'model_weight_sha256':ref['file_sha256']['model.safetensors'],
             'scores':[{'q':q,'d':d,'value':score[q,d],'reused':(q,d) in previous} for q,d in row_key]})
        index={x:i for i,x in enumerate(row_key)}
        for pname,pool in pools.items():
            cell[name+'_'+pname]={q:max(pool[q][:k],key=lambda d:score[q,d]) for q in qids}
        calc_by_model[name]={'new_pairs':len(new),'cached_pairs':len(row_key)-len(new),
            'archived_logit_8_row_max_abs_error':check,'new_forward_seconds':elapsed,'per_pool_cost':{}}
        for pname,pool in pools.items():
            ii=[index[q,d] for q in qids for d in pool[q][:k]]
            calc_by_model[name]['per_pool_cost'][pname]={
                'cross_encoder_pairs':len(ii),'actual_unpadded_input_tokens':sum(len(enc['input_ids'][j]) for j in ii),
                'truncated_pairs':sum(len(uncapped['input_ids'][j])>512 for j in ii)}
        del model
    # Compute sharp bounds before consulting any new grade.
    results={}
    for name,winners in cell.items():
        g=[known.get((q,x)) for q,x in winners.items()]
        results[name]={'hit1_query_count_bounds':list(map(sum,zip(*map(grade_bits,g)))),
                       'known_grade_counts':{str(x):g.count(x) for x in (0,1,2,None)}}
    comparisons=[('rrf_rank1','bm25_rank1')]
    for name in settings['reranking']['models']:
        comparisons.append((name+'_rrf',name+'_bm25'))
    for pn in pools:
        names=settings['reranking']['models']
        comparisons.append((names[1]+'_'+pn,names[0]+'_'+pn))
    return cell, {'cells':results,'paired_contrasts':{l+' minus '+r:paired_hits(cell[l],cell[r],known) for l,r in comparisons},
                  'reranker_cost_and_cache':calc_by_model,
                  'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'device':'CPU','dtype':'FP32','batch':4,'threads':4,'tokenization_sha256':binary_tokenizer}}


def main():
    settings=read(ROOT/'PROTOCOL.json')
    frozen,qs,docs,lexical,known,original,label_hashes=load()
    dense,build=dense_rank(qs,docs,settings)
    pools={'bm25':lexical,'rrf':{q:rrf(lexical[q],dense[q]) for q in lexical}}
    lens=eval_retrievers(qs,pools,known,settings)
    save(ROOT/'retrieval_rankings.json',{'pools':pools,'dense':dense})
    k=settings['pool_depth']
    label_counts={}
    for name,pool in pools.items():
        aa=[availability(q,rs[:k],known) for q,rs in pool.items()]
        label_counts[name]=aa
    assert not any(a!=b for a,b in label_counts['bm25'])
    before={'exact_available_BM25_5':sum(x[0] for x in label_counts['bm25']),
            'RRF_minus_BM25_availability_bounds_count':[sum(
                label_counts['rrf'][i][t]-a[0] for i,a in enumerate(label_counts['bm25'])) for t in (0,1)],
            'queries_with_same_pool_members':sum(set(lexical[q][:5])==set(pools['rrf'][q][:5]) for q in lexical)}
    partial={'queries':len(qs),'source_groups':len({q['source_group_id'] for q in qs}),'consensus_known_query_passages':len(known),
             'label_csv_sha256':label_hashes,'protocol_sha256':sha(ROOT/'PROTOCOL.json'),
             'candidate_summary_by_k':lens,'primary_availability':before,'dense_runtime':build,
             'limits':['A new static dense branch, not the BGE-M3 production index.',
                       'Fixed development queries with missing judgments; absence of known sufficiency need not mean absence.',
                       'Rank-sweep k>5 here measures availability only.',
                       'Cross-encoder seconds are scoring elapsed for new rows with cache reuse, not end-to-end service latency.']}
    save(ROOT/'RETRIEVAL_RESULTS.json',partial)
    print('Retrieval stage',json.dumps(partial,ensure_ascii=False),flush=True)
    selected,out=rerank(qs,docs,pools,known,original,frozen,settings)
    save(ROOT/'MATCHED_BUDGET_RESULTS.json',{**partial,**out})
    save(ROOT/'SELECTED_TOP1.json',selected)
    save(ROOT/'KNOWN_GRADES.json',[{'query_id':q,'chunk_id':d,'grade':g} for (q,d),g in sorted(known.items())])
    print('COMPLETE',json.dumps(out),flush=True)


if __name__=='__main__':
    main()
