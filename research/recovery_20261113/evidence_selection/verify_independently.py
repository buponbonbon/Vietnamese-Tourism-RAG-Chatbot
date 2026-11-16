"""Independent source/label, outranker-count, analytic K and bootstrap checks.

Run with system Python (numpy and openpyxl). Does not import the scorer,
cohort builder or metric implementation. Does not change human records.
"""
from pathlib import Path
from collections import defaultdict
import csv, hashlib, json, math, unicodedata
from urllib.parse import urlsplit
import numpy as np
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
norm = lambda v: ' '.join(unicodedata.normalize('NFC', '' if v is None else str(v)).split())


def csvrows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def urlkey(value):
    p = urlsplit(norm(value))
    return p.netloc.lower(), p.path.rstrip('/'), p.query


def workbook(path, draft):
    bytext = defaultdict(list)
    for row in draft:
        bytext[norm(row['query'])].append(row)
    wb = load_workbook(path, read_only=True, data_only=False, keep_links=False)
    questions, passages, opaque = {}, {}, {}
    try:
        rows = wb['CAU_HOI'].iter_rows()
        assert [c.value for c in next(rows)][:4] == ['Mã câu','Câu hỏi','Nhãn hành vi','Giữ / sửa / loại câu']
        for cells in rows:
            if not any(c.value is not None for c in cells): continue
            assert not any(c.data_type == 'f' for c in cells)
            r = [norm(c.value) for c in cells]
            found = bytext[r[1]]
            assert len(found) == 1
            q = found[0]; qid = q['benchmark_id']
            assert qid not in questions and r[0] not in opaque
            assert r[4] == norm(q['anchor_evidence']) and urlkey(r[5]) == urlkey(q['source_url'])
            opaque[r[0]] = qid
            questions[qid] = dict(behavior=r[2], decision=r[3], revision=r[7], status=r[11])
        rows = wb['BANG_CHUNG'].iter_rows()
        assert [c.value for c in next(rows)][:4] == ['Mã câu','Câu hỏi','Mã đoạn văn','Mức bằng chứng']
        for cells in rows:
            if not any(c.value is not None for c in cells): continue
            assert not any(c.data_type == 'f' for c in cells)
            r = [norm(c.value) for c in cells]
            qid = opaque[r[0]]; key = (qid, r[2]); assert key not in passages
            assert r[1] == norm(next(q['query'] for q in draft if q['benchmark_id'] == qid))
            if r[3] in ('1','2'): assert r[5] and r[5] in r[4]
            passages[key] = dict(grade=r[3], text=r[4], url=r[6], status=r[8])
    finally:
        wb.close()
    assert len(questions) == 268 and len(passages) == 1688
    return questions, passages


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
    protocol = json.loads((ROOT/'protocol.json').read_text())
    assert digest(ROOT/'cohort.json') == protocol['cohort_sha256']
    assert digest(ROOT/'scoring_pairs.json') == protocol['scoring_pairs_sha256']
    for fn,expected in protocol['script_hashes'].items(): assert digest(ROOT/fn) == expected
    files = dict(reviewer1=ROOT/'source_inputs/reviewer_01_round2_completed.xlsx',
                 reviewer2=ROOT/'source_inputs/reviewer_02_round2_COMPLETED.xlsx',
                 benchmark=ROOT/'source_inputs/benchmark_draft.csv',
                 native_bm25_top50=ROOT/'source_inputs/native_bm25_top50.csv',
                 corpus=(ROOT/'source_inputs/vietnam_tourism_rag_dataset.csv') if (ROOT/'source_inputs/vietnam_tourism_rag_dataset.csv').exists() else PROJECT/'data/processed/vietnam_tourism_rag_dataset.csv')
    for key,p in files.items(): assert digest(p) == protocol['source_artifacts']['input_hashes'][key], key
    draft = csvrows(files['benchmark']); byid = {r['benchmark_id']:r for r in draft}
    chunks = {r['chunk_id']:r for r in csvrows(files['corpus'])}
    qa,pa = workbook(files['reviewer1'],draft); qb,pb = workbook(files['reviewer2'],draft)
    assert set(qa)==set(qb) and set(pa)==set(pb)
    for key in pa:
        for record in (pa[key],pb[key]):
            assert record['text']==norm(chunks[key[1]]['text_content'])
            assert urlkey(record['url'])==urlkey(chunks[key[1]]['url'])
    native = defaultdict(list)
    for r in csvrows(files['native_bm25_top50']):
        if int(r['rank']) <= 5: native[r['benchmark_id']].append(r)
    eligible = []
    for qid in sorted(qa):
        a,b = qa[qid],qb[qid]
        if any(r['status']!='DONE' or r['decision']!='KEEP' or r['revision'] for r in (a,b)): continue
        if a['behavior']!=b['behavior'] or a['behavior'] not in ('answer','correct'): continue
        rows = sorted(native[qid],key=lambda r:int(r['rank']))
        assert [int(r['rank']) for r in rows] == [1,2,3,4,5]
        if any((qid,r['candidate_chunk_id']) not in pa for r in rows): continue
        keys = [(qid,r['candidate_chunk_id']) for r in rows]
        if any(pa[k]['status']!='DONE' or pb[k]['status']!='DONE' or pa[k]['grade']!=pb[k]['grade'] or pa[k]['grade'] not in ('0','1','2') for k in keys): continue
        positive = [cid for (q,cid) in pa if q==qid and pa[q,cid]['status']==pb[q,cid]['status']=='DONE' and pa[q,cid]['grade']==pb[q,cid]['grade']=='2']
        if not positive: continue
        eligible.append(qid)
    cohort = json.loads((ROOT/'cohort.json').read_text())
    pairs = json.loads((ROOT/'scoring_pairs.json').read_text())
    assert eligible == [q['benchmark_id'] for q in cohort] and len(eligible)==140
    for q in cohort:
        qid=q['benchmark_id']; original=byid[qid]
        assert q['source_group_id']==original['source_group_id']
        assert norm(q['query'])==norm(original['query'])
        rows=sorted(native[qid],key=lambda r:int(r['rank']))
        ids=q['native_row_ids']; assert len(ids)==5
        for i,r in zip(ids,rows):
            p=pairs[i]; cid=r['candidate_chunk_id']
            assert p['row_id']==i and p['query_id']==qid and p['chunk_id']==cid
            assert norm(p['text'])==pa[qid,cid]['text']==norm(r['candidate_text'])
            assert p['grade']==int(pa[qid,cid]['grade'])==int(pb[qid,cid]['grade'])
        available=any(pairs[i]['grade']==2 for i in ids)
        assert available==q['has_grade2_in_native_pool']
        if available: assert q['expanded_row_ids']==ids
        else:
            assert q['expanded_row_ids'][:5]==ids and len(q['expanded_row_ids'])==6
            p=pairs[q['expanded_row_ids'][-1]]; cid=original['anchor_chunk_id']
            assert p['chunk_id']==cid and cid not in [pairs[i]['chunk_id'] for i in ids]
            assert norm(p['text'])==pa[qid,cid]['text'] and p['grade']==2
            assert pa[qid,cid]['grade']==pb[qid,cid]['grade']=='2'
            assert pa[qid,cid]['status']==pb[qid,cid]['status']=='DONE'
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
    report=dict(status='PASS',models=13,model_query_checks=1820,original_workbook_rows_checked=3912,cohort_queries=140,oracle_queries=95,K_queries=41,bootstrap_contrasts=7,max_absolute_metric_or_interval_discrepancy=max_error,protocol_sha256=digest(ROOT/'protocol.json'),results_sha256=digest(ROOT/'results.json'),verifier_sha256=digest(Path(__file__)))
    (ROOT/'independent_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__': main()
