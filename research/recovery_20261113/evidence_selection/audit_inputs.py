"""Exposure and duplicate diagnostics; never changes the fixed cohort or grades."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,unicodedata
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
norm=lambda s:' '.join(unicodedata.normalize('NFC',s).split())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    qs=json.loads((ROOT/'cohort.json').read_text());pairs=json.loads((ROOT/'scoring_pairs.json').read_text())
    seen=defaultdict(list)
    for p in pairs:seen[(p['query_id'],norm(p['text']))].append(p)
    duplicate=[rows for rows in seen.values() if len(rows)>1]
    contradiction=[rows for rows in duplicate if len({r['grade'] for r in rows})>1]
    exposure={}
    for name in ['train_moc_0.json','train_moc_13.json','cross_encoder_VAL.json','cross_encoder_TEST.json']:
        local=ROOT/'source_inputs'/name
        path=local if local.exists() else PROJECT/'data/processed'/name;rows=json.loads(path.read_text())
        queries={norm(r['query']) for r in rows};texts=set();qp=set()
        for r in rows:
            for key in ('positive','negative'):
                s=r[key][0] if isinstance(r[key],list) else r[key]
                texts.add(norm(s));qp.add((norm(r['query']),norm(s)))
        exposure[name]=dict(sha256=digest(path),training_or_pair_rows=len(rows),exact_query_overlap=sum(norm(q['query']) in queries for q in qs),scored_rows_with_exact_passage_overlap=sum(norm(p['text']) in texts for p in pairs),exact_query_passage_overlap=sum((norm(p['query']),norm(p['text'])) in qp for p in pairs),native_rows_with_exact_passage_overlap=sum(p['role']=='native' and norm(p['text']) in texts for p in pairs),oracle_rows_with_exact_passage_overlap=sum(p['role']!='native' and norm(p['text']) in texts for p in pairs))
    report=dict(status='COMPLETE',normalization='Unicode NFC, collapsed whitespace, case preserved',cohort_queries=len(qs),native_grade_counts=dict(Counter(str(p['grade']) for p in pairs if p['role']=='native')),behavior_counts=dict(Counter(q['behavior'] for q in qs)),recorded_split_counts=dict(Counter(q['recorded_split'] for q in qs)),family_ids=len({q['family_id'] for q in qs}),duplicate_same_query_text_groups=len(duplicate),duplicate_text_grade_conflicts=len(contradiction),duplicate_rows=[[r['row_id'] for r in rows] for rows in duplicate],exposure=exposure,limits=['Development cohort pools multiple recorded splits, not a newly held-out test.','Exact passage reuse is a limit on generalization; it does not prove query-dependent relevance labels contradictory.','No exact overlap does not establish semantic disjointness or absence of pretrained exposure.'])
    (ROOT/'input_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
