"""Freeze native, substitution, and nested comparison pools with audited labels."""
from collections import Counter
import csv
import json
from corpus_stage4 import ROOT,load,save,digest

corpus={p['id']:p for p in load('corpus.json')}
with (ROOT/'adjudicated_native.tsv').open() as f:
    grades={(r['id'],r['pid']): None if r['grade']=='unknown' else int(r['grade'])
            for r in csv.DictReader(f,delimiter='\t')}
questions=load('native_pools.json')
for row in questions:
    native=row['native']
    local={p:grades[row['id'],p] for p in native}
    complete=all(v is not None for v in local.values())
    known_positive=any(v==1 for v in local.values())
    row['availability']='present' if known_positive else ('absent' if complete else 'uncertain')
    row['native_grades']=local
    assert len(native)==5 and len(set(native))==5
    if row['availability']=='absent':
        assert row['anchor'] not in native
        active=native[:4]+[row['anchor']]
        active_labels={p:local[p] for p in native[:4]} | {row['anchor']:1}
        kind='oracle'
        ref=row['anchor']
    else:
        active=native
        active_labels=local
        kind='native'
        ref=next((p for p in active if active_labels[p]==1),None)
    neg=next((p for p in active if active_labels[p]==0),None)
    # This comparison is absent when the available evidence remains uncertain.
    row['nested']=[ref,neg] if ref and neg else None
    row['active']=active
    row['active_grades']=active_labels
    row['condition']=kind
    row['candidate_ids']=list(dict.fromkeys(native+active))
    assert len(active)==5 and len(set(active))==5
    assert all(pid in corpus for pid in row['candidate_ids'])
save('benchmark.json',{'protocol_sha256':digest(ROOT/'protocol.md'),
                       'corpus_sha256':digest(ROOT/'corpus.json'),
                       'annotation_sha256':digest(ROOT/'adjudicated_native.tsv'),
                       'questions':questions})
print(Counter((r['cohort'],r['availability']) for r in questions))
print('Scored pairs per model',sum(len(q['candidate_ids']) for q in questions))
