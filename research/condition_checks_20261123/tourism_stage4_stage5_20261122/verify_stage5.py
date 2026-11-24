"""Independently confirm result inputs, labels and ordering against saved feed rows."""
from collections import defaultdict
import csv
from datetime import date
from hashlib import file_digest
import io
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


def checksum(path):
    with path.open('rb') as handle:
        return file_digest(handle,'sha256').hexdigest()


def itertable(feed,name):
    with feed.open(name+'.txt') as contents:
        yield from csv.DictReader(io.TextIOWrapper(contents,encoding='utf-8-sig',newline=''))


def source_windows(corpus, protocol, audits):
    proof_total = 0
    for agency, audit in audits.items():
        archive = ROOT/'sources'/f'{agency}.zip'
        assert checksum(archive) == audit['archive_sha256']
        docs = [d for d in corpus if d['agency']==agency]
        trips_required = {d['trip'] for d in docs}
        date_key = protocol['service_date'].replace('-','')
        weekday = date.fromisoformat(protocol['service_date']).strftime('%A').lower()
        physical = defaultdict(dict)
        with ZipFile(archive) as feed:
            eligible={r['service_id'] for r in itertable(feed,'calendar')
                      if r['start_date'] <= date_key <= r['end_date'] and r[weekday]=='1'}
            exceptions = [(r['service_id'],r['exception_type']) for r in itertable(feed,'calendar_dates')
                          if r['date']==date_key]
            for sid,t in exceptions:
                if t=='1':
                    eligible.add(sid)
                else:
                    eligible.discard(sid)
            trips={r['trip_id']:(r['route_id'],r['service_id']) for r in itertable(feed,'trips')
                   if r['trip_id'] in trips_required}
            stop_map={r['stop_id']:r.get('parent_station') or r['stop_id'] for r in itertable(feed,'stops')}
            for r in itertable(feed,'stop_times'):
                if r['trip_id'] in trips:
                    physical[r['trip_id']][int(r['stop_sequence'])] = r
        for d in docs:
            route, service = trips[d['trip']]
            assert route==d['route'] and service in eligible and d['date']==protocol['service_date']
            chunks=physical[d['trip']]
            indexes=sorted(chunks)
            start=indexes.index(d['first_stop_sequence'])
            sequences=d['source_sequences']
            assert sequences==indexes[start:start+len(d['stops'])]
            ordered=[chunks[i] for i in sequences]
            assert d['source_stop_ids']==[r['stop_id'] for r in ordered]
            assert d['stops']==[stop_map[r['stop_id']] for r in ordered]
            assert d['pickup']==[r.get('pickup_type','') in ('0','') for r in ordered[:-1]]+[False]
            assert d['dropoff']==[False]+[r.get('drop_off_type','') in ('0','') for r in ordered[1:]]
            proof_total+=1
    return proof_total


def independent_label(question,document):
    if (question['agency'],question['route'])!=(document['agency'],document['route']):
        return False
    boarded=False
    for station,pickup,dropoff in zip(document['stops'],document['pickup'],document['dropoff']):
        if boarded and station==question['to'] and dropoff:
            return True
        if station==question['from'] and pickup:
            boarded=True
    return False


def main():
    for line in (ROOT/'stage5_freeze.sha256').read_text().splitlines():
        h,n=line.split()
        assert h==checksum(ROOT/n),n
    eval_path=ROOT/'stage5_eval.json'
    audit=json.loads(eval_path.read_text())
    protocol=json.loads((ROOT/'stage5_protocol.json').read_text())
    report=json.loads((ROOT/'stage5_results.json').read_text())
    assert checksum(eval_path)==report['benchmark_sha256']
    documents=audit['candidates']
    questions=audit['questions']
    provenance=source_windows(documents.values(),protocol,{x['agency']:x for x in audit['source_audit']})
    pair_ids=set()
    for q in questions:
        labels={}
        for universe,pool in q['pools'].items():
            ids=pool['candidates']
            assert len(set(ids))==5
            exact=[p for p in ids if independent_label(q,documents[p])]
            assert exact==pool['sufficient']
            labels[universe]=len(exact)
            pair_ids.update((q['id'],p) for p in ids)
        assert labels['stress']==1
    for p in ROOT.glob('stage5_scores_*.json'):
        s=json.loads(p.read_text())
        assert checksum(eval_path)==s['benchmark_sha256']
        assert {(q,d) for q,d,v in s['scores']}==pair_ids
    outcomes = json.loads((ROOT/'stage5_outcomes.json').read_text())
    index = defaultdict(list)
    for r in outcomes:
        index[r['pool'],r['method'],r['gate']].append(r)
    for record in report['summary']:
        rows = index[record['pool'],record['method'],record['gate']]
        assert record['N']==len(questions)==len(rows)
        for metric in ('success','available','gold_excluded','all_gold_excluded'):
            assert record[metric]==sum(r[metric] for r in rows)
    verification={'windows_retraced_to_archived_gtfs':provenance,
                  'queries':len(questions),'five_candidate_pools':2*len(questions),
                  'pairs_per_checkpoint':len(pair_ids),'method_gate_summaries':len(report['summary'])}
    (ROOT/'stage5_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(verification)


if __name__ == '__main__':
    main()
