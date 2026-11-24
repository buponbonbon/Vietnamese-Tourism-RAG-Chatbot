"""Trace every scored window to its dated source trip and recompute sufficiency."""
from collections import defaultdict
from datetime import date
from hashlib import file_digest
from pathlib import Path
from zipfile import ZipFile
import csv
import io
import json

ROOT=Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as f:
        return file_digest(f,'sha256').hexdigest()


def table(z,name):
    with z.open(f'{name}.txt') as source:
        yield from csv.DictReader(io.TextIOWrapper(source,encoding='utf-8-sig'))


def active_ids(z,day):
    value=day.replace('-','')
    weekday=date.fromisoformat(day).strftime('%A').lower()
    active={r['service_id'] for r in table(z,'calendar')
            if r[weekday]=='1' and r['start_date']<=value<=r['end_date']}
    for row in table(z,'calendar_dates'):
        if row['date']!=value:
            continue
        if row['exception_type']=='1':
            active.add(row['service_id'])
        else:
            assert row['exception_type']=='2'
            active.discard(row['service_id'])
    return active


def raw_seconds(stamp):
    hh,mm,ss=stamp.split(':')
    return ((int(hh)*60+int(mm))*60)+int(ss)


def prove_source(provenance,docs):
    all_days=sorted({x['date'] for x in docs})
    p=ROOT/'sources'/f"{provenance['agency']}.zip"
    assert digest(p)==provenance['feed_sha256']
    requested={x['trip'] for x in docs}
    with ZipFile(p) as z:
        stops={r['stop_id']:r.get('parent_station') or r['stop_id'] for r in table(z,'stops')}
        available={d:active_ids(z,d) for d in all_days}
        trips={r['trip_id']:r for r in table(z,'trips') if r['trip_id'] in requested}
        cache=defaultdict(dict)
        for row in table(z,'stop_times'):
            if row['trip_id'] in requested:
                cache[row['trip_id']][int(row['stop_sequence'])]=row
    for d in docs:
        trip=trips[d['trip']]
        assert trip['service_id'] in available[d['date']] and trip['route_id']==d['route']
        stops_there=cache[d['trip']]
        for sequence,sid,arr,dep,pickup,dropoff in d['sequence']:
            raw=stops_there[sequence]
            assert stops[raw['stop_id']]==sid
            assert raw_seconds(raw['arrival_time'])==arr and raw_seconds(raw['departure_time'])==dep
            assert (raw.get('pickup_type','') in ('0',''))==pickup
            assert (raw.get('drop_off_type','') in ('0',''))==dropoff


def enough(q,d):
    if any(q[field]!=d[field] for field in ('agency','route','date')):
        return False
    boarded=None
    for _,stop,arr,dep,pickup,dropoff in d['sequence']:
        if stop==q['from'] and pickup: boarded=dep
        if stop==q['to'] and dropoff and boarded is not None and arr>boarded: return True
    return False


def main():
    for line in (ROOT/'timed_freeze.sha256').read_text().splitlines():
        h,f=line.split()
        assert h==digest(ROOT/f)
    data=json.loads((ROOT/'timed_eval.json').read_text())
    qs=data['questions']; docs=data['candidates']
    for source in data['source_audit']:
        prove_source(source,[d for d in docs.values() if d['agency']==source['agency']])
    expected=set()
    for q in qs:
        for p in q['pools'].values():
            assert len(set(p['candidates']))==5
            assert [d for d in p['candidates'] if enough(q,docs[d])]==p['sufficient']
            expected.update((q['id'],x) for x in p['candidates'])
    files=sorted(ROOT.glob('timed_scores_*.json'))
    assert len(files)==7
    for path in files:
        j=json.loads(path.read_text())
        assert j['benchmark_sha256']==digest(ROOT/'timed_eval.json')
        assert len(j['scores'])==len(expected)==len({(q,p) for q,p,s in j['scores']}&expected)
    result={'reference_windows_checked':len(docs),'questions':len(qs),'schedule_dates':sorted({q['date'] for q in qs}),
            'route_groups':len({(q['agency'],q['route']) for q in qs}),
            'frozen_logits':len(expected)*len(files),'native_available':sum(bool(q['pools']['native']['sufficient']) for q in qs)}
    (ROOT/'timed_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':main()
