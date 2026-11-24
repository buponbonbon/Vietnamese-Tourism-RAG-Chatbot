"""Setup a held-out two-date test from real GTFS event records before inference."""
from collections import Counter, defaultdict
from datetime import date
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'tourism_stage4_stage5_20261122'))
from build_stage5 import active_ids, BM25, digest, key, rows

AGENCIES={'la':'LA Metro','stm':'STM Montréal','septarail':'SEPTA Philadelphia'}
DAYS=('2026-11-23','2026-11-29')


def minute(value):
    h,m,s=map(int,value.split(':'))
    return h*3600+m*60+s


def snapshot(agency,label,manifest):
    path=ROOT/'sources'/f'{agency}.zip'
    assert digest(path)==manifest['sha256']
    doc={}
    with ZipFile(path) as z:
        assert set(('calendar.txt','calendar_dates.txt')) <= set(z.namelist())
        active={day:active_ids(z,date.fromisoformat(day)) for day in DAYS}
        routes={r['route_id']:r['route_long_name'] or r['route_short_name'] or r['route_id']
                for r in rows(z,'routes') if r['route_type'] in ('0','1','2')}
        stops={r['stop_id']:r for r in rows(z,'stops')}
        trips={r['trip_id']:r for r in rows(z,'trips')
               if r['route_id'] in routes and any(r['service_id'] in a for a in active.values())}
        timetable=defaultdict(list)
        for r in rows(z,'stop_times'):
            if r['trip_id'] in trips and r['arrival_time'] and r['departure_time']:
                station=stops[r['stop_id']]
                sid=station.get('parent_station') or r['stop_id']
                timetable[r['trip_id']].append(
                    (int(r['stop_sequence']),sid,minute(r['arrival_time']),minute(r['departure_time']),
                     r.get('pickup_type','') in ('','0'),r.get('drop_off_type','') in ('','0')))
    physical=defaultdict(set)
    for tid,time in timetable.items():
        for _,sid,*_ in time:
            physical[trips[tid]['route_id'],stops[sid]['stop_name'].casefold()].add(sid)

    for tid, timetable0 in sorted(timetable.items()):
        route=trips[tid]['route_id']
        times=sorted(timetable0)
        for length in (3,4,5):
            for i in range(len(times)-length+1):
                seq=times[i:i+length]
                sids=tuple(s[1] for s in seq)
                if len(set(sids))!=length or any(len(physical[route,stops[s]['stop_name'].casefold()])!=1 for s in sids):
                    continue
                if not all(s[4] for s in seq[:-1]) or not all(s[5] for s in seq[1:]):
                    continue
                if any(s[2]>s[3] for s in seq) or any(a[3]>=b[2] for a,b in zip(seq,seq[1:])):
                    continue
                for day in DAYS:
                    if trips[tid]['service_id'] not in active[day]:
                        continue
                    shape=day,route,sids
                    if shape in doc:
                        continue
                    pid=key(agency,shape)[:24]
                    names=[stops[s]['stop_name'].strip() for s in sids]
                    if any(any(c in name for c in ('|','→',';')) for name in names):
                        continue
                    order=int(key('render',pid)[:4],16)%3
                    render=seq if order!=2 else seq[::-1]
                    suffix=' | '.join(f'{stops[r[1]]["stop_name"].strip()} ({clock(r[2])}, {clock(r[3])})'
                                     for r in render)
                    if order==0:
                        suffix=' → '.join(f'{stops[r[1]]["stop_name"].strip()}' for r in seq)
                        suffix='Thứ tự chạy: '+suffix
                    else:
                        suffix='Điểm dừng (giờ đến, giờ đi): '+suffix
                    heading=f'{label} / {routes[route]} / {day}'
                    doc[shape]={'agency':agency,'label':label,'date':day,'route':route,'route_name':routes[route],
                                'id':pid,'stops':sids,'names':names,'trip':tid,'sequence':seq,'format':order,
                                'text':heading+'\n'+suffix}
    assert doc and all(any(d['date']==day for d in doc.values()) for day in DAYS), agency
    return doc.values()


def clock(sec):
    h, ms = divmod(sec,3600)
    m,s=divmod(ms,60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def eligible(doc,route,agency,a,b,day):
    return (doc['route']==route and doc['agency']==agency and doc['date']==day
            and a in doc['stops'] and b in doc['stops'] and doc['stops'].index(a)<doc['stops'].index(b))


def questions(docs):
    bm25=BM25(docs)
    contexts=defaultdict(list)
    for doc in docs.values():
        contexts[doc['agency'],doc['route'],doc['date']].append(doc)
    evaluation=[]
    for context,local in sorted(contexts.items()):
        agency,route,day=context
        other_day=next(d for d in DAYS if d!=day)
        remote=contexts.get((agency,route,other_day),[])
        choices=defaultdict(list)
        for doc in local:
            for i,a in enumerate(doc['stops']):
                for b in doc['stops'][i+1:]:
                    choices[a,b].append(doc)
        other_choices={(a,b) for d in remote for i,a in enumerate(d['stops']) for b in d['stops'][i+1:]}
        options=[(key('family',agency,route,a,b),a,b) for a,b in choices
                 if a<b and (b,a) in choices and (a,b) in other_choices and (b,a) in other_choices]
        for family,a,b in sorted(options):
            rows_out=[]
            for first,last in ((a,b),(b,a)):
                witness=choices[first,last][0]
                names={s:n for s,n in zip(witness['stops'],witness['names'],strict=True)}
                question=(f'Lịch ngày {day} của {witness["label"]}, tuyến {witness["route_name"]}, '
                          f'cho phép đi từ {names[first]} đến {names[last]} theo đoạn nào?')
                prior=bm25.scores(question)
                rank=sorted(prior,key=lambda p:(-prior[p],p))
                current=lambda p:eligible(docs[p],route,agency,first,last,day)
                reversed_=lambda p:eligible(docs[p],route,agency,last,first,day)
                outdated=lambda p:eligible(docs[p],route,agency,first,last,other_day)
                missing=lambda p:(docs[p]['agency'],docs[p]['route'],docs[p]['date'])==context and not {first,last}<=set(docs[p]['stops'])
                other=lambda p:(docs[p]['agency'],docs[p]['route'])!=(agency,route)
                controls=[next((p for p in rank if predicate(p)),None) for predicate in
                          (current,reversed_,outdated,missing,other)]
                if not all(controls):
                    break
                assert len(set(controls))==5
                pools={}
                for kind,ids in [('native',rank[:5]),('controlled',controls)]:
                    ordered=sorted(ids,key=lambda p:(-prior[p],p))
                    pools[kind]={'candidates':ordered,'sufficient':[p for p in ordered if current(p)],
                                 'bm25':[prior[p] for p in ordered]}
                qid=key('timed-question',context,first,last)[:24]
                rows_out.append({'id':qid,'family':family[:24],'agency':agency,'route':route,
                                 'from':first,'to':last,'date':day,'text':question,'pools':pools,
                                 'control_roles':dict(zip(('correct','reverse','other_date','missing_stop','other_route'),controls))})
            if len(rows_out)==2:
                evaluation.extend(rows_out)
                break
    return evaluation


def main():
    meta={d['name']:d for d in json.loads((ROOT/'sources/new_feeds.json').read_text())}
    docs={}
    reports=[]
    for agency,label in AGENCIES.items():
        added=list(snapshot(agency,label,meta[agency]))
        assert not (set(docs)&{d['id'] for d in added})
        docs.update({d['id']:d for d in added})
        reports.append({'agency':agency,'windows':len(added),'feed_sha256':meta[agency]['sha256']})
        print(agency,len(added),flush=True)
    scored=questions(docs)
    print('questions',len(scored),Counter(q['agency'] for q in scored),flush=True)
    assert {q['agency'] for q in scored}==set(AGENCIES)
    ids={p for q in scored for pool in q['pools'].values() for p in pool['candidates']}
    result={'source_audit':reports,'document_count':len(docs),'questions':scored,
            'candidates':{p:docs[p] for p in sorted(ids)}}
    with (ROOT/'timed_eval.json').open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    with (ROOT/'timed_freeze.sha256').open('x') as f:
        for name in ('timed_transfer_protocol.md','timed_build.py','timed_eval.json'):
            print(digest(ROOT/name),name,file=f)

if __name__=='__main__':main()
