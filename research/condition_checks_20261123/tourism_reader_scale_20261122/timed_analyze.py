"""Derive itinerary compatibility from visible text and apply loss-aware ranking."""
from collections import Counter,defaultdict
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
import csv
import json
import math
import re
import unicodedata

ROOT=Path(__file__).resolve().parent
PROMPT=re.compile(r'^Lịch ngày (\d{4}-\d\d-\d\d) của (.+?), tuyến (.+?), cho phép đi từ (.+?) đến (.+?) theo đoạn nào\?$')
EVENT=re.compile(r'^(.+?) \((\d{2,}:\d\d:\d\d), (\d{2,}:\d\d:\d\d)\)$')
MODES=('raw','route_stop','route_stop_date','mention_order','clock_order','missing_abstain','missing_closed')


def norm(s):
    return ' '.join(unicodedata.normalize('NFC',s).casefold().split())


def seconds(t):
    h,m,s=map(int,t.split(':'))
    return 3600*h+60*m+s if m<60 and s<60 else None


@dataclass(frozen=True)
class Reading:
    agency:str|None
    route:str|None
    day:str|None
    mentions:tuple[str,...]
    times:dict | None
    explicit_order:bool

    def time_direction(self,start,end):
        if self.explicit_order:
            return self.mentions.index(start)<self.mentions.index(end)
        if not self.times or start not in self.times or end not in self.times:
            return None
        arrival, departure=self.times[end][0],self.times[start][1]
        if arrival==departure:
            return None
        return departure<arrival


def parse_document(text):
    head,_,body=text.partition('\n')
    header=head.split(' / ')
    agency,route,day=(norm(x) if x and x!='[inconnu]' else None for x in header) if len(header)==3 else (None,None,None)
    if day:
        try:
            date.fromisoformat(day)
        except ValueError:
            day=None
    if body.startswith('Thứ tự chạy: '):
        mentions=tuple(norm(s) for s in body.removeprefix('Thứ tự chạy: ').split(' → '))
        return Reading(agency,route,day,mentions,None,True)
    if body.startswith('Điểm dừng (giờ đến, giờ đi): '):
        records=[EVENT.fullmatch(x) for x in body.removeprefix('Điểm dừng (giờ đến, giờ đi): ').split(' | ')]
        if all(records):
            times={}
            for m in records:
                name,arr,dep=m.groups()
                a,b=seconds(arr),seconds(dep)
                if a is None or b is None or a>b:
                    return Reading(agency,route,day,(),None,False)
                name=norm(name)
                if name in times:
                    return Reading(agency,route,day,(),None,False)
                times[name]=(a,b)
            return Reading(agency,route,day,tuple(times),times,False)
    if body.startswith('Điểm dừng không ghi thứ tự: '):
        return Reading(agency,route,day,tuple(norm(x) for x in body.removeprefix(
            'Điểm dừng không ghi thứ tự: ').split(' | ')),None,False)
    return Reading(agency,route,day,(),None,False)


def parse_question(text):
    match=PROMPT.fullmatch(text)
    return tuple(norm(t) for t in match.groups()) if match else None


def masked_view(token,doc):
    header,_,body=doc.partition('\n')
    value=int(sha256(('simulated-view-omission:'+token).encode()).hexdigest()[:8],16)%5
    if value==0:
        return header.rsplit(' / ',1)[0]+' / [inconnu]\n'+body
    if value==1:
        parsed=parse_document(doc)
        return header+'\nĐiểm dừng không ghi thứ tự: '+' | '.join(parsed.mentions)
    return doc


def flags(q,p,mode):
    if mode=='raw':
        return False,False
    if not q:
        return mode=='missing_closed',True
    day,agency,route,start,end=q
    checks=[p.agency==agency if p.agency else None,p.route==route if p.route else None]
    if mode!='route_stop':
        checks.append(p.day==day if p.day else None)
    contains={start,end}<=set(p.mentions) if p.mentions else None
    checks.append(contains)
    if contains and mode in ('mention_order','clock_order','missing_abstain','missing_closed'):
        direction=(p.mentions.index(start)<p.mentions.index(end)) if mode=='mention_order' else p.time_direction(start,end)
        checks.append(direction)
    return ((False in checks) or (mode=='missing_closed' and None in checks)), None in checks


def outcome(q,scene,docs,mode,method,scores):
    pool=q['pools'][scene]
    ids=pool['candidates']
    gold=set(pool['sufficient'])
    parsed_q=parse_question(q['text'])
    ratings={}
    for pid in ids:
        exposed=masked_view(pid,docs[pid]['text']) if mode in ('missing_abstain','missing_closed') else docs[pid]['text']
        ratings[pid]=flags(parsed_q,parse_document(exposed),mode)
    logits=dict(zip(ids,pool['bm25'],strict=True)) if method=='BM25' else {pid:scores[q['id'],pid] for pid in ids}
    rank=sorted(ids,key=lambda pid:(ratings[pid][0],-logits[pid],pid))
    top,second=rank[:2]
    tie=method!='BM25' and logits[top]==logits[second] and ratings[top][0]==ratings[second][0]
    return {
        'query':q['id'],'network':q['agency'],'route':q['route'],'day':q['date'],'family':q['family'],
        'selected':top,'success':top in gold and not tie,'available':bool(gold),
        'gold_demoted':sum(ratings[p][0] for p in gold),
        'all_gold_demoted':bool(gold) and all(ratings[p][0] for p in gold),
        'uncertain_fields':sum(x[1] for x in ratings.values()),
        'demotions':sum(x[0] for x in ratings.values()),
    }


def profile(rows,base):
    metrics=('success','available','gold_demoted','all_gold_demoted','uncertain_fields','demotions')
    return {'n':len(rows),**{m:sum(r[m] for r in rows) for m in metrics},
            'repair':sum(r['success'] and not base[r['query']]['success'] for r in rows),
            'regression':sum(not r['success'] and base[r['query']]['success'] for r in rows)}


def main():
    eval_file=ROOT/'timed_eval.json'
    benchmark=json.loads(eval_file.read_text())
    doc=benchmark['candidates']
    qs=benchmark['questions']
    pairs={(q['id'],p) for q in qs for pool in q['pools'].values() for p in pool['candidates']}
    models={'BM25':{}}
    for f in sorted(ROOT.glob('timed_scores_*.json')):
        score=json.loads(f.read_text())
        assert score['benchmark_sha256']==sha256(eval_file.read_bytes()).hexdigest(),f.name
        values={(q,p):s for q,p,s in score['scores'] if math.isfinite(s)}
        assert set(values)==pairs and len(values)==len(score['scores'])
        models[f.stem.removeprefix('timed_scores_')]=values
    results=[]; individual=[]
    for method,scores in models.items():
        for scene in ('native','controlled'):
            baseline={r['query']:r for r in (outcome(q,scene,doc,'raw',method,scores) for q in qs)}
            for m in MODES:
                rows=[outcome(q,scene,doc,m,method,scores) for q in qs]
                rr={k:profile([r for r in rows if r['network']==k],baseline) for k in sorted({r['network'] for r in rows})}
                results.append({'pool':scene,'method':method,'gate':m,**profile(rows,baseline),'networks':rr})
                individual.extend({'pool':scene,'method':method,'gate':m,**r} for r in rows)
    report={'queries':len(qs),'pairs_per_model':len(pairs),'formats':dict(Counter(d['format'] for d in doc.values())),
            'window_count':benchmark['document_count'],'results':results}
    (ROOT/'timed_results.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'timed_winners.json').write_text(json.dumps(individual,indent=2)+'\n')
    with (ROOT/'timed_results.csv').open('w') as output:
        names=['pool','method','gate','n','available','success','repair','regression','gold_demoted','all_gold_demoted','uncertain_fields','demotions']
        writer=csv.DictWriter(output,fieldnames=names,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    for result in results:
        print(*(result[x] for x in ('pool','method','gate','success','available','repair','regression','all_gold_demoted')))
    assert all(r['gold_demoted']==0 for r in results if r['gate'] in ('clock_order','missing_abstain'))

if __name__=='__main__':main()
