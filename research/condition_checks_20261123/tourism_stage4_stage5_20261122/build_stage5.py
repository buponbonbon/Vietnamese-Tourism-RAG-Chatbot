"""Build transfer test from held-out archived public GTFS snapshots."""
from collections import Counter, defaultdict
import csv
from datetime import date
from hashlib import sha256, file_digest
import io
import json
import math
from pathlib import Path
import re
import unicodedata
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
TOKENS = re.compile(r'[^\W_]+', re.UNICODE)


def key(*parts):
    return sha256(json.dumps(parts, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def digest(path):
    with path.open('rb') as source:
        return file_digest(source, 'sha256').hexdigest()


def dump(name, value):
    with (ROOT / name).open('x', encoding='utf-8') as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')


def rows(feed, table):
    with feed.open(table + '.txt') as raw:
        yield from csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig', newline=''))


def active_ids(feed, day):
    day_id = day.strftime('%Y%m%d')
    weekday = ('monday','tuesday','wednesday','thursday','friday','saturday','sunday')[day.weekday()]
    active = {r['service_id'] for r in rows(feed, 'calendar')
              if r['start_date'] <= day_id <= r['end_date'] and r[weekday] == '1'}
    for row in rows(feed, 'calendar_dates'):
        if row['date'] == day_id:
            if row['exception_type'] == '1':
                active.add(row['service_id'])
            else:
                assert row['exception_type'] == '2'
                active.discard(row['service_id'])
    return active


def build_documents(network, label, meta, protocol):
    source = ROOT / 'sources' / (network + '.zip')
    assert meta['sha256'] == digest(source)
    day = date.fromisoformat(protocol['service_date'])
    with ZipFile(source) as feed:
        active = active_ids(feed, day)
        routes = {r['route_id']:r for r in rows(feed, 'routes')
                  if r['route_type'] in protocol['route_types']}
        trips = {t['trip_id']: t['route_id'] for t in rows(feed, 'trips')
                 if t['service_id'] in active and t['route_id'] in routes}
        stops = {s['stop_id']:s for s in rows(feed, 'stops')}
        times = defaultdict(list)
        for row in rows(feed, 'stop_times'):
            trip = row['trip_id']
            if trip not in trips:
                continue
            sid = row['stop_id']
            physical = stops[sid].get('parent_station') or sid
            times[trip].append((int(row['stop_sequence']), physical, sid,
                                row.get('pickup_type','') in ('', '0'),
                                row.get('drop_off_type','') in ('', '0')))
    names = {k:' '.join(unicodedata.normalize('NFC', v['stop_name']).split())
             for k,v in stops.items()}
    name_groups = defaultdict(set)
    for trip, seq in times.items():
        for _, stop, *_ in seq:
            name_groups[(trips[trip], names[stop].casefold())].add(stop)

    docs = {}
    for trip, times0 in sorted(times.items()):
        seq = sorted(times0)
        route = trips[trip]
        for n in protocol['lengths']:
            for i in range(len(seq) - n + 1):
                window = seq[i:i+n]
                stop_ids = tuple(r[1] for r in window)
                if len(set(stop_ids)) < n or any(
                    len(name_groups[route, names[s].casefold()]) != 1 for s in stop_ids
                ):
                    continue
                pickup = tuple(r[3] for r in window)
                dropoff = tuple(r[4] for r in window)
                if not pickup[0] or not dropoff[-1]:
                    continue
                # Terminal pickup and initial drop-off cannot change evidence inside this window.
                pickup = (*pickup[:-1], False)
                dropoff = (False, *dropoff[1:])
                shape = route, stop_ids, pickup, dropoff
                if shape in docs:
                    continue
                route_name = (routes[route]['route_short_name'].strip()
                              or routes[route]['route_long_name'].strip() or route)
                stop_names = [names[s] for s in stop_ids]
                text = (f'{label}, tuyến {route_name}, ngày {day}. '
                        f'Thứ tự điểm dừng: {" → ".join(stop_names)}.')
                if not all(pickup[:-1]):
                    text += ' Không đón khách tại ' + ', '.join(
                        s for s, allowed in zip(stop_names[:-1], pickup[:-1]) if not allowed) + '.'
                if not all(dropoff[1:]):
                    text += ' Không trả khách tại ' + ', '.join(
                        s for s, allowed in zip(stop_names[1:], dropoff[1:]) if not allowed) + '.'
                docs[shape] = {
                    'id':key('window', network, str(day), shape)[:24], 'agency':network,
                    'route':route, 'route_name':route_name, 'date':str(day), 'text':text,
                    'stops':stop_ids, 'names':stop_names, 'pickup':pickup, 'dropoff':dropoff,
                    'trip':trip, 'first_stop_sequence':window[0][0],
                    'source_stop_ids':[r[2] for r in window],
                    'source_sequences':[r[0] for r in window],
                }
    assert len(docs) > 100, (network,len(docs))
    return list(docs.values()), {'agency':network, 'archive_sha256':meta['sha256'],
                                'routes':len(routes), 'trip_instances':len(times),
                                'windows':len(docs), 'active_services':sorted(active)}


def tokens(text):
    return TOKENS.findall(text.casefold())


class BM25:
    def __init__(self, docs):
        self.postings = defaultdict(list)
        sizes = []
        for doc in docs.values():
            counts = Counter(tokens(doc['text']))
            n = sum(counts.values())
            sizes.append(n)
            for term, tf in counts.items():
                self.postings[term].append((doc['id'], tf, n))
        self.mean = sum(sizes) / len(sizes)
        self.count = len(sizes)

    def scores(self, text):
        scores = defaultdict(float)
        for term, qtf in sorted(Counter(tokens(text)).items()):
            posting = self.postings[term]
            df = len(posting)
            idf = math.log1p((self.count - df + 0.5) / (df + 0.5))
            for doc, tf, length in posting:
                denominator = tf + 1.5 * (1 - 0.75 + 0.75 * length / self.mean)
                scores[doc] += qtf * idf * tf * 2.5 / denominator
        return scores


def enough(doc, context, origin, destination):
    if (doc['agency'], doc['route']) != context or not {origin, destination}.issubset(doc['stops']):
        return False
    a, b = doc['stops'].index(origin), doc['stops'].index(destination)
    return a < b and doc['pickup'][a] and doc['dropoff'][b]


def build_evaluation(docs, protocol):
    lexical = BM25(docs)
    by_route = defaultdict(list)
    for doc in docs.values():
        by_route[doc['agency'], doc['route']].append(doc)
    evaluation = []
    for context, windows in sorted(by_route.items()):
        decisions = defaultdict(list)
        for doc in windows:
            for i, first in enumerate(doc['stops']):
                for last in doc['stops'][i+1:]:
                    if enough(doc, context, first, last):
                        decisions[first,last].append(doc)
        families = []
        for (first, last), witnesses in decisions.items():
            if first >= last or (last, first) not in decisions:
                continue
            # Distinct directed windows are required for two reverse candidates in each query.
            if min(len(witnesses),len(decisions[last, first])) < 2:
                continue
            families.append((key('family', context, first, last)[:24], first, last))
        for family, first, last in sorted(families)[:protocol['max_families_per_route']]:
            family_rows = []
            for start, end in ((first,last),(last,first)):
                sample = decisions[start,end][0]
                a,b = (sample['names'][sample['stops'].index(s)] for s in (start,end))
                query = (f'Tôi muốn đi từ {a} đến {b} trên tuyến {sample["route_name"]} của '
                         f'{protocol["agencies"][context[0]]} vào ngày {sample["date"]}. '
                         'Đoạn lịch trình nào xác nhận chiều đi này?')
                scores = lexical.scores(query)
                ranked = sorted(scores, key=lambda pid:(-scores[pid],pid))
                native = ranked[:5]
                positive = [p for p in ranked if enough(docs[p], context, start, end)]
                reverse = [p for p in ranked if enough(docs[p], context, end, start)]
                same_missing = [p for p in ranked if (docs[p]['agency'],docs[p]['route'])==context
                                and not {start,end}.issubset(docs[p]['stops'])]
                other = [p for p in ranked if (docs[p]['agency'],docs[p]['route'])!=context]
                if not positive or len(reverse) < 2 or not same_missing or not other:
                    break
                stress = [positive[0],*reverse[:2],same_missing[0],other[0]]
                assert len(set(stress)) == len(native) == 5
                pools = {}
                for name, pids in [('natural',native),('stress',stress)]:
                    ordered = sorted(pids, key=lambda p:(-scores[p],p))
                    pools[name] = {'candidates':ordered,
                                   'sufficient':[p for p in ordered if enough(docs[p],context,start,end)],
                                   'bm25':[scores[p] for p in ordered]}
                assert len(pools['stress']['sufficient']) == 1
                qid = key('query', family, start, end)[:24]
                family_rows.append({'id':qid,'family':family,'agency':context[0],
                                    'route':context[1],'from':start,'to':end,
                                    'text':query,'pools':pools})
            if len(family_rows) == 2:
                evaluation.extend(family_rows)
    return evaluation


def main():
    protocol = json.loads((ROOT/'stage5_protocol.json').read_text())
    feeds = {m['name']:m for m in json.loads((ROOT/'sources/feeds.json').read_text())}
    corpus = {}
    audits = []
    for agency, label in protocol['agencies'].items():
        documents, audit = build_documents(agency,label,feeds[agency],protocol)
        print(agency,len(documents),flush=True)
        corpus.update((d['id'],d) for d in documents)
        audits.append(audit)
    questions = build_evaluation(corpus,protocol)
    keep = {p for q in questions for pool in q['pools'].values() for p in pool['candidates']}
    for model in ('natural','stress'):
        print(model,len(questions),sum(bool(q['pools'][model]['sufficient']) for q in questions),flush=True)
    dump('stage5_eval.json',{'source_audit':audits, 'n_corpus':len(corpus),
                            'questions':questions,'candidates':{p:corpus[p] for p in sorted(keep)}})
    files = [ROOT/'stage5_protocol.json',ROOT/'build_stage5.py',ROOT/'stage5_eval.json']
    with (ROOT/'stage5_freeze.sha256').open('x') as freeze:
        for p in files:
            print(digest(p), p.name, file=freeze)


if __name__ == '__main__':
    main()
