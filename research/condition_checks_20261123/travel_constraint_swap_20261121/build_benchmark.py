"""Construct frozen, GTFS-backed directional contrasts before any model inference."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, timedelta
from hashlib import sha256, file_digest
import io
import json
import math
from pathlib import Path
import re
import unicodedata
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def digest(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def load_json(name):
    return json.loads((ROOT / name).read_text())


def save(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def key(*parts):
    return sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def tokens(text):
    return TOKEN.findall(unicodedata.normalize("NFC", text).casefold())


def features(text, n=1):
    t = tokens(text)
    if n == 1:
        return t
    return list(zip(t, t[1:]))


def table(z, name):
    with z.open(name + ".txt") as f:
        yield from csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig", newline=""))


def date_from(value):
    return date(int(value[:4]), int(value[4:6]), int(value[6:8]))


def choose_service(z):
    feed = next(table(z, "feed_info"))
    start = date_from(feed["feed_start_date"])
    day = start + timedelta(days=(2 - start.weekday()) % 7)
    assert day <= date_from(feed["feed_end_date"])
    as_string = day.strftime("%Y%m%d")
    active = {
        r["service_id"] for r in table(z, "calendar")
        if r["start_date"] <= as_string <= r["end_date"] and r["wednesday"] == "1"
    }
    for row in table(z, "calendar_dates"):
        if row["date"] == as_string:
            if row["exception_type"] == "1":
                active.add(row["service_id"])
            else:
                assert row["exception_type"] == "2"
                active.discard(row["service_id"])
    assert active
    return day.isoformat(), active, feed.get("feed_version")


def read_feed(network, spec, lengths):
    src = ROOT / "sources" / spec["file"]
    assert digest(src) == spec["sha256"]
    with ZipFile(src) as z:
        day, service, version = choose_service(z)
        stops = {r["stop_id"]: r for r in table(z, "stops")}
        routes = {r["route_id"]: r for r in table(z, "routes")}
        trips = {r["trip_id"]: r for r in table(z, "trips") if r["service_id"] in service}
        records = defaultdict(list)
        for r in table(z, "stop_times"):
            if r["trip_id"] not in trips:
                continue
            stop = stops[r["stop_id"]]
            phys = stop.get("parent_station") or r["stop_id"]
            flags = tuple(r.get(k, "0") in ("0", "") for k in ("pickup_type", "drop_off_type"))
            records[r["trip_id"]].append((int(r["stop_sequence"]), phys, flags, r["stop_id"]))

    names = {
        sid: " ".join(unicodedata.normalize("NFC", stops[sid]["stop_name"]).split())
        for sid in stops
    }
    names_on_route = defaultdict(lambda: defaultdict(set))
    for trip, stop_rows in records.items():
        for _, phys, _, _ in stop_rows:
            names_on_route[trips[trip]["route_id"]][names[phys].casefold()].add(phys)

    display_routes = {r: meta["route_short_name"].strip() or r for r, meta in routes.items()}
    alias_count = Counter(label.casefold() for label in display_routes.values())
    display_routes = {r: f"{label} ({r})" if alias_count[label.casefold()] > 1 else label
                      for r, label in display_routes.items()}
    windows = {}
    for trip, stop_rows in sorted(records.items()):
        ordered = sorted(stop_rows)
        assert len({x[0] for x in ordered}) == len(ordered)
        info = trips[trip]
        route = info["route_id"]
        for n in lengths:
            for i in range(len(ordered) - n + 1):
                fragment = ordered[i:i+n]
                if not fragment[0][2][0] or not fragment[-1][2][1]:
                    continue
                seq = tuple(r[1] for r in fragment)
                stop_names = [names[d] for d in seq]
                if len(set(seq)) != n or len(set(stop_names)) != n:
                    continue
                if any(len(names_on_route[route][nm.casefold()]) != 1 for nm in stop_names):
                    continue
                # Merge identical printed windows by unioning observed boardable pairs
                # from the date. Never infer pickup and dropoff across separate trips.
                pairs = {(seq[a], seq[b]) for a in range(n) for b in range(a+1, n)
                         if fragment[a][2][0] and fragment[b][2][1]}
                idx = route, seq
                if idx in windows:
                    existing = windows[idx]
                    for start, end in pairs:
                        existing["witness_pairs"].setdefault((start, end), trip)
                    continue
                route_label = display_routes[route]
                prefix = f"{spec['label']}, tuyến {route_label}, lịch ngày {day}. "
                plain = prefix + "Thứ tự điểm dừng trong đoạn lịch trình: " + " → ".join(stop_names) + "."
                explicit = (prefix + f"Điểm lên đầu đoạn: {stop_names[0]}. Điểm xuống cuối đoạn: {stop_names[-1]}. "
                            + "Các điểm dừng ở giữa theo thứ tự: " + " → ".join(stop_names[1:-1]) + ".")
                windows[idx] = {
                    "id": key("doc", network, route, day, seq)[:24],
                    "network": network, "agency": spec["label"], "route": route, "route_label": route_label,
                    "day": day, "stops": list(seq), "names": stop_names, "plain": plain, "explicit": explicit,
                    "source_trip": trip, "source_sequence_numbers": [r[0] for r in fragment],
                    "witness_pairs": {p: trip for p in pairs},
                }

    metadata = {"network": network, "day": day, "active_service_ids": sorted(service),
                "active_trips_with_stops": len(records), "window_count": len(windows),
                "feed_version": version, "sha256": digest(src)}
    return windows, metadata


def relevant(doc, route, start, end):
    return doc["route"] == route and (start, end) in doc["witness_pairs"]


class BM25:
    def __init__(self, docs, n=1, field="plain"):
        self.docs = docs
        self.n = n
        self.index = defaultdict(list)
        self.length = {}
        for doc in docs.values():
            vector = Counter(features(doc[field], n))
            self.length[doc["id"]] = sum(vector.values())
            for term, n_seen in vector.items():
                self.index[term].append((doc["id"], n_seen))
        self.mean_length = sum(self.length.values()) / len(self.length)

    def rank(self, query):
        score = defaultdict(float)
        k1, b, n = 1.5, .75, len(self.length)
        for term, qtf in sorted(Counter(features(query, self.n)).items()):
            docs = self.index[term]
            df = len(docs)
            idf = math.log1p((n-df+.5)/(df+.5))
            for d, tf in docs:
                ratio = self.length[d] / self.mean_length
                score[d] += qtf * idf * tf * (k1+1)/(tf+k1*(1-b+b*ratio))
        return score


def query(a, b, doc):
    return (f"Tôi muốn đi từ {a} đến {b} trên tuyến {doc['route_label']} của "
            f"{doc['agency']} vào ngày {doc['day']}. Đoạn lịch trình nào xác nhận chiều đi này?")


def build(protocol):
    all_docs = {}
    all_pairs = []
    audit = {"source_snapshots": [], "possible_endpoint_pairs": {}, "selected_pairs": {}}
    for network, spec in sorted(protocol["sources"].items()):
        windows, meta = read_feed(network, spec, protocol["data_rules"]["trip_window_lengths"])
        audit["source_snapshots"].append(meta)
        docs = {d["id"]: d for d in windows.values()}
        assert len(docs) == len(windows)
        all_docs.update(docs)
        lexical = BM25(docs)
        endpoints = {}
        for (route, seq), first in windows.items():
            reverse = seq[::-1]
            if seq >= reverse or (route, reverse) not in windows:
                continue
            group = route, *sorted((seq[0], seq[-1]))
            current = endpoints.get(group)
            if current is None or (-len(seq), seq) < (-len(current), current):
                endpoints[group] = seq
        audit["possible_endpoint_pairs"][network] = len(endpoints)
        groups = defaultdict(list)
        for (route, first, last), seq in endpoints.items():
            physical = route[:-1] if network == "subway" and route in ("6X","7X","FX") else route
            groups[physical].append((key("pair", network, route, first, last), route, seq))
        audit["selected_pairs"][network] = 0
        for physical, pairs in sorted(groups.items()):
            for stable, route, seq in sorted(pairs)[:3]:
                direct, reverse = (windows[route, s] for s in (seq, seq[::-1]))
                assert Counter(tokens(direct["plain"])) == Counter(tokens(reverse["plain"]))
                assert Counter(tokens(direct["explicit"])) == Counter(tokens(reverse["explicit"]))
                assert direct["plain"] != reverse["plain"]
                q = [query(direct["names"][0], direct["names"][-1], direct),
                     query(reverse["names"][0], reverse["names"][-1], reverse)]
                assert Counter(tokens(q[0])) == Counter(tokens(q[1]))
                scores = lexical.rank(q[0])
                assert scores == lexical.rank(q[1])
                order = sorted(docs, key=lambda d: (-scores.get(d, 0.), d))
                a, b = seq[0], seq[-1]
                distractors = [d for d in order
                               if docs[d]["route"] != route
                               or not {a, b}.issubset(docs[d]["stops"])][:3]
                ids = sorted([direct["id"],reverse["id"],*distractors],
                             key=lambda d: key("pool-order", stable, d))
                assert len(ids) == len(set(ids)) == 5
                expected = [direct["id"], reverse["id"]]
                assert expected == [
                    next(d for d in ids if relevant(docs[d], route, x, y))
                    for x,y in ((a,b),(b,a))]
                for idx,x,y in ((0,a,b),(1,b,a)):
                    assert sum(relevant(docs[d],route,x,y) for d in ids) == 1
                natural = [{"hits_at_1": int(relevant(docs[order[0]], route, x,y)),
                            "available_at_5": int(any(relevant(docs[d], route, x,y) for d in order[:5]))}
                           for x,y in ((a,b),(b,a))]
                all_pairs.append({
                    "id": stable[:24], "network":network, "route_group": f"{network}:{physical}",
                    "route":route, "day":direct["day"], "pool":ids, "queries":q,
                    "gold":expected, "directed_endpoints":[[a,b],[b,a]], "natural_bm25":natural
                })
                audit["selected_pairs"][network] += 1
    assert len({x["id"] for x in all_pairs}) == len(all_pairs)
    for d in all_docs.values():
        d["witness_pairs"] = [[a,b,tid] for (a,b),tid in sorted(d["witness_pairs"].items())]
    return all_docs, all_pairs, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="benchmark.json")
    args = p.parse_args()
    docs, pairs, audit = build(load_json("protocol.json"))
    save(ROOT / args.out, {"protocol_sha256": digest(ROOT / "protocol.json"),
                           "audit": audit, "docs": docs, "pairs": pairs})
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print("Families",len(pairs),"source route groups",len({p["route_group"] for p in pairs}))


if __name__ == "__main__":
    main()
