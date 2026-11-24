"""Independently verify every evaluated passage and ground-truth direction against GTFS."""

import argparse
import csv
from collections import defaultdict
from datetime import date
from hashlib import file_digest
from io import TextIOWrapper
import json
from pathlib import Path
import unicodedata
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


def h(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def rows(z, table):
    with TextIOWrapper(z.open(table + ".txt"), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="benchmark.json")
    args = parser.parse_args()
    source = ROOT / args.benchmark
    protocol = json.loads((ROOT / "protocol.json").read_text())
    b = json.loads(source.read_text())
    if args.benchmark == "equal_stop.json":
        parent = ROOT / "benchmark.json"
        assert h(parent) == b["original_benchmark_sha256"]
        assert h(ROOT / "followup_protocol.json") == b["followup_sha256"]
        original = {r["id"]: r for r in json.loads(parent.read_text())["pairs"]}
        for p in b["pairs"]:
            old = original[p["id"]]
            assert {k:v for k,v in p.items() if k != "pool"} == {k:v for k,v in old.items() if k != "pool"}
            assert len({len(b["docs"][d]["stops"]) for d in p["pool"]}) == 1
    assert h(ROOT / "protocol.json") == b["protocol_sha256"]
    checked = 0
    for agency, info in protocol["sources"].items():
        feed = ROOT / "sources" / info["file"]
        assert h(feed) == info["sha256"]
        test = [p for p in b["pairs"] if p["network"] == agency]
        ids = {d for p in test for d in p["pool"]}
        docs = {d: b["docs"][d] for d in ids}
        assert all(d["network"] == agency for d in docs.values())
        trips_needed = {d["source_trip"] for d in docs.values()}
        with ZipFile(feed) as z:
            start, end = (next(iter(rows(z, "feed_info")))[x] for x in ("feed_start_date","feed_end_date"))
            active = defaultdict(set)
            weekday_fields = ("monday","tuesday","wednesday","thursday","friday","saturday","sunday")
            for service_day in {p["day"] for p in test}:
                calendar = rows(z, "calendar")
                day_digits = service_day.replace("-", "")
                assert start <= day_digits <= end
                for cal in calendar:
                    if cal["start_date"] <= day_digits <= cal["end_date"]:
                        if cal[weekday_fields[date.fromisoformat(service_day).weekday()]] == "1":
                            active[service_day].add(cal["service_id"])
                for exception in rows(z, "calendar_dates"):
                    if exception["date"] != day_digits:
                        continue
                    if exception["exception_type"] == "1":
                        active[service_day].add(exception["service_id"])
                    elif exception["exception_type"] == "2":
                        active[service_day].discard(exception["service_id"])
            trip_meta = {t["trip_id"]: t for t in rows(z, "trips") if t["trip_id"] in trips_needed}
            stops = {t["stop_id"]: t for t in rows(z, "stops")}
            stop_times = defaultdict(list)
            with TextIOWrapper(z.open("stop_times.txt"), encoding="utf-8-sig", newline="") as f:
                for t in csv.DictReader(f):
                    if t["trip_id"] in trips_needed:
                        stop_times[t["trip_id"]].append(t)
        verified_paths = {}
        for d, doc in docs.items():
            t = trip_meta[doc["source_trip"]]
            assert t["route_id"] == doc["route"] and t["service_id"] in active[doc["day"]]
            all_stops = sorted(stop_times[doc["source_trip"]], key=lambda t: int(t["stop_sequence"]))
            positions = {int(t["stop_sequence"]): i for i,t in enumerate(all_stops)}
            ix = [positions[n] for n in doc["source_sequence_numbers"]]
            assert ix == list(range(ix[0],ix[0]+len(ix)))
            subset = [all_stops[i] for i in ix]
            physical = [stops[t["stop_id"]].get("parent_station") or t["stop_id"] for t in subset]
            assert physical == doc["stops"]
            assert doc["names"] == [
                " ".join(unicodedata.normalize("NFC", stops[s]["stop_name"]).split()) for s in physical]
            assert subset[0].get("pickup_type","0") in ("0","")
            assert subset[-1].get("drop_off_type","0") in ("0","")
            verified_paths[d] = t["route_id"], physical
            checked += 1
        for p in test:
            assert len(p["pool"]) == len(set(p["pool"])) == 5
            assert p["gold"][0] != p["gold"][1]
            for (a, c), target in zip(p["directed_endpoints"], p["gold"]):
                ordered = [
                    d for d in p["pool"] if verified_paths[d][0] == p["route"]
                    and a in verified_paths[d][1] and c in verified_paths[d][1]
                    and verified_paths[d][1].index(a) < verified_paths[d][1].index(c)
                ]
                assert ordered == [target]
                assert verified_paths[target][1][0] == a and verified_paths[target][1][-1] == c
    print(f"PASS: original GTFS witnesses and uniquely sufficient printed direction for {len(b['pairs'])*2} questions;"
          f" {checked} distinct shown passages. Benchmark SHA-256 {h(source)}.")


if __name__ == "__main__":
    main()
