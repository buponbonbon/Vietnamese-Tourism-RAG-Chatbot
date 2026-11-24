"""Validate saved outcomes independently of the ranking/summarization routine."""
import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def hash_file(name):
    return sha256((ROOT / name).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs-only', action='store_true',
                        help='Verify frozen outputs without local HTML snapshots.')
    args = parser.parse_args()
    for expected, name in (r.split() for r in (ROOT / 'freeze.sha256').read_text().splitlines()):
        assert hash_file(name) == expected, name
    if not args.outputs_only:
        rebuilt = subprocess.check_output(
            [sys.executable, str(ROOT / 'build_audit.py')], cwd=ROOT
        )
        assert rebuilt == (ROOT / 'benchmark.json').read_bytes()
        print('PASS exact benchmark rebuild from the archived pages.')
    d = read('benchmark.json')
    expected = read('results.json')
    cases = {r['id']: r for r in d['questions']}
    pairs = {(rid, c['id']) for rid, row in cases.items() for c in row['candidates']}
    per_case = {(r['model'],r['case']): r for r in
                csv.DictReader((ROOT / 'per_query_scores.csv').open())}
    assert len(per_case) == 8 * len(cases)
    for file in sorted(ROOT.glob('scores_*.json')):
        model = file.stem.removeprefix('scores_')
        data = read(file.name)
        assert data['benchmark_sha256'] == hash_file('benchmark.json')
        logits = {(q, p): s for q, p, s in data['scores']}
        assert pairs == logits.keys() and len(logits) == len(data['scores'])
        assert all(math.isfinite(x) for x in logits.values())
        two, five, losses = 0, 0, 0
        for rid, row in cases.items():
            k2 = logits[rid,row['sufficient']] > logits[rid,row['near']]
            pos = logits[rid,row['sufficient']]
            k5 = sum(s >= pos for (q,_), s in logits.items() if q == rid) == 1
            assert not k5 or k2
            rec = per_case[model,rid]
            assert k2 == bool(int(rec['k2'])) and k5 == bool(int(rec['k5']))
            two += k2
            five += k5
            losses += k2 and not k5
        actual = expected['main'][model]
        assert (actual['k2'], actual['k5'], actual['loses_pair_pass']) == (two, five, losses), model
        print(f'{model} PASS {two} of {len(cases)} at K=2 and {five} at K=5')
    print('PASS checkpoint score alignment and 7 neural outcome tables.')


if __name__ == '__main__':
    main()
