"""Recompute lexical and cross-encoder outcomes from the frozen pools."""
from collections import Counter
import csv
import json
from hashlib import sha256
import math
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parent

def tokens(s):
    return re.findall(r'[^\W_]+', unicodedata.normalize('NFC', s).casefold())


def bm25(questions):
    counts = Counter()
    articles = {}
    for q in questions:
        for d in q['candidates']:
            bag = Counter(tokens(d['text']))
            articles[d['id']] = bag
            counts.update(bag.keys())
    n = len(articles)
    avg_dl = sum(sum(a.values()) for a in articles.values()) / n
    for q in questions:
        query = set(tokens(q['query']))
        for d in q['candidates']:
            a = articles[d['id']]
            norm = 1.5 * (0.25 + 0.75 * sum(a.values()) / avg_dl)
            score = sum(math.log1p((n-counts[w]+0.5)/(counts[w]+0.5))
                        * 2.5 * a[w] / (a[w]+norm) for w in sorted(query & a.keys()))
            yield q['id'], d['id'], score


def rows_for(questions, triples):
    scores = {(q, p): x for q, p, x in triples}
    assert len(scores) == 5 * len(questions)
    outputs = []
    for q in questions:
        sid, gold, near = q['id'], q['sufficient'], q['near']
        a, b = scores[sid, gold], scores[sid, near]
        ranking = sorted(q['candidates'], key=lambda p: (-scores[sid,p['id']], p['id']))
        competing = [p for p in ranking if p['id'] != gold]
        top_negative = competing[0]
        top_negative_score = scores[sid, top_negative['id']]
        eligible = a > b
        strict_first = a > top_negative_score
        outputs.append({'case': sid, 'language': q['evidence_language'],
                        'k2': int(eligible), 'k5': int(strict_first),
                        'retains_pair_pass': int(eligible and strict_first),
                        'loses_pair_pass': int(eligible and not strict_first),
                        'top_negative': top_negative['id'], 'top_negative_reason': top_negative['decision_reason'],
                        'gold_minus_near': a-b, 'gold_minus_top_negative': a-top_negative_score,
                        'ties_k2': int(a == b), 'ties_k5': int(a == top_negative_score)})
    return outputs


def main():
    benchmark = ROOT / 'benchmark.json'
    digest = sha256(benchmark.read_bytes()).hexdigest()
    qs = json.loads(benchmark.read_text())['questions']
    models = ['BGE', 'M0_42', 'M0_123', 'M0_999', 'M1_42', 'M1_123', 'M1_999']
    outcomes = {'BM25': rows_for(qs, bm25(qs))}
    for n in models:
        data = json.loads((ROOT / f'scores_{n}.json').read_text())
        assert data['benchmark_sha256'] == digest
        outcomes[n] = rows_for(qs, data['scores'])
    fields = ('k2', 'k5', 'loses_pair_pass', 'ties_k2', 'ties_k5')
    def summary(choices):
        return {n: {'n': len(rs), **{f:sum(x[f] for x in rs) for f in fields}}
                for n, orig in outcomes.items() if (rs := [r for r in orig if choices(r)])}
    main_rows = summary(lambda x: True)
    langs = {l:summary(lambda x: x['language']==l) for l in sorted({q['evidence_language'] for q in qs})}
    pairs = {str(seed): {metric: sum(int(b[metric]) - int(a[metric]) for a,b in
                                zip(outcomes[f'M0_{seed}'], outcomes[f'M1_{seed}'], strict=True))
                         for metric in ['k2','k5']} for seed in [42,123,999]}
    result = dict(benchmark_sha256=digest, n=len(qs), main=main_rows, by_language=langs,
                  without_cat_ba=summary(lambda x:x['case']!='p3_11'), paired_count_delta_m1_minus_m0=pairs)
    with (ROOT / 'per_query_scores.csv').open('w', newline='') as f:
        first = next(iter(outcomes.values()))[0]
        writer = csv.DictWriter(f, fieldnames=['model',*first])
        writer.writeheader()
        for n, rows in outcomes.items():
            for row in rows:
                writer.writerow({'model':n,**row})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
