"""Exploratory addition after the initial retrieval-only results."""

from analyze_external import contrast
from run_external import combine, digest, load, load_source, ROOT, save

_, queries = load_source()
rank = load("RANKS.json")
unicode = load("BM25_UNICODE.json")
names = ("bm25", "dense", "rrf", "bm25_unicode", "rrf_unicode")
depths = (1, 5, 10, 20, 50)
hits = {name: {str(k): [] for k in depths} for name in names}
for q in queries:
    key, gold = q["id"], q["doc"]
    filtered = rank[key]
    pools = {**filtered, "bm25_unicode": unicode[key],
             "rrf_unicode": combine(unicode[key], filtered["dense"])}
    for name in names:
        for k in depths:
            hits[name][str(k)].append(gold in pools[name][:k])
articles = [q["article"] for q in queries]
counts = {name: {k: sum(h) for k, h in arm.items()} for name, arm in hits.items()}
paired = {
    f"rrf_unicode_minus_bm25_unicode_at_{k}": contrast(
        hits["rrf_unicode"][str(k)], hits["bm25_unicode"][str(k)], articles
    ) for k in (1, 5)
}
save("LEXICAL_SENSITIVITY.json", {
    "stage": "Exploratory retrieval-only follow-up after seeing the pre-specified retrieval-only results.",
    "method": "Substitute symmetric Unicode BM25 as the lexical RRF arm. Preserve dense ranks, RRF constant and depth.",
    "n": len(queries),
    "hits": counts,
    "paired": paired,
    "ranks_sha256": digest(ROOT / "RANKS.json"),
    "unicode_bm25_sha256": digest(ROOT / "BM25_UNICODE.json")
})
print(counts)
print(paired)
