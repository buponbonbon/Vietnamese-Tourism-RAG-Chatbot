"""Retrieval-stage component ablation for the Vietnamese Tourism RAG project.

Scientific scope:
  full_proposed   : BM25 + dense -> RRF Top-K -> chosen reranker -> Top-N
  wo_hybrid       : dense only       -> Top-K -> SAME reranker -> Top-N
  wo_reranker     : BM25 + dense -> RRF Top-K (no reranker) -> Top-N

These are *retrieval* outcomes, not end-to-end answer quality or faithfulness.
Single annotated gold_chunk_id is the relevance reference; other relevant chunks
may exist and are NOT credited by exact-gold metrics. Use an independently
validated benchmark for a publication claim. No model training or LLM calls.

Windows PowerShell (from repository root, after activating env_rag):
  python -m results.ablation_and_system_eval --self-test
  python -m results.ablation_and_system_eval --check-only --max-queries 10 --allow-draft
  python -m results.ablation_and_system_eval --max-queries 10 --allow-draft --top-k 10 --top-n 3 --batch-size 2

To evaluate approved labels: omit --allow-draft, and annotate all selected rows DONE.
Existing output is NEVER overwritten without --overwrite.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DEFAULT = ROOT / 'results' / 'downstream_benchmark' / 'benchmark_annotation_draft_200.csv'
OUT_DEFAULT = ROOT / 'results' / 'ablation_outputs'
CORPUS = ROOT / 'data' / 'processed' / 'vietnam_tourism_rag_dataset.csv'
INVERTED = ROOT / 'data' / 'processed' / 'inverted_index.json'
FAISS_DIR = ROOT / 'data' / 'vectors'
RUNS = {'B': ('run_1', '0'), 'C': ('run_2', '6'), 'D': ('run_3', '13')}
CONDITIONS = ('full_proposed', 'wo_hybrid', 'wo_reranker')


def sha256(path: Path) -> str:
    dig = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            dig.update(chunk)
    return dig.hexdigest()


def resolve_checkpoint(variant: str, seed: int, explicit: str = '') -> Path:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
    else:
        run, moc = RUNS[variant]
        candidate = ROOT / 'models' / run / f'vietnam_tourism_reranker_moc_{moc}_seed_{seed}'
    if not all((candidate / x).is_file() for x in ('config.json', 'model.safetensors')):
        raise FileNotFoundError(f'Checkpoint missing config or weights: {candidate}')
    return candidate


def load_benchmark(path: Path, max_queries: int, allow_draft: bool) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f'Benchmark missing: {path}')
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
    elif path.suffix.lower() == '.json':
        rows = json.loads(path.read_text(encoding='utf-8-sig'))
    else:
        raise ValueError('Benchmark must be CSV or JSON')
    if not isinstance(rows, list):
        raise ValueError('Benchmark must contain a list of objects')
    if max_queries < 0:
        raise ValueError('--max-queries must be >= 0')
    rows = rows[:max_queries] if max_queries else rows
    if not rows:
        raise ValueError('Selected benchmark is empty')
    needed = {'benchmark_id', 'query', 'gold_chunk_id', 'annotation_status', 'target_behavior'}
    identifiers = set()
    selected = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or not needed.issubset(r):
            raise ValueError(f'Benchmark row {i} missing required fields: {needed - set(r) if isinstance(r, dict) else needed}')
        row = {str(k): '' if v is None else str(v).strip() for k, v in r.items()}
        bid = row['benchmark_id']
        if not bid or bid in identifiers:
            raise ValueError(f'Empty/duplicate benchmark_id at row {i}: {bid!r}')
        identifiers.add(bid)
        if not row['query'] or not row['gold_chunk_id']:
            raise ValueError(f'{bid}: query/gold_chunk_id must be nonempty')
        if row['target_behavior'].lower() not in ('answer', 'correct'):
            raise ValueError(f'{bid}: invalid target_behavior')
        if not allow_draft and row['annotation_status'].upper() != 'DONE':
            raise ValueError(f'{bid}: not DONE; for exploratory runs add --allow-draft')
        selected.append(row)
    return selected


def normalize_hits(raw: Any, limit: int) -> list[str]:
    """Take ranked (chunk_id, score) pairs; preserve rank; drop duplicate IDs."""
    result: list[str] = []
    seen = set()
    if raw is None:
        return result
    for item in raw:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise TypeError('Expected retrieval output as [(chunk_id, score), ...]')
        cid = str(item[0])
        if cid not in seen:
            result.append(cid)
            seen.add(cid)
        if len(result) >= limit:
            break
    return result


def rank_metric(gold: str, ranked_ids: list[str], top_n: int) -> dict[str, float | int | None]:
    """Exact single-gold relevance. Strictly truncate before computing metrics."""
    shortlist = ranked_ids[:top_n]
    rank = shortlist.index(gold) + 1 if gold in shortlist else None
    return {
        'gold_rank_at_n': rank,
        'hit_at_n': int(rank is not None),
        'mrr_at_n': 1.0 / rank if rank else 0.0,
        'ndcg_at_n': 1.0 / math.log2(rank + 1) if rank else 0.0,
    }


def summarize(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [r for r in rows if r['condition'] == condition]
    if not selected:
        raise ValueError(f'No samples in {condition}')
    n = len(selected)
    eligible = [r for r in selected if r['gold_in_candidate_top_k']]
    return {
        'n': n,
        'candidate_gold_recall_at_k': sum(r['gold_in_candidate_top_k'] for r in selected) / n,
        'exact_gold_hit_at_n': sum(r['hit_at_n'] for r in selected) / n,
        'mrr_at_n': sum(r['mrr_at_n'] for r in selected) / n,
        'ndcg_at_n': sum(r['ndcg_at_n'] for r in selected) / n,
        'eligible_n': len(eligible),
        'candidate_conditioned_hit_at_n': (sum(r['hit_at_n'] for r in eligible) / len(eligible)) if eligible else None,
        'candidate_conditioned_mrr_at_n': (sum(r['mrr_at_n'] for r in eligible) / len(eligible)) if eligible else None,
    }


def paired_delta(rows: list[dict[str, Any]], left: str, right: str, seed: int = 42) -> dict[str, Any]:
    """Paired resampling across benchmark IDs (not independent gold chunks)."""
    left_map = {r['benchmark_id']: r for r in rows if r['condition'] == left}
    right_map = {r['benchmark_id']: r for r in rows if r['condition'] == right}
    if left_map.keys() != right_map.keys() or not left_map:
        raise ValueError('Unpaired/incomplete ablation records')
    n = len(left_map)
    out = {'n_pairs': n}
    rng = random.Random(seed)
    for key in ('hit_at_n', 'mrr_at_n', 'ndcg_at_n'):
        diffs = [float(left_map[k][key]) - float(right_map[k][key]) for k in sorted(left_map)]
        delta = sum(diffs) / n
        boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
        out[key] = {'delta_left_minus_right': delta,
                    'bootstrap_95ci': [boots[49], boots[1949]]}
    return out


def score_pairs(reranker: Any, query: str, texts: list[str], batch_size: int) -> list[float]:
    """Raw logits, without the production confidence gate; avoids silent filtering."""
    import torch
    scores = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            pairs = [[query, text] for text in texts[start:start+batch_size]]
            inputs = reranker.tokenizer(pairs, padding=True, truncation=True,
                                        max_length=512, return_tensors='pt').to(reranker.device)
            logits = reranker.model(**inputs).logits.reshape(-1).float().cpu().tolist()
            if len(logits) != len(pairs):
                raise RuntimeError('Unexpected reranker logit shape')
            scores.extend(logits)
    return scores


def rerank_ids(reranker: Any, query: str, ids: list[str], text_by_id: dict[str, str], batch_size: int) -> list[str]:
    if not ids:
        return []
    logits = score_pairs(reranker, query, [text_by_id[cid] for cid in ids], batch_size)
    if not all(math.isfinite(x) for x in logits):
        raise RuntimeError('Non-finite reranker score')
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-logits[i], i))]


def new_record(sample: dict[str, str], condition: str, candidates: list[str], ranked: list[str], top_k: int, top_n: int) -> dict[str, Any]:
    gold = sample['gold_chunk_id']
    if len(ranked) != len(candidates) or set(ranked) != set(candidates):
        raise RuntimeError('Reranker must return a permutation of the SAME retrieval candidates')
    return {
        'benchmark_id': sample['benchmark_id'],
        'query': sample['query'],
        'gold_chunk_id': gold,
        'condition': condition,
        'top_k': top_k,
        'top_n': top_n,
        'candidate_ids': candidates,
        'ranked_ids': ranked,
        'selected_top_n_ids': ranked[:top_n],
        'gold_in_candidate_top_k': int(gold in candidates),
        **rank_metric(gold, ranked, top_n),
    }


def evaluate(benchmark: list[dict[str, str]], args: argparse.Namespace, checkpoint: Path) -> list[dict[str, Any]]:
    """Retrieve once per query/method, unload embedder, then rerank. No LLM."""
    import torch
    from tqdm import tqdm
    from src.retrieval import HybridRetriever
    from src.reranking import TourismReranker

    print('[1/3] Load retriever and corpus (no generator or LLM)')
    retriever = HybridRetriever(csv_path=str(CORPUS), index_path=str(INVERTED), faiss_index_dir=str(FAISS_DIR))
    frame = retriever.df
    if not {'chunk_id', 'text_content'}.issubset(frame.columns):
        raise ValueError('Retriever corpus missing chunk_id/text_content')
    texts: dict[str, str] = {}
    for cid, content in zip(frame['chunk_id'], frame['text_content']):
        key = str(cid)
        if key in texts and texts[key] != str(content):
            raise ValueError(f'Conflicting duplicate chunk_id in corpus: {key}')
        texts[key] = str(content)
    for sample in benchmark:
        cid = sample['gold_chunk_id']
        if cid not in texts:
            raise ValueError(f"Benchmark gold_chunk_id not in retrieval corpus: {sample['benchmark_id']} / {cid}")
        # A stale benchmark can retain the same ID after corpus content changes.
        # Catch the mismatch instead of silently evaluating the wrong evidence.
        annotated_evidence = sample.get('gold_evidence', '')
        if annotated_evidence:
            clean = lambda s: re.sub(r'\s+', ' ', s).strip()
            if clean(annotated_evidence) != clean(texts[cid]):
                raise ValueError(f"Benchmark gold_evidence differs from corpus: {sample['benchmark_id']} / {cid}")

    prepared: list[tuple[dict[str, str], list[str], list[str]]] = []
    records: list[dict[str, Any]] = []
    print('[2/3] Retrieve BM25 + dense + RRF candidates')
    for sample in tqdm(benchmark, desc='Retrieval', unit='query'):
        q = sample['query']
        bm25 = retriever.bm25_search(q, top_k=args.top_k)
        dense = retriever.semantic_search(q, top_k=args.top_k)
        fused = retriever.rrf_merge(bm25, dense, top_n=args.top_k)
        hybrid_ids = normalize_hits(fused, args.top_k)
        dense_ids = normalize_hits(dense, args.top_k)
        for cid in set(hybrid_ids + dense_ids):
            if cid not in texts:
                raise ValueError(f'Retrieved chunk_id not present in corpus: {cid}')
        prepared.append((sample, hybrid_ids, dense_ids))
        records.append(new_record(sample, 'wo_reranker', hybrid_ids, hybrid_ids, args.top_k, args.top_n))

    # Free the dense embedding model before loading the cross-encoder on 4GB GPUs.
    del retriever
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f'[3/3] Load {args.variant} seed={args.seed} reranker and rank candidates')
    reranker = TourismReranker(model_name=str(checkpoint))
    for sample, hybrid_ids, dense_ids in tqdm(prepared, desc='Reranking', unit='query'):
        full_ranked = rerank_ids(reranker, sample['query'], hybrid_ids, texts, args.batch_size)
        dense_ranked = rerank_ids(reranker, sample['query'], dense_ids, texts, args.batch_size)
        records.append(new_record(sample, 'full_proposed', hybrid_ids, full_ranked, args.top_k, args.top_n))
        records.append(new_record(sample, 'wo_hybrid', dense_ids, dense_ranked, args.top_k, args.top_n))
    return records


def self_test() -> None:
    # Pure-Python checks, no torch or model downloads.
    m = rank_metric('gold', ['irrelevant', 'gold', 'other'], 3)
    assert m['gold_rank_at_n'] == 2 and m['mrr_at_n'] == 0.5
    assert math.isclose(m['ndcg_at_n'], 1.0 / math.log2(3))
    assert rank_metric('gold', ['irrelevant', 'gold'], 1)['hit_at_n'] == 0
    assert rank_metric('gold', [], 3)['mrr_at_n'] == 0
    assert normalize_hits([('x', 3), ('x', 2), ('y', 1)], 2) == ['x', 'y']
    sample = {'benchmark_id': 'x', 'query': 'q', 'gold_chunk_id': 'gold'}
    no_rerank = new_record(sample, 'wo_reranker', ['other', 'gold'], ['other', 'gold'], 2, 1)
    full = new_record(sample, 'full_proposed', ['other', 'gold'], ['gold', 'other'], 2, 1)
    dense = new_record(sample, 'wo_hybrid', ['other'], ['other'], 2, 1)
    assert no_rerank['hit_at_n'] == 0 and full['hit_at_n'] == 1
    assert dense['gold_in_candidate_top_k'] == 0
    rows = [no_rerank, full, dense]
    assert summarize(rows, 'full_proposed')['exact_gold_hit_at_n'] == 1
    assert paired_delta(rows, 'full_proposed', 'wo_reranker')['hit_at_n']['delta_left_minus_right'] == 1
    try:
        new_record(sample, 'full_proposed', ['a', 'b'], ['a', 'a'], 2, 1)
        raise AssertionError('Permutation check failed')
    except RuntimeError:
        pass
    print('[PASS] 8 checks: rank metrics, cutoff, missing gold, duplicate retrieval IDs,')
    print('       shared-pool ranking, candidate failure, paired comparison, permutation guard')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--self-test', action='store_true', help='No GPU; test metric and selection logic')
    p.add_argument('--check-only', action='store_true', help='No GPU; check benchmark and checkpoint availability')
    p.add_argument('--benchmark', type=Path, default=BENCHMARK_DEFAULT)
    p.add_argument('--max-queries', type=int, default=0, help='0 = all; otherwise first N rows')
    p.add_argument('--allow-draft', action='store_true', help='Exploratory pilot only; not for paper reporting')
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--top-n', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--variant', choices=sorted(RUNS), default='D', help='Reranker B=0%%, C=6.85%%, D=13.7%%')
    p.add_argument('--seed', type=int, choices=(42, 123, 999), default=42)
    p.add_argument('--checkpoint', type=str, default='', help='Optional explicit saved checkpoint path')
    p.add_argument('--output-dir', type=Path, default=OUT_DEFAULT)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.top_k < 1 or not 1 <= args.top_n <= args.top_k or args.batch_size < 1:
        raise ValueError('Require top-k>=1, 1<=top-n<=top-k and batch-size>=1')
    benchmark = load_benchmark(args.benchmark, args.max_queries, args.allow_draft)
    checkpoint = resolve_checkpoint(args.variant, args.seed, args.checkpoint)
    print(f'Benchmark: {len(benchmark)} selected rows | {args.benchmark}')
    print(f'Labels: {dict(Counter(x["annotation_status"] for x in benchmark))}')
    print(f'Checkpoint: {checkpoint}')
    print(f'Retrieval: top-k={args.top_k}; evaluated top-n={args.top_n}')
    if args.allow_draft:
        print('[WARNING] DRAFT PILOT — NOT VALIDATED FOR PUBLICATION')
    if args.check_only:
        print('[PASS] Benchmark schema/status and checkpoint layout; no inference performed')
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f'ablation_{args.variant}_seed{args.seed}_n{len(benchmark)}_k{args.top_k}_top{args.top_n}'
    output_jsonl = args.output_dir / f'{tag}_records.jsonl'
    output_summary = args.output_dir / f'{tag}_summary.json'
    if not args.overwrite and (output_jsonl.exists() or output_summary.exists()):
        raise FileExistsError('Outputs already exist. Back them up, then pass --overwrite if intended.')

    rows = evaluate(benchmark, args, checkpoint)
    if len(rows) != 3 * len(benchmark):
        raise RuntimeError('Missing ablation condition rows')
    keys = [(r['benchmark_id'], r['condition']) for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError('Duplicate benchmark_id/condition rows')
    conditions = {name: summarize(rows, name) for name in CONDITIONS}
    metadata = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'experiment': 'retrieval-stage exact-gold single-label component ablation',
        'is_human_validated': not args.allow_draft,
        'known_limitations': [
            'Only the annotated gold_chunk_id counts as relevant; other valid evidence is not credited.',
            'This does not measure generator behavior, factual faithfulness, or answer correctness.',
            'First-stage retrieval recall and reranking quality affect observed scores.',
        ],
        'benchmark_file': str(args.benchmark.resolve()),
        'benchmark_sha256': sha256(args.benchmark),
        'corpus_file': str(CORPUS),
        'corpus_sha256': sha256(CORPUS),
        'inverted_index': str(INVERTED),
        'inverted_index_sha256': sha256(INVERTED),
        'faiss_index_directory': str(FAISS_DIR),
        'checkpoint': str(checkpoint),
        'checkpoint_config_sha256': sha256(checkpoint / 'config.json'),
        'checkpoint_weights_bytes': (checkpoint / 'model.safetensors').stat().st_size,
        'checkpoint_weights_mtime_ns': (checkpoint / 'model.safetensors').stat().st_mtime_ns,
        'script_sha256': sha256(Path(__file__)),
        'variant': args.variant, 'seed': args.seed, 'top_k': args.top_k,
        'top_n': args.top_n, 'n_queries': len(benchmark),
        'batch_size': args.batch_size, 'allow_draft': args.allow_draft,
        'no_confidence_filter': True,
    }
    summary = {
        'metadata': metadata,
        'conditions': conditions,
        'paired_full_minus_wo_hybrid': paired_delta(rows, 'full_proposed', 'wo_hybrid', args.seed),
        'paired_full_minus_wo_reranker': paired_delta(rows, 'full_proposed', 'wo_reranker', args.seed),
    }
    # Write temporary outputs and rename to reduce chances of incomplete files.
    temp_rows = output_jsonl.with_suffix(output_jsonl.suffix + '.tmp')
    temp_summary = output_summary.with_suffix(output_summary.suffix + '.tmp')
    with temp_rows.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temp_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    temp_rows.replace(output_jsonl)
    temp_summary.replace(output_summary)
    print('\n=========== RETRIEVAL ABLATION ===========')
    for name, d in conditions.items():
        print(f'{name:16s} | recall@K {d["candidate_gold_recall_at_k"]:.3f} | '
              f'hit@N {d["exact_gold_hit_at_n"]:.3f} | '
              f'MRR@N {d["mrr_at_n"]:.3f} | nDCG@N {d["ndcg_at_n"]:.3f}')
    print('Records:', output_jsonl)
    print('Summary:', output_summary)
    if args.allow_draft:
        print('[WARNING] Exploratory DRAFT results only, not publication-ready')


if __name__ == '__main__':
    main()
