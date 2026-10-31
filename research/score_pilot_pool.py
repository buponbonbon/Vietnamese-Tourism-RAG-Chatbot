"""Offline CPU inference on a frozen, train-only draft pilot; no quality metrics.

Scores every selected candidate, preserving negative logits and membership. This
checks execution and latency only. It does not calibrate a threshold or train.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

from research import benchmark_v2 as v2
from research.parallel_preflight import checked_pool


def select_rows(questions, grouped, selected_ids, top_k):
    ids = set(selected_ids)
    if len(ids) != len(selected_ids) or not ids or top_k < 1:
        raise ValueError('Pilot IDs must be nonempty and unique; top-k must be positive')
    rows = []
    for qid in sorted(ids):
        if qid not in questions or questions[qid]['split'] != 'train':
            raise ValueError('Draft inference is restricted to train questions')
        group = sorted(grouped.get(qid, []), key=lambda r: int(r['rank']))
        if len(group) < top_k:
            raise ValueError('Insufficient frozen candidates for pilot query')
        rows.extend(group[:top_k])
    return rows


def attach_scores(rows, scores):
    if len(rows) != len(scores) or any(not math.isfinite(float(s)) for s in scores):
        raise ValueError('Scores must be finite and cover every candidate exactly once')
    grouped = defaultdict(list)
    for row, score in zip(rows, scores):
        item = dict(row, raw_logit=float(score))
        grouped[row['benchmark_id']].append(item)
    output = []
    for qid in sorted(grouped):
        group = sorted(grouped[qid], key=lambda r: (-r['raw_logit'], int(r['rank']), r['candidate_chunk_id']))
        output.extend(dict(r, rerank_rank=rank) for rank, r in enumerate(group, 1))
    return output


def run(args):
    if args.out.exists():
        raise FileExistsError('Choose a new output version: ' + str(args.out))
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError('batch-size and threads must be positive')
    questions, _, _, grouped = checked_pool(args.bundle, args.corpus, args.candidates)
    pilot = json.loads((args.pilot_packets/'packet_manifest.json').read_text())
    if pilot['benchmark_sha256'] != v2.digest(args.bundle/'benchmark_draft.csv') or pilot['candidate_input_sha256'] != v2.digest(args.candidates):
        raise ValueError('Pilot packet provenance mismatch')
    rows = select_rows(questions, grouped, pilot['selected_benchmark_ids'], args.top_k)
    model_path = args.model.resolve()
    required = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors')
    if any(not (model_path/name).is_file() for name in required):
        raise FileNotFoundError('A complete local checkpoint is required; no network fallback')
    model_hashes = {name: v2.digest(model_path/name) for name in required}
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).float().cpu().eval()
    load_seconds = time.perf_counter()-start
    scores, timings = [], []
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset+args.batch_size]
        tick = time.perf_counter()
        inputs = tokenizer([[r['query'], r['candidate_text']] for r in batch],
                           padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        with torch.inference_mode():
            logits = model(**inputs).logits
        if logits.numel() != len(batch):
            raise ValueError('Checkpoint must produce exactly one scalar per candidate')
        scores.extend(logits.reshape(-1).float().tolist())
        timings.append(time.perf_counter()-tick)
        if offset == 0 or (offset//args.batch_size+1) % 10 == 0:
            print(f'Scored {min(offset+len(batch), len(rows))}/{len(rows)} frozen candidates', flush=True)
    output = attach_scores(rows, scores)
    out = v2.immutable_output(args.out)
    v2.write_csv(out/'raw_scores.csv', output)
    v2.write_json(out/'inference_manifest.json', dict(
        purpose='Training-query draft pipeline smoke; no evidence of retrieval quality',
        paper_ready=False, human_review_complete=False, training_performed=False,
        heldout_query_inference_performed=False, metrics_computed=False,
        candidate_pool_sha256=v2.digest(args.candidates),
        pilot_manifest_sha256=v2.digest(args.pilot_packets/'packet_manifest.json'),
        benchmark_sha256=v2.digest(args.bundle/'benchmark_draft.csv'),
        checkpoint=str(model_path), checkpoint_hashes=model_hashes,
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        device='cpu', precision='float32', padding='max_length', max_length=512,
        batch_size=args.batch_size, cpu_threads=args.threads,
        queries=len(pilot['selected_benchmark_ids']), candidates=len(rows), top_k=args.top_k,
        model_load_seconds=load_seconds, tokenizer_and_inference_seconds=sum(timings),
        raw_logits_below_zero=sum(s < 0 for s in scores),
        membership_preserved=True, score_filter_applied=False,
        scores_sha256=v2.digest(out/'raw_scores.csv'),
        limitations=['Queries and labels are drafts and may require rewriting.',
                     'Only a D checkpoint was scored; no baseline comparison or accuracy claim.',
                     'Latency includes tokenization and inference, excludes retrieval and generation.',
                     'CPU FP32 smoke does not reproduce the prior CUDA BF16 test protocol.',
                     'Raw logit zero is not validated as a factual-confidence threshold.']))
    print(f'Completed CPU pilot inference: {len(rows)} candidates, {sum(timings):.2f}s; {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, default=v2.ROOT/'data/processed/vietnam_tourism_rag_dataset.csv')
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--pilot-packets', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
