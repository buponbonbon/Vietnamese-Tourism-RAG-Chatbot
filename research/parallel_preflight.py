"""Preparation while human review is pending; never supplies labels or trains.

Audit legacy text reuse, validate frozen BM25 pools, select a train-only annotation
pilot, and write a blocked experimental matrix. All outputs use new directories.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from research import benchmark_v2 as v2

ROOT = v2.ROOT


def checked_pool(bundle, corpus, candidates):
    """Check provenance and membership instead of trusting eligibility columns."""
    v2.validate_bundle(bundle, corpus, allow_draft=True)
    questions = {q['benchmark_id']: q for q in v2.read_csv(bundle/'benchmark_draft.csv')}
    chunks = {r['chunk_id']: r for r in v2.read_csv(bundle/'corpus_manifest.csv')}
    rows = v2.read_csv(candidates)
    manifest = json.loads((candidates.parent/'mining_manifest.json').read_text())
    expected = {'corpus_sha256': v2.digest(corpus),
                'benchmark_sha256': v2.digest(bundle/'benchmark_draft.csv'),
                'candidate_review_sha256': v2.digest(candidates)}
    if any(manifest.get(k) != value for k, value in expected.items()):
        raise ValueError('Candidate provenance/hash mismatch')
    seen, grouped = set(), defaultdict(list)
    for row in rows:
        qid, cid = row['benchmark_id'], row['candidate_chunk_id']
        if qid not in questions or cid not in chunks or (qid, cid) in seen:
            raise ValueError('Unknown or duplicate candidate membership')
        seen.add((qid, cid))
        q, chunk = questions[qid], chunks[cid]
        if row['query'] != q['query'] or row['query_split'] != q['split'] or row['candidate_split'] != chunk['split']:
            raise ValueError('Candidate query or partition differs from frozen inputs')
        eligible = str(int(q['split'] == chunk['split'] == 'train'))
        if row['train_partition_eligible'] != eligible or row['label_status'] != 'UNLABELED':
            raise ValueError('Invalid eligibility or unexpected candidate labels')
        if row['relevance_grade'] or row['annotation_status'] != 'TODO':
            raise ValueError('Preparation requires an unreviewed candidate pool')
        grouped[qid].append(row)
    for group in grouped.values():
        ranks = sorted(int(r['rank']) for r in group)
        if ranks != list(range(1, len(group)+1)):
            raise ValueError('Candidate ranks must be unique and contiguous')
    if len(grouped) != manifest['queries'] or len(rows) != manifest['candidates']:
        raise ValueError('Candidate counts differ from mining manifest')
    return questions, chunks, rows, grouped


def pilot_questions(questions, families, seed):
    """Select whole TRAIN families before inspecting scores or human judgments."""
    grouped = defaultdict(list)
    for q in questions.values():
        if q['split'] == 'train':
            grouped[q['family_id']].append(q)
    eligible = sorted(k for k, group in grouped.items()
                      if len(group) == 3 and {q['proposed_behavior'] for q in group} == set(v2.BEHAVIORS))
    if families < 1 or len(eligible) < families:
        raise ValueError('Insufficient complete train families for pilot')
    random.Random(seed).shuffle(eligible)
    return sorted((q for family in eligible[:families] for q in grouped[family]), key=lambda q: q['benchmark_id'])


def pilot(args):
    questions, _, _, grouped = checked_pool(args.bundle, args.corpus, args.candidates)
    selected = pilot_questions(questions, args.families, args.seed)
    if args.top_k < 1 or any(len(grouped.get(q['benchmark_id'], [])) < args.top_k for q in selected):
        raise ValueError('Every pilot query needs the requested candidate depth')
    corpus = {r['chunk_id']: r for r in v2.load_corpus(args.corpus)}
    qrows, erows = [], []
    for q in selected:
        qid = q['benchmark_id']
        qrows.append(dict(benchmark_id=qid, query=q['query'], anchor_evidence=q['anchor_evidence'],
                          source_url=q['source_url'], task_type='', behavior_label='', reference_answer='',
                          query_revision='', evidence_search_notes='', reviewer_id='', review_status='TODO'))
        ids = {q['anchor_chunk_id']} | {r['candidate_chunk_id'] for r in grouped[qid] if int(r['rank']) <= args.top_k}
        for cid in sorted(ids):
            chunk = corpus[cid]
            erows.append(dict(benchmark_id=qid, query=q['query'], chunk_id=cid,
                              candidate_evidence=chunk['text_content'], source_url=chunk['canonical_url'],
                              relevance_grade='', evidence_quote='', reviewer_id='', review_status='TODO'))
    out = v2.immutable_output(args.out)
    for reviewer in (1, 2):
        v2.write_csv(out/f'reviewer_{reviewer}_questions.csv', qrows)
        v2.write_csv(out/f'reviewer_{reviewer}_qrels.csv', erows)
    v2.write_json(out/'packet_manifest.json', dict(
        schema=v2.SCHEMA, paper_ready=False, purpose='Train-only annotation-instruction pilot; excluded from held-out results',
        benchmark_sha256=v2.digest(args.bundle/'benchmark_draft.csv'),
        candidate_input_sha256=v2.digest(args.candidates), selection_seed=args.seed,
        selected_benchmark_ids=[q['benchmark_id'] for q in selected], selected_split='train',
        families=args.families, rows_per_reviewer=len(qrows), qrel_rows_per_reviewer=len(erows),
        candidate_top_k=args.top_k, hidden=['proposed_behavior', 'target_behavior', 'split', 'other_reviewer_labels'],
        limitations=['BM25 plus anchor evidence only; dense/RRF review outstanding.',
                     'Small pool cannot establish corpus-wide unanswerability.',
                     'Select unclear/NEEDS_REVIEW when the evidence is insufficient.']))
    print(f'Pilot: {len(qrows)} train questions / {len(erows)} passage reviews per reviewer: {out}')


def overlap_report(train, evaluation):
    """Exact NFC/whitespace reuse; counts rows separately from distinct passages."""
    indexes = {role: {v2.normalize(r.get(role, r.get('pos' if role == 'positive' else 'neg')))
                      for r in train} - {''} for role in ('positive', 'negative')}
    counts = {}
    for erole in ('positive', 'negative'):
        for trole in ('positive', 'negative'):
            hits = [v2.normalize(r.get(erole, r.get('pos' if erole == 'positive' else 'neg')))
                    for r in evaluation]
            hits = [text for text in hits if text and text in indexes[trole]]
            counts[f'eval_{erole}_in_train_{trole}'] = dict(rows=len(hits), distinct_texts=len(set(hits)))
    for field in ('query', 'doc_id'):
        a = {v2.normalize(r.get(field)) for r in train} - {''}
        b = {v2.normalize(r.get(field)) for r in evaluation} - {''}
        counts[f'{field}_intersection'] = len(a & b)
    return counts


def audit(args):
    questions, chunks, candidates, grouped = checked_pool(args.bundle, args.corpus, args.candidates)
    comparisons = []
    loaded = {path: json.loads(path.read_text(encoding='utf-8-sig')) for path in args.train+args.evaluation}
    for tpath in args.train:
        for epath in args.evaluation:
            comparisons.append(dict(train=tpath.name, evaluation=epath.name,
                                    train_rows=len(loaded[tpath]), evaluation_rows=len(loaded[epath]),
                                    **overlap_report(loaded[tpath], loaded[epath])))
    source_splits = defaultdict(set)
    for r in chunks.values():
        if r['source_group_id']:
            source_splits[r['source_group_id']].add(r['split'])
    heldout = [q for q in questions.values() if q['split'] in ('validation', 'calibration', 'test')]
    train_view = [r for r in candidates if questions[r['benchmark_id']]['split'] == 'train'
                  and chunks[r['candidate_chunk_id']]['split'] == 'train']
    reused_candidate_ids = {r['candidate_chunk_id'] for r in candidates}
    # This is an annotation queue, never a training dataset or negative labels.
    fields = list(candidates[0])
    out = v2.immutable_output(args.out)
    v2.write_csv(out/'train_partition_candidate_review.csv', train_view, fields)
    summary = dict(
        paper_ready=False, confirmatory_evaluation=False, training_performed=False,
        input_hashes={str(p): v2.digest(p) for p in loaded}, legacy_exact_reuse=comparisons,
        v2=dict(questions=len(questions), source_groups_spanning_partitions=sum(len(s)>1 for s in source_splits.values()),
                heldout_queries=len(heldout),
                heldout_anchors_with_known_legacy_train_exposure=sum(chunks[q['anchor_chunk_id']]['legacy_train_exposed']=='1' for q in heldout),
                heldout_anchors_with_known_legacy_eval_exposure=sum(chunks[q['anchor_chunk_id']]['legacy_evaluation_exposed']=='1' for q in heldout)),
        pool=dict(queries=len(grouped), candidates=len(candidates), unique_chunks=len(reused_candidate_ids),
                  query_split_counts=dict(Counter(q['split'] for q in questions.values() if q['benchmark_id'] in grouped)),
                  candidate_partition_counts=dict(Counter(r['candidate_split'] for r in candidates)),
                  train_partition_review_candidates=len(train_view),
                  nontrain_candidates_excluded_from_train_queue=len(candidates)-len(train_view)),
        frozen_pool_sha256=v2.digest(args.candidates),
        train_queue_sha256=v2.digest(out/'train_partition_candidate_review.csv'),
        limitations=['Exact text reuse is a contamination risk signal, not a measured causal effect.',
                     'Known exposure uses observable IDs and exact normalized text, not all possible semantic reuse.',
                     'Retrieving held-out documents for evaluation is allowed; training exports must exclude them.',
                     'Candidates remain unlabeled; none is an approved negative.'])
    v2.write_json(out/'parallel_audit.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def plan(args):
    v2.validate_bundle(args.bundle, args.corpus, allow_draft=True)
    common = dict(initial_checkpoint='BAAI/bge-reranker-base', initialization_revision=None,
                  max_length=512, optimizer='AdamW', learning_rate=2e-5,
                  effective_batch_size=16, total_optimizer_updates=None, precision=None,
                  base_training_sha256=None, addition_count=None,
                  margin_policy='explicit_constant_override', margin=1.0,
                  primary_selection='validation_only; metric to freeze after benchmark review')
    descriptions = {'B0': 'No additions', 'B1': 'Ordinary additions',
                    'B2': 'Synthetic adversarial additions', 'B3': 'Natural retrieved hard negatives',
                    'B4': 'Prespecified synthetic/natural mixture'}
    rows = []
    for arm, description in descriptions.items():
        for seed in (42, 123, 999):
            rows.append(dict(run_id=f'data_{arm}_s{seed}', experiment='data', arm=arm, seed=seed,
                             description=description, loss='unweighted_hinge', fixed_training_arm=arm,
                             status='BLOCKED_REVIEW_AND_INPUT_FREEZE'))
    for loss in ('unweighted_hinge', 'sigmoid_weighted', 'detached_sigmoid_weighted'):
        for seed in (42, 123, 999):
            rows.append(dict(run_id=f'loss_{loss}_s{seed}', experiment='loss', arm='B4', seed=seed,
                             description='Identical B4 data, explicit margin and update budget', loss=loss,
                             fixed_training_arm='B4', status='BLOCKED_REVIEW_AND_INPUT_FREEZE'))
    out = v2.immutable_output(args.out)
    v2.write_csv(out/'experiment_matrix.csv', rows)
    v2.write_json(out/'experiment_plan.json', dict(
        status='PROVISIONAL_NOT_PREREGISTERED', training_started=False,
        benchmark_sha256=v2.digest(args.bundle/'benchmark_draft.csv'),
        data_runs=15, loss_comparison_rows=9,
        duplicate_control_note='The B4 unweighted control may be reused if every effective setting and input hash matches; 21 distinct runs in that case.',
        common_proposed_settings=common, runs=rows,
        blocked_on=['Human-reviewed benchmark and qrels; strict validation gate',
                    'Approved train-only base/addition banks, with matched query/source composition',
                    'Addition count and mixture ratio selected using train availability only',
                    'One common optimizer-update budget, precision and pinned initialization revision',
                    'Executable trainer logging effective margins, weighting/gradient behavior and updates'],
        constraints=['B1/B2/B3/B4 have exactly the same addition count; B0 has zero additions.',
                     'All data arms use the same optimizer-update budget, not the same epochs.',
                     'Loss comparison fixes B4 data/order/margins and changes only the specified weighting.',
                     'Detached weighting retains the multiplier; it is not alpha-off.',
                     'No test-dependent tuning or cherry-picking seeds.']))
    print(f'Wrote blocked plan: 15 data rows + 9 loss rows (21 distinct runs if the control is shared): {out}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('pilot', 'audit', 'plan'):
        child = sub.add_parser(name)
        child.add_argument('--bundle', type=Path, required=True)
        child.add_argument('--corpus', type=Path, default=ROOT/'data/processed/vietnam_tourism_rag_dataset.csv')
        child.add_argument('--out', type=Path, required=True)
        if name != 'plan':
            child.add_argument('--candidates', type=Path, required=True)
        if name == 'pilot':
            child.add_argument('--families', type=int, default=10)
            child.add_argument('--top-k', type=int, default=5)
            child.add_argument('--seed', type=int, default=42)
        elif name == 'audit':
            child.add_argument('--train', type=Path, nargs='+', default=[ROOT/f'data/processed/train_moc_{n}.json' for n in (0, 6, 13)])
            child.add_argument('--evaluation', type=Path, nargs='+', default=[ROOT/f'data/processed/cross_encoder_{n}.json' for n in ('VAL', 'TEST')])
    args = parser.parse_args(argv)
    {'pilot': pilot, 'audit': audit, 'plan': plan}[args.command](args)


if __name__ == '__main__':
    main()
