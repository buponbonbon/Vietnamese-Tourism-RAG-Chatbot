"""Prepare a new, unapproved benchmark revision from recorded human decisions.

Keeps the original bundle intact, archives exclusions, and clears review approval
for changed questions/qrels. Rewrite proposals are assistant drafts, not new human
judgments. No training, inferred refusal labels, or novelty claims are produced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil

from research import benchmark_v2 as v2


def counterfactual(template, true_value, false_value):
    if template.count('{value}') != 1 or not true_value or not false_value or true_value == false_value:
        raise ValueError('A complete one-slot assertion and distinct values are required')
    return template.replace('{value}', true_value), template.replace('{value}', false_value)


def check_anchor(parent, candidate, partitions):
    if (candidate['canonical_url'] != parent['source_url']
            or partitions[candidate['chunk_id']]['source_group_id'] != parent['source_group_id']
            or partitions[candidate['chunk_id']]['split'] != parent['split']):
        raise ValueError('Replacement anchor must retain source, component and partition')


def csv_union(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ['benchmark_id']
    v2.write_csv(path, rows, fields)


def prepare(args):
    if args.out.exists():
        raise FileExistsError('Choose a new bundle version')
    v2.validate_bundle(args.parent, args.corpus, allow_draft=True)
    parent_hash = v2.digest(args.parent/'benchmark_draft.csv')
    summary = json.loads((args.reviews/'review_summary.json').read_text())
    if summary['benchmark_sha256'] != parent_hash:
        raise ValueError('Review ledger belongs to another benchmark version')
    review_manifest = json.loads((args.reviews/'output_manifest.json').read_text())
    for name in ('question_consensus.csv', 'qrel_consensus.csv'):
        if v2.digest(args.reviews/name) != review_manifest[name]:
            raise ValueError('Review ledger changed after reconciliation')
    parents = v2.read_csv(args.parent/'benchmark_draft.csv')
    reviews = {r['benchmark_id']:r for r in v2.read_csv(args.reviews/'question_consensus.csv')}
    proposals = v2.read_csv(args.rewrites)
    if len({r['family_id'] for r in proposals}) != len(proposals):
        raise ValueError('Duplicate rewrite family')
    rewrites = {r['family_id']:r for r in proposals}
    if any(r['parent_benchmark_sha256'] != parent_hash for r in proposals):
        raise ValueError('Rewrite proposals refer to a different original')
    chunks = {r['chunk_id']:r for r in v2.load_corpus(args.corpus)}
    partitions = {r['chunk_id']:r for r in v2.read_csv(args.parent/'corpus_manifest.csv')}
    corrects = {q['family_id']:q for q in parents if q['proposed_behavior']=='correct'}
    retained, excluded, changes, risks = [], [], [], []
    changed_ids = set()
    for parent in parents:
        q = dict(parent)
        review = reviews[q['benchmark_id']]
        decision = review['consensus_decision']
        if not decision or not review['consensus_behavior']:
            raise ValueError('Resolve categorical disagreements before preparing a revision')
        if decision == 'REJECT':
            excluded.append(dict(parent, recorded_decision=decision, review_notes=review['notes_1']))
            continue
        proposal = rewrites.get(q['family_id'])
        changed = False
        q.update(parent_query=parent['query'], parent_reference_answer=review['reference_1'],
                 parent_evidence_search_notes=review['search_notes_1'],
                 parent_review_decision=decision, proposed_reference_answer='', revision_origin='',
                 revision_rationale='', task_type=review['task_type_1'],
                 target_behavior=review['consensus_behavior'], reference_answer=review['reference_1'],
                 evidence_search_notes=review['search_notes_1'], annotation_status='REVIEW_RECORDED',
                 reviewer_1_id=review['reviewer_1_id'], reviewer_2_id=review['reviewer_2_id'],
                 reviewer_1_label=review['reviewer_1_label'], reviewer_2_label=review['reviewer_2_label'])
        needs_change = (decision == 'REVISE' or (proposal and proposal['force_correct_revision']=='1'
                                                and q['proposed_behavior']=='correct'))
        if needs_change:
            if proposal is None:
                raise ValueError('Missing concrete rewrite: ' + q['benchmark_id'])
            old_correct = corrects[q['family_id']]
            anchor_id = proposal['anchor_override'] or parent['anchor_chunk_id']
            anchor = chunks[anchor_id]
            check_anchor(parent, anchor, partitions)
            true_value = proposal['true_value_override'] or old_correct['trap_original_value']
            false_value = proposal['false_value_override'] or old_correct['trap_replacement_value']
            if v2.normalize(true_value) not in anchor['text_content']:
                raise ValueError('Proposed answer value is absent from the selected anchor')
            if q['proposed_behavior']=='answer':
                query = proposal['proposed_answer_query']
                proposed_answer = true_value
                if true_value in query:
                    risks.append(dict(benchmark_id=q['benchmark_id'], issue='ANSWER_VALUE_IN_QUERY', detail=true_value))
            elif q['proposed_behavior']=='correct':
                truth, falsehood = counterfactual(proposal['assertion_template'], true_value, false_value)
                query = f'Theo nguồn đã lưu, nhận định “{falsehood}” có đúng không? Nếu sai, hãy sửa và dẫn bằng chứng.'
                proposed_answer = 'Nhận định không khớp với nguồn. ' + truth
                q.update(trap_original_value=true_value, trap_replacement_value=false_value,
                         trap_type='review_pending_factual_counterfactual')
            else:
                raise ValueError('Refusal rewrites require separate evidence-scope design')
            if not query or '[GIÁ TRỊ CẦN XÁC MINH]' in query or '...' in query or '…' in query:
                raise ValueError('Rewrite is empty, contains a masked placeholder or remains truncated')
            q.update(query=query, anchor_chunk_id=anchor_id, anchor_evidence=anchor['text_content'],
                     task_type=proposal['proposed_task_type'], target_behavior='', reference_answer='',
                     evidence_search_notes='',
                     proposed_reference_answer=proposed_answer, annotation_status='TODO',
                     revision_origin='Assistant-grounded proposal; not approved by returned reviews',
                     revision_rationale=proposal['rationale'])
            for key in v2.REVIEW_FIELDS:
                q[key] = ''
            changed_ids.add(q['benchmark_id'])
            changed = True
        if q['proposed_behavior']=='refuse':
            risks.append(dict(benchmark_id=q['benchmark_id'], issue='REFUSAL_SCOPE_PENDING',
                              detail='Source-level absence is not corpus-wide absence; dense/RRF and neighbor review pending.'))
        if q['task_type']=='other':
            risks.append(dict(benchmark_id=q['benchmark_id'], issue='TOURISM_TASK_RELEVANCE_REVIEW',
                              detail='Check usefulness for tourist decisions; management/statistics tasks must be reported separately.'))
        if q['proposed_behavior']=='correct':
            true_value = q['trap_original_value']
            text = q['anchor_evidence']
            position = text.find(true_value)
            before = text[max(0,position-35):position]
            if re.search(r'khoảng|trên|hơn|gần|ước|ít nhất', before, re.I):
                risks.append(dict(benchmark_id=q['benchmark_id'], issue='APPROXIMATE_VALUE_NOT_STRICT_CONTRADICTION',
                                  detail='A small numeric change may remain compatible with an approximate/lower-bound source statement.'))
            if re.fullmatch(r'\d{1,3}-\d{1,3}',true_value):
                risks.append(dict(benchmark_id=q['benchmark_id'], issue='RANGE_COUNTERFACTUAL_REVIEW',
                                  detail='Check that the changed range is contradicted rather than a compatible subrange.'))
        retained.append(q)
        changes.append(dict(benchmark_id=q['benchmark_id'], family_id=q['family_id'], split=q['split'],
                            recorded_decision=decision, query_changed=str(int(changed)),
                            old_query=parent['query'], proposed_query=q['query'],
                            old_anchor=parent['anchor_chunk_id'], proposed_anchor=q['anchor_chunk_id'],
                            review_status=q['annotation_status']))
    selected = {q['benchmark_id']:q for q in retained}
    qrels, seen = [], set()
    for row in v2.read_csv(args.reviews/'qrel_consensus.csv'):
        qid, cid = row['benchmark_id'], row['chunk_id']
        if qid not in selected:
            continue
        changed = qid in changed_ids
        qrel = dict(benchmark_id=qid, chunk_id=cid, relevance_grade='' if changed else row['consensus_grade'],
                    evidence_quote='' if changed else row['evidence_quote_1'],
                    annotation_status='TODO' if changed else 'REVIEW_RECORDED',
                    parent_relevance_grade=row['consensus_grade'], parent_evidence_quote=row['evidence_quote_1'],
                    **{key:'' for key in v2.REVIEW_FIELDS})
        if not changed:
            for key in ('reviewer_1_id','reviewer_2_id','reviewer_1_label','reviewer_2_label'):
                qrel[key] = row[key]
        qrels.append(qrel); seen.add((qid,cid))
    for qid, q in selected.items():
        if (qid,q['anchor_chunk_id']) not in seen:
            qrels.append(dict(benchmark_id=qid,chunk_id=q['anchor_chunk_id'],relevance_grade='',
                              evidence_quote='',annotation_status='TODO',parent_relevance_grade='',parent_evidence_quote='',
                              **{key:'' for key in v2.REVIEW_FIELDS}))
    out = v2.immutable_output(args.out)
    shutil.copyfile(args.parent/'corpus_manifest.csv',out/'corpus_manifest.csv')
    csv_union(out/'benchmark_draft.csv',retained)
    csv_union(out/'qrels_review.csv',qrels)
    csv_union(out/'excluded_by_recorded_review.csv',excluded)
    csv_union(out/'question_change_log.csv',changes)
    v2.write_csv(out/'draft_risks.csv',risks,['benchmark_id','issue','detail'])
    meta = deepcopy(json.loads((args.parent/'build_manifest.json').read_text()))
    meta.update(paper_ready=False, rows=len(retained), draft_identity_sha256=v2.identity_digest(retained),
                draft_behavior_counts=dict(Counter(q['proposed_behavior'] for q in retained)),
                parent_bundle=str(args.parent.resolve()), parent_benchmark_sha256=parent_hash,
                parent_review_summary_sha256=v2.digest(args.reviews/'review_summary.json'),
                rewrite_proposals_sha256=v2.digest(args.rewrites),
                revision_summary=dict(excluded_rows=len(excluded), rewritten_queries=len(changed_ids),
                                      rewritten_answer_queries=sum(q['proposed_behavior']=='answer' and q['benchmark_id'] in changed_ids for q in retained),
                                      rewritten_correction_queries=sum(q['proposed_behavior']=='correct' and q['benchmark_id'] in changed_ids for q in retained),
                                      changed_anchor_rows=sum(r['old_anchor']!=r['proposed_anchor'] for r in changes),
                                      draft_risk_counts=dict(Counter(r['issue'] for r in risks))))
    meta['output_hashes'] = {name:v2.digest(out/name) for name in ('corpus_manifest.csv','benchmark_draft.csv','qrels_review.csv')}
    meta['limitations'] += ['Assistant rewrites and new references have not received human approval.',
                            'Changed-query qrels were reset; parent judgments remain in the lineage fields.',
                            'All refusal proposals remain pending corpus-wide evidence review.',
                            'The refusal drafting template is uniform; realistic task diversity remains to be improved.',
                            'Numeric perturbations of approximate values/ranges need semantic contradiction checks.',
                            'Tourism statistics/management questions are not all tourist-decision tasks.']
    v2.write_json(out/'build_manifest.json',meta)
    result=v2.validate_bundle(out,args.corpus,allow_draft=True)
    assert not result['review_gate_passed']
    print(json.dumps(dict(rows=len(retained),qrels=len(qrels),**meta['revision_summary'],out=str(out)),ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--reviews',type=Path,required=True)
    parser.add_argument('--rewrites',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--corpus',type=Path,default=v2.ROOT/'data/processed/vietnam_tourism_rag_dataset.csv')
    prepare(parser.parse_args())


if __name__=='__main__':
    main()
