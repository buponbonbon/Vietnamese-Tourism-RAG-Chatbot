"""Recover review identities from immutable text when the export mapping is absent.

This is a recovery audit, not the strict benchmark freeze gate. It never supplies
human labels, adjudicates disagreements, or certifies reviewer independence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil

from research import benchmark_v2 as v2
from research.reconcile_reviews import agreement, read_workbook


def bind_review(review, questions, chunks):
    """Require unique exact normalized query, anchor, URL and passage identities."""
    by_query = defaultdict(list)
    for q in questions:
        by_query[v2.normalize(q['query'])].append(q)
    mapping, bound_questions, bound_pairs = {}, {}, {}
    reviewer_ids = set()
    for opaque, row in review['CAU_HOI'].items():
        matches = by_query[v2.normalize(row['query'])]
        if len(matches) != 1:
            raise ValueError('Question has no unique exact wording match: ' + opaque)
        q = matches[0]
        qid = q['benchmark_id']
        if qid in bound_questions:
            raise ValueError('Multiple returned rows bind to one question: ' + qid)
        if v2.normalize(row['anchor_evidence']) != v2.normalize(q['anchor_evidence']):
            raise ValueError('Anchor changed: ' + opaque)
        if v2.canonical_url(row['source_url']) != v2.canonical_url(q['source_url']):
            raise ValueError('Question source changed: ' + opaque)
        mapping[opaque] = qid
        bound_questions[qid] = dict(row, benchmark_id=qid, original_review_id=opaque)
        reviewer_ids.add(v2.normalize(row['reviewer_id']))
    for (opaque, cid), row in review['BANG_CHUNG'].items():
        if opaque not in mapping or cid not in chunks:
            raise ValueError('Unknown question or corpus chunk in passage row')
        qid = mapping[opaque]
        chunk = chunks[cid]
        if v2.normalize(row['query']) != v2.normalize(bound_questions[qid]['query']):
            raise ValueError('Passage row has different query wording: ' + opaque)
        if v2.normalize(row['candidate_evidence']) != v2.normalize(chunk['text_content']):
            raise ValueError('Passage text differs from corpus: ' + cid)
        if v2.canonical_url(row['source_url']) != v2.canonical_url(chunk['url']):
            raise ValueError('Passage source differs from corpus: ' + cid)
        bound_pairs[qid, cid] = dict(row, benchmark_id=qid, original_review_id=opaque)
        reviewer_ids.add(v2.normalize(row['reviewer_id']))
    if len(reviewer_ids) != 1 or '' in reviewer_ids:
        raise ValueError('Expected one nonempty reviewer ID per file')
    return bound_questions, bound_pairs, next(iter(reviewer_ids))


def compare(bound, questions):
    first_q, first_p, first_id = bound[0]
    second_q, second_p, second_id = bound[1]
    if first_id == second_id:
        raise ValueError('Distinct reviewer IDs are required')
    if set(first_q) != set(second_q) or set(first_p) != set(second_p):
        raise ValueError('Reviewer memberships differ after exact-text recovery')
    issues = []
    def flag(qid, code, reviewer='', cid='', detail=''):
        issues.append(dict(benchmark_id=qid, chunk_id=cid, reviewer_id=reviewer,
                           issue_type=code, detail=detail))
    for qs, ps, reviewer in bound:
        for qid, row in qs.items():
            if row['review_status'] != 'DONE' or row['behavior_label'] not in (*v2.BEHAVIORS, 'unclear'):
                flag(qid, 'QUESTION_INCOMPLETE', reviewer)
            if row['question_decision'] != 'KEEP':
                flag(qid, 'QUESTION_CHANGE_REQUEST', reviewer, detail=row['question_decision'])
            if row['behavior_label'] == 'unclear':
                flag(qid, 'BEHAVIOR_UNCLEAR', reviewer)
            if row['query_revision']:
                flag(qid, 'WORDING_REQUIRES_NEW_REVIEW', reviewer)
        for (qid, cid), row in ps.items():
            if row['review_status'] != 'DONE' or row['relevance_grade'] not in ('0', '1', '2'):
                flag(qid, 'PASSAGE_INCOMPLETE', reviewer, cid)
            if row['relevance_grade'] in ('1', '2'):
                quote = v2.normalize(row['evidence_quote'])
                if not quote or quote not in v2.normalize(row['candidate_evidence']):
                    flag(qid, 'QUOTE_NOT_LITERAL', reviewer, cid)
    for qid in sorted(first_q):
        for field in ('behavior_label', 'question_decision', 'task_type'):
            if first_q[qid][field] != second_q[qid][field]:
                flag(qid, 'DISAGREEMENT_' + field.upper(), detail=first_q[qid][field]+' / '+second_q[qid][field])
    for key in sorted(first_p):
        if first_p[key]['relevance_grade'] != second_p[key]['relevance_grade']:
            flag(key[0], 'DISAGREEMENT_GRADE', cid=key[1],
                 detail=first_p[key]['relevance_grade']+' / '+second_p[key]['relevance_grade'])
    missing = sorted(q['benchmark_id'] for q in questions if q['benchmark_id'] not in first_q)
    for qid in missing:
        flag(qid, 'NO_RETURNED_QUESTION_REVIEW')
    summary = dict(
        questions_in_bundle=len(questions), recovered_questions_per_reviewer=len(first_q),
        recovered_passages_per_reviewer=len(first_p), questions_without_returned_review=len(missing),
        reviewer_counts={rid:dict(behavior=dict(Counter(r['behavior_label'] for r in qs.values())),
                                  decision=dict(Counter(r['question_decision'] for r in qs.values())))
                         for qs, ps, rid in bound},
        agreement={field:agreement([first_q[k][field] for k in sorted(first_q)],
                                   [second_q[k][field] for k in sorted(first_q)])
                   for field in ('behavior_label', 'question_decision', 'task_type')},
        issue_counts=dict(Counter(r['issue_type'] for r in issues)),
        strict_review_gate_passed=False,
        independence='NOT_ESTABLISHED_BY_RECOVERY_AUDIT; latest R02 is a post-feedback return',
        scope='Exact normalized text/URL recovery; original round-2 opaque-ID mapping unavailable',
        limitations=['Literal quotes do not establish semantic sufficiency.',
                     'No automatic final labels or human adjudication supplied.',
                     'No claims about newer inaccessible AI-edited bundles.'])
    summary['agreement']['relevance_grade'] = agreement(
        [first_p[k]['relevance_grade'] for k in sorted(first_p)],
        [second_p[k]['relevance_grade'] for k in sorted(first_p)])
    return summary, issues


def run(args):
    manifest = json.loads((args.bundle/'build_manifest.json').read_text())
    if manifest['corpus_sha256'] != v2.digest(args.corpus):
        raise ValueError('Corpus hash differs from bundle manifest')
    for name, expected in manifest['output_hashes'].items():
        if v2.digest(args.bundle/name) != expected:
            raise ValueError('Bundle hash mismatch: ' + name)
    questions = v2.read_csv(args.bundle/'benchmark_draft.csv')
    chunks = {r['chunk_id']:r for r in v2.load_corpus(args.corpus)}
    reviews = [read_workbook(p) for p in (args.first, args.second)]
    bound = [bind_review(r, questions, chunks) for r in reviews]
    summary, issues = compare(bound, questions)
    out = v2.immutable_output(args.out)
    (out/'inputs').mkdir()
    for n, p in enumerate((args.first, args.second), 1):
        shutil.copyfile(p, out/'inputs'/f'reviewer_{n:02d}_returned.xlsx')
    for n, (qs, ps, reviewer) in enumerate(bound, 1):
        v2.write_csv(out/f'reviewer_{n:02d}_questions.csv', [qs[k] for k in sorted(qs)])
        v2.write_csv(out/f'reviewer_{n:02d}_passages.csv', [ps[k] for k in sorted(ps)])
    v2.write_csv(out/'issues.csv', issues,
                 ['benchmark_id','chunk_id','reviewer_id','issue_type','detail'])
    summary['inputs'] = dict(benchmark_sha256=v2.digest(args.bundle/'benchmark_draft.csv'),
                             corpus_sha256=v2.digest(args.corpus),
                             returned_sha256=[v2.digest(p) for p in (args.first,args.second)])
    summary['outputs'] = {p.name:v2.digest(p) for p in sorted(out.glob('*.csv'))}
    v2.write_json(out/'recovery_audit.json', summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first',type=Path,required=True)
    parser.add_argument('--second',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--corpus',type=Path,default=Path('data/processed/vietnam_tourism_rag_dataset.csv'))
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
