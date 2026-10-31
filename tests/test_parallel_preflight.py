"""Guard held-out membership and provenance during unreviewed preparation."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from research import benchmark_v2 as v2
from research import parallel_preflight as prep
from research.score_pilot_pool import attach_scores, select_rows


class ParallelPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.corpus = self.root/'corpus.csv'
        text = 'Ngày 12/10/2024 có lễ hội du lịch và tham quan di sản. '
        rows = [dict(chunk_id=f'c{i}', doc_id=f'd{i}', url=f'https://example.org/{i}',
                     title=f'Điểm đến {i}', province='A',
                     text_content=text+' '.join(f'vùng{i}điểm{j}' for j in range(80))) for i in range(100)]
        v2.write_csv(self.corpus, rows)
        self.bundle = self.root/'bundle'
        with contextlib.redirect_stdout(io.StringIO()):
            v2.build(argparse.Namespace(corpus=self.corpus, legacy_train=[], legacy_evaluation=[],
                                       seed=42, holdout_province=[], groups_per_split=1, out=self.bundle))
        self.questions = {q['benchmark_id']:q for q in v2.read_csv(self.bundle/'benchmark_draft.csv')}
        self.pool = self.root/'pool'
        self.pool.mkdir()
        self.path = self.pool/'candidate_review.csv'
        self.rows = [dict(benchmark_id=q['benchmark_id'], query=q['query'], query_split=q['split'],
                          candidate_chunk_id=q['anchor_chunk_id'], candidate_split=q['split'],
                          train_partition_eligible=str(int(q['split']=='train')), rank=1,
                          label_status='UNLABELED', relevance_grade='', annotation_status='TODO')
                     for q in self.questions.values()]
        self.save_pool()

    def save_pool(self):
        v2.write_csv(self.path, self.rows)
        v2.write_json(self.pool/'mining_manifest.json', dict(
            corpus_sha256=v2.digest(self.corpus), benchmark_sha256=v2.digest(self.bundle/'benchmark_draft.csv'),
            candidate_review_sha256=v2.digest(self.path), queries=len(self.questions), candidates=len(self.rows)))

    def test_pool_hash_mismatch_blocks_use(self):
        self.path.write_bytes(self.path.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError, 'provenance/hash'):
            prep.checked_pool(self.bundle, self.corpus, self.path)

    def test_eligibility_is_derived_from_frozen_partitions(self):
        row = next(r for r in self.rows if r['query_split']=='test')
        row['train_partition_eligible'] = '1'
        self.save_pool()
        with self.assertRaisesRegex(ValueError, 'eligibility'):
            prep.checked_pool(self.bundle, self.corpus, self.path)

    def test_query_revision_cannot_silently_reuse_old_pool(self):
        self.rows[0]['query'] = 'Changed query requires remine'
        self.save_pool()
        with self.assertRaisesRegex(ValueError, 'query or partition'):
            prep.checked_pool(self.bundle, self.corpus, self.path)

    def test_pilot_selects_whole_train_families(self):
        result = prep.pilot_questions(self.questions, 1, 42)
        self.assertEqual(len(result), 3)
        self.assertEqual({q['split'] for q in result}, {'train'})
        self.assertEqual({q['proposed_behavior'] for q in result}, set(v2.BEHAVIORS))
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            prep.pilot_questions(self.questions, 2, 42)

    def test_pilot_packets_have_same_ids_and_blank_judgments(self):
        out = self.root/'packets'
        with contextlib.redirect_stdout(io.StringIO()):
            prep.pilot(argparse.Namespace(bundle=self.bundle, corpus=self.corpus, candidates=self.path,
                                         families=1, seed=42, top_k=1, out=out))
        first = v2.read_csv(out/'reviewer_1_questions.csv')
        self.assertEqual(first, v2.read_csv(out/'reviewer_2_questions.csv'))
        self.assertEqual(len(first), 3)
        self.assertTrue(all(not q['behavior_label'] and not q['reviewer_id'] for q in first))
        self.assertFalse(any('proposed_behavior' in q or 'split' in q for q in first))

    def test_draft_inference_rejects_heldout_queries(self):
        qid = next(q['benchmark_id'] for q in self.questions.values() if q['split']=='test')
        with self.assertRaisesRegex(ValueError, 'restricted to train'):
            select_rows(self.questions, {}, [qid], 1)

    def test_negative_scores_preserve_candidate_membership(self):
        rows = [dict(benchmark_id='train-q', rank=i, candidate_chunk_id=f'c{i}') for i in (1, 2, 3)]
        scored = attach_scores(rows, [-2., 1., -2.])
        self.assertEqual([r['candidate_chunk_id'] for r in scored], ['c2', 'c1', 'c3'])
        self.assertEqual(len(scored), len(rows))
        self.assertEqual([r['rerank_rank'] for r in scored], [1, 2, 3])
        with self.assertRaisesRegex(ValueError, 'finite'):
            attach_scores(rows, [1., float('nan'), 2.])

    def test_cross_role_reuse_separates_rows_and_distinct_texts(self):
        report = prep.overlap_report([dict(query='a', positive='unrelated', negative='known  text')],
                                     [dict(query='b', positive='known text', negative='x'),
                                      dict(query='c', positive='known text', negative='y')])
        self.assertEqual(report['eval_positive_in_train_negative'], dict(rows=2, distinct_texts=1))
        self.assertEqual(report['query_intersection'], 0)

    def test_plan_is_blocked_and_does_not_invent_annotation_budget(self):
        out = self.root/'plan'
        with contextlib.redirect_stdout(io.StringIO()):
            prep.plan(argparse.Namespace(bundle=self.bundle, corpus=self.corpus, out=out))
        plan = json.loads((out/'experiment_plan.json').read_text())
        self.assertFalse(plan['training_started'])
        self.assertIsNone(plan['common_proposed_settings']['addition_count'])
        self.assertIsNone(plan['common_proposed_settings']['total_optimizer_updates'])
        self.assertEqual(len(plan['runs']), 24)
        self.assertEqual({r['seed'] for r in plan['runs']}, {42, 123, 999})


if __name__ == '__main__':
    unittest.main()
