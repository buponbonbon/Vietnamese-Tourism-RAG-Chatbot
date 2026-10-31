"""Revision lineage tests on synthetic review records, not benchmark results."""
import argparse
import contextlib
import io
import json
import unittest

import test_benchmark_v2 as fixtures
from research import benchmark_v2 as v2
from research import prepare_reviewed_revision as revision


class RevisionTests(unittest.TestCase):
    def setUp(self):
        fixtures.BundleTests.setUp(self)
        self.parents=v2.read_csv(self.bundle/'benchmark_draft.csv')
        self.reviews=self.base/'reviews';self.reviews.mkdir()
        self.selected=next(q for q in self.parents if q['proposed_behavior']=='answer')
        self.excluded=next(q for q in self.parents if q['proposed_behavior']=='correct' and q['family_id']!=self.selected['family_id'])
        rows=[]
        for q in self.parents:
            rows.append(dict(benchmark_id=q['benchmark_id'],consensus_behavior=q['proposed_behavior'],
                             consensus_decision='REJECT' if q==self.excluded else 'REVISE' if q==self.selected else 'KEEP',
                             reference_1='Fixture response',notes_1='Fixture notes',task_type_1='other',
                             search_notes_1='Fixture old search',reviewer_1_id='fixture-r1',reviewer_2_id='fixture-r2',
                             reviewer_1_label=q['proposed_behavior'],reviewer_2_label=q['proposed_behavior']))
        v2.write_csv(self.reviews/'question_consensus.csv',rows)
        evidence=[]
        for q in self.parents:
            evidence.append(dict(benchmark_id=q['benchmark_id'],chunk_id=q['anchor_chunk_id'],consensus_grade='2',
                                 evidence_quote_1=q['anchor_evidence'],reviewer_1_id='fixture-r1',reviewer_2_id='fixture-r2',
                                 reviewer_1_label='2',reviewer_2_label='2'))
        v2.write_csv(self.reviews/'qrel_consensus.csv',evidence)
        v2.write_json(self.reviews/'review_summary.json',dict(benchmark_sha256=v2.digest(self.bundle/'benchmark_draft.csv')))
        v2.write_json(self.reviews/'output_manifest.json',{name:v2.digest(self.reviews/name) for name in ('question_consensus.csv','qrel_consensus.csv')})
        self.proposal=dict(family_id=self.selected['family_id'],parent_benchmark_sha256=v2.digest(self.bundle/'benchmark_draft.csv'),
                           proposed_answer_query='Ngày lễ hội trong nguồn là ngày nào?',assertion_template='',
                           proposed_task_type='visit_timing',anchor_override='',true_value_override='',false_value_override='',
                           force_correct_revision='0',rationale='Synthetic assistant rewrite fixture')
        self.path=self.base/'proposals.csv'
        v2.write_csv(self.path,[self.proposal])
        self.out=self.base/'revision'
        self.args=argparse.Namespace(parent=self.bundle,reviews=self.reviews,rewrites=self.path,out=self.out,corpus=self.corpus)

    def run_prepare(self):
        with contextlib.redirect_stdout(io.StringIO()): revision.prepare(self.args)

    def test_rewrite_clears_old_approval_and_search_notes(self):
        old=v2.digest(self.bundle/'benchmark_draft.csv');self.run_prepare()
        row=next(q for q in v2.read_csv(self.out/'benchmark_draft.csv') if q['benchmark_id']==self.selected['benchmark_id'])
        self.assertEqual(row['annotation_status'],'TODO')
        self.assertEqual(row['reviewer_1_id'],'')
        self.assertEqual(row['target_behavior'],'')
        self.assertEqual(row['evidence_search_notes'],'')
        self.assertEqual(row['parent_evidence_search_notes'],'Fixture old search')
        self.assertEqual(v2.digest(self.bundle/'benchmark_draft.csv'),old)

    def test_exclusions_are_archived_in_new_membership(self):
        self.run_prepare()
        self.assertNotIn(self.excluded['benchmark_id'],{q['benchmark_id'] for q in v2.read_csv(self.out/'benchmark_draft.csv')})
        self.assertEqual(v2.read_csv(self.out/'excluded_by_recorded_review.csv')[0]['benchmark_id'],self.excluded['benchmark_id'])
        self.assertFalse(v2.validate_bundle(self.out,self.corpus,allow_draft=True)['review_gate_passed'])

    def test_changed_query_qrels_cannot_silently_reuse_old_grades(self):
        self.run_prepare()
        row=next(q for q in v2.read_csv(self.out/'qrels_review.csv') if q['benchmark_id']==self.selected['benchmark_id'])
        self.assertEqual(row['relevance_grade'],'')
        self.assertEqual(row['parent_relevance_grade'],'2')
        self.assertEqual(row['reviewer_1_id'],'')

    def test_cross_partition_anchor_override_is_rejected(self):
        other=next(q for q in self.parents if q['split']!=self.selected['split'])
        self.proposal['anchor_override']=other['anchor_chunk_id'];v2.write_csv(self.path,[self.proposal])
        with self.assertRaisesRegex(ValueError,'retain source'):
            self.run_prepare()

    def test_review_hash_changes_are_rejected(self):
        path=self.reviews/'question_consensus.csv';path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'changed after reconciliation'):
            self.run_prepare()

    def test_missing_counterfactual_slot_and_equal_values_are_rejected(self):
        with self.assertRaises(ValueError): revision.counterfactual('No value slot','1','2')
        with self.assertRaises(ValueError): revision.counterfactual('Claim {value}','1','1')
        self.assertEqual(revision.counterfactual('Claim {value}','1','2'),('Claim 1','Claim 2'))

    def test_existing_version_is_not_overwritten(self):
        self.run_prepare()
        with self.assertRaises(FileExistsError): self.run_prepare()

    def test_round_without_exclusions_keeps_an_empty_archive(self):
        path=self.reviews/'question_consensus.csv'
        rows=v2.read_csv(path)
        for row in rows:
            if row['consensus_decision']=='REJECT': row['consensus_decision']='KEEP'
        v2.write_csv(path,rows)
        manifest=json.loads((self.reviews/'output_manifest.json').read_text())
        manifest['question_consensus.csv']=v2.digest(path)
        v2.write_json(self.reviews/'output_manifest.json',manifest)
        self.run_prepare()
        self.assertEqual(v2.read_csv(self.out/'excluded_by_recorded_review.csv'),[])


if __name__=='__main__':
    unittest.main()
