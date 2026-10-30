"""Protocol integrity tests on synthetic fixtures, not research results."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from research import benchmark_v2 as v2


class SourceGroupingTests(unittest.TestCase):
    def test_url_alias_and_duplicate_passage_stay_together(self):
        self.assertEqual(v2.canonical_url('http://www.example.org/a/?utm_source=x#section'),
                         v2.canonical_url('https://example.org/a'))
        rows=[dict(canonical_url='https://example.org/a',doc_id='a',text_content='same',text_sha256='same'),
              dict(canonical_url='https://example.org/b',doc_id='b',text_content='same',text_sha256='same'),
              dict(canonical_url='https://example.org/b',doc_id='c',text_content='other',text_sha256='other')]
        v2.group_sources(rows)
        self.assertEqual(len({r['source_group_id'] for r in rows}),1)

    def test_legacy_negative_exposure_and_geography_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'legacy.json'
            path.write_text(json.dumps([dict(query='q',positive='elsewhere',negative='known text')]))
            rows=[dict(canonical_url='https://example.org/a',doc_id='a',text_content='known text',
                       text_sha256=v2.hashlib.sha256(b'known text').hexdigest(),province='A')]
            v2.group_sources(rows)
            exposed=v2.legacy_exposure(rows,[path])
            self.assertEqual(exposed,{rows[0]['source_group_id']})
            v2.assign_splits(rows,42,exposed,set())
            self.assertEqual(rows[0]['split'],'train')
            v2.assign_splits(rows,42,exposed,set(),['A'])
            self.assertEqual(rows[0]['split'],'quarantine')


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        self.corpus=self.base/'corpus.csv'
        text='Ngày 12/10/2024, lễ hội du lịch có hoạt động tham quan và giới thiệu di sản. '
        self.source_rows=[dict(chunk_id=f'D{i}_chunk_1',doc_id=f'D{i}',url=f'https://example.org/{i}',
                               title=f'Điểm đến {i}',province='A',
                               text_content=text+' '.join(f'vùng{i}điểm{j}' for j in range(80))+'.')
                          for i in range(100)]
        v2.write_csv(self.corpus,self.source_rows)
        self.bundle=self.base/'bundle'
        self.args=argparse.Namespace(corpus=self.corpus,legacy_train=[],legacy_evaluation=[],
                                    seed=42,holdout_province=[],groups_per_split=1,out=self.bundle)
        with contextlib.redirect_stdout(io.StringIO()):
            v2.build(self.args)

    def approve_fixture(self):
        questions=v2.read_csv(self.bundle/'benchmark_draft.csv')
        for q in questions:
            q.update(target_behavior=q['proposed_behavior'],reference_answer='Synthetic fixture only',
                     task_type='tourism decision fixture',evidence_search_notes='Synthetic test review only',
                     refusal_scope='corpus_snapshot' if q['proposed_behavior']=='refuse' else '',
                     annotation_status='VERIFIED',reviewer_1_id='fixture-reviewer-1',
                     reviewer_2_id='fixture-reviewer-2',reviewer_1_label=q['proposed_behavior'],
                     reviewer_2_label=q['proposed_behavior'])
        v2.write_csv(self.bundle/'benchmark_draft.csv',questions)
        by_id={q['benchmark_id']:q for q in questions}
        qrels=v2.read_csv(self.bundle/'qrels_review.csv')
        for qrel in qrels:
            q=by_id[qrel['benchmark_id']]
            grade='0' if q['target_behavior']=='refuse' else '2'
            qrel.update(relevance_grade=grade,evidence_quote=q['anchor_evidence'] if grade=='2' else '',
                        annotation_status='VERIFIED',reviewer_1_id='fixture-reviewer-1',
                        reviewer_2_id='fixture-reviewer-2',reviewer_1_label=grade,reviewer_2_label=grade)
        v2.write_csv(self.bundle/'qrels_review.csv',qrels)
        return questions,qrels

    def test_drafts_are_structurally_valid_but_block_confirmatory_use(self):
        result=v2.validate_bundle(self.bundle,self.corpus,allow_draft=True)
        self.assertEqual(result['rows'],12)
        self.assertFalse(result['review_gate_passed'])
        with self.assertRaisesRegex(ValueError,'not VERIFIED'):
            v2.validate_bundle(self.bundle,self.corpus)

    def test_synthetic_completed_review_fixture_passes_gate(self):
        self.approve_fixture()
        self.assertTrue(v2.validate_bundle(self.bundle,self.corpus)['review_gate_passed'])

    def test_verification_status_without_reviewers_is_insufficient(self):
        questions=v2.read_csv(self.bundle/'benchmark_draft.csv')
        for q in questions:
            q['annotation_status']='VERIFIED'
        v2.write_csv(self.bundle/'benchmark_draft.csv',questions)
        with self.assertRaisesRegex(ValueError,'two distinct reviewers'):
            v2.validate_bundle(self.bundle,self.corpus)

    def test_disagreement_requires_independent_adjudication(self):
        questions,_=self.approve_fixture()
        questions[0]['reviewer_2_label']='refuse' if questions[0]['target_behavior']!='refuse' else 'answer'
        questions[0]['adjudicator_id']=' fixture-reviewer-1 '
        questions[0]['adjudication_notes']='fixture disagreement'
        v2.write_csv(self.bundle/'benchmark_draft.csv',questions)
        with self.assertRaisesRegex(ValueError,'independent adjudication'):
            v2.validate_bundle(self.bundle,self.corpus)

    def test_quote_and_multiple_sufficient_qrels_checked(self):
        questions,qrels=self.approve_fixture()
        good=next(r for r in qrels if r['relevance_grade']=='2')
        good['evidence_quote']='This sentence does not occur in the corpus.'
        v2.write_csv(self.bundle/'qrels_review.csv',qrels)
        with self.assertRaisesRegex(ValueError,'quote missing'):
            v2.validate_bundle(self.bundle,self.corpus)

    def test_partial_relevance_does_not_make_a_refusal_contradictory(self):
        _,qrels=self.approve_fixture()
        row=next(r for r in qrels if r['relevance_grade']=='0')
        row.update(relevance_grade='1',reviewer_1_label='1',reviewer_2_label='1',
                   evidence_quote='Ngày 12/10/2024, lễ hội du lịch có hoạt động tham quan và giới thiệu di sản.')
        v2.write_csv(self.bundle/'qrels_review.csv',qrels)
        self.assertTrue(v2.validate_bundle(self.bundle,self.corpus)['review_gate_passed'])

    def test_counterfactual_membership_cannot_be_moved_or_deleted(self):
        questions=v2.read_csv(self.bundle/'benchmark_draft.csv')
        questions.pop()
        v2.write_csv(self.bundle/'benchmark_draft.csv',questions)
        with self.assertRaisesRegex(ValueError,'membership/source identities'):
            v2.validate_bundle(self.bundle,self.corpus,allow_draft=True)

    def test_manifest_and_corpus_are_frozen_and_output_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            with contextlib.redirect_stdout(io.StringIO()):
                v2.build(self.args)
        manifest=self.bundle/'corpus_manifest.csv'
        manifest.write_text(manifest.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'manifest changed'):
            v2.validate_bundle(self.bundle,self.corpus,allow_draft=True)

    def test_build_is_repeatable_byte_for_byte(self):
        second=self.base/'second'
        self.args.out=second
        with contextlib.redirect_stdout(io.StringIO()):
            v2.build(self.args)
        for name in ['corpus_manifest.csv','benchmark_draft.csv','qrels_review.csv','build_manifest.json']:
            self.assertEqual((self.bundle/name).read_bytes(),(second/name).read_bytes())

    def test_sentence_start_and_date_perturbation(self):
        row=dict(source_group_id='g',chunk_id='c',split='train',province='A',title='Lễ hội',
                 canonical_url='https://example.org/a',text_content='Ngày 31/10/2024 có lễ hội du lịch.')
        family=v2.make_family(row)
        self.assertIn('Ngày',family[0]['query'])
        self.assertEqual(family[1]['trap_replacement_value'],'30/10/2024')

    def test_partial_qrels_cannot_support_a_final_answer(self):
        _,qrels=self.approve_fixture()
        row=next(r for r in qrels if r['relevance_grade']=='2')
        row.update(relevance_grade='1',reviewer_1_label='1',reviewer_2_label='1')
        v2.write_csv(self.bundle/'qrels_review.csv',qrels)
        with self.assertRaisesRegex(ValueError,'no relevant evidence'):
            v2.validate_bundle(self.bundle,self.corpus)

    def test_strict_gate_rechecks_unresolved_near_duplicates(self):
        questions=v2.read_csv(self.bundle/'benchmark_draft.csv')
        manifest=v2.read_csv(self.bundle/'corpus_manifest.csv')
        anchor=next(q for q in questions if q['split']=='train')
        candidate=next(r['chunk_id'] for r in manifest if r['split']=='test')
        for row in self.source_rows:
            if row['chunk_id']==candidate:
                row['text_content']=anchor['anchor_evidence']+' Bổ sung.'
        v2.write_csv(self.corpus,self.source_rows)
        self.bundle=self.base/'near-bundle';self.args.out=self.bundle
        with contextlib.redirect_stdout(io.StringIO()):v2.build(self.args)
        self.approve_fixture()
        with self.assertRaisesRegex(ValueError,'near-duplicate flags'):
            v2.validate_bundle(self.bundle,self.corpus)

    def test_packets_hide_proposed_labels_and_other_reviewers(self):
        out=self.base/'packets'
        with contextlib.redirect_stdout(io.StringIO()):
            v2.packets(argparse.Namespace(bundle=self.bundle,corpus=self.corpus,out=out))
        rows=v2.read_csv(out/'reviewer_1_questions.csv')
        self.assertEqual(len(rows),12)
        self.assertNotIn('proposed_behavior',rows[0])
        self.assertNotIn('split',rows[0])
        self.assertNotIn('reviewer_2_label',rows[0])
        self.assertEqual(rows[0]['behavior_label'],'')


class NearDuplicateTests(unittest.TestCase):
    def test_prefix_filter_finds_verified_high_jaccard_copy(self):
        tokens=[f'word{i}' for i in range(100)]
        other=tokens.copy();other[50]='changed'
        rows=[dict(chunk_id='a',source_group_id='ga',split='train',text_content=' '.join(tokens)),
              dict(chunk_id='b',source_group_id='gb',split='test',text_content=' '.join(other)),
              dict(chunk_id='c',source_group_id='gc',split='test',text_content='completely different tourism evidence')]
        matches=v2.near_matches(rows,{'a'},.85)
        self.assertEqual([r['candidate_chunk_id'] for r in matches],['b'])
        self.assertGreater(matches[0]['token_5gram_jaccard'],.90)

    def test_source_link_needs_review_and_merges_confirmed_sources(self):
        rows=[dict(chunk_id='a',doc_id='a',canonical_url='https://example.org/a',text_content='one',text_sha256='one'),
              dict(chunk_id='b',doc_id='b',canonical_url='https://example.org/b',text_content='two',text_sha256='two')]
        link=dict(anchor_chunk_id='a',candidate_chunk_id='b',duplicate_decision='duplicate',
                  annotation_status='TODO')
        with self.assertRaisesRegex(ValueError,'not VERIFIED'):
            v2.group_sources(rows,[link])
        link.update(annotation_status='VERIFIED',reviewer_1_id='fixture-one',reviewer_2_id='fixture-two',
                    reviewer_1_label='duplicate',reviewer_2_label='duplicate')
        v2.group_sources(rows,[link])
        self.assertEqual(rows[0]['source_group_id'],rows[1]['source_group_id'])

    def test_pair_quarantine_retains_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'pairs.json'
            path.write_text(json.dumps([dict(query='q',positive='same',negative='same'),
                                        dict(query='q2',positive='ok',negative='')]))
            original=path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                v2.audit_pairs(argparse.Namespace(train=[path],evaluation=[],out=root/'audit'))
            flags=v2.read_csv(root/'audit/quarantine_review.csv')
            self.assertEqual(len(flags),2)
            self.assertEqual(path.read_bytes(),original)


if __name__=='__main__':
    unittest.main()
