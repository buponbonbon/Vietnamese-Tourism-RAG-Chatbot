"""Reconciliation tests use synthetic reviewers and evidence, never research labels."""
import copy
from pathlib import Path
import tempfile
import unittest

try:
    from openpyxl import Workbook
except ModuleNotFoundError as exc:
    if exc.name == 'openpyxl':
        raise unittest.SkipTest('Install research/requirements-export.txt for workbook tests') from exc
    raise

from research import reconcile_reviews as rr


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {f'Q{i}': f'B{i}' for i in range(3)}
        self.questions = {f'B{i}':dict(family_id=f'F{i}', split='train', query=f'Question {i}') for i in range(3)}
        self.chunks = {f'C{i}':dict(chunk_id=f'C{i}', text_content=f'The fixture passage number {i} contains 20 places.', split='train') for i in range(3)}
        self.reviews = []
        for number in (1, 2):
            questions, evidence = {}, {}
            for i, label in enumerate(('answer', 'correct', 'refuse')):
                row = {c[0]:'' for c in rr.QUESTION_COLUMNS}
                row.update(benchmark_id=f'Q{i}', query=f'Question {i}', behavior_label=label,
                           question_decision='KEEP', reference_answer=f'Fixture response {i}',
                           task_type='other', evidence_search_notes='Synthetic fixture search',
                           reviewer_id=f'fixture-r{number}', review_status='DONE')
                questions[f'Q{i}'] = row
                item = {c[0]:'' for c in rr.QREL_COLUMNS}
                item.update(benchmark_id=f'Q{i}', query=f'Question {i}', chunk_id=f'C{i}',
                            relevance_grade='1' if label=='refuse' else '2',
                            candidate_evidence=self.chunks[f'C{i}']['text_content'],
                            evidence_quote=f'number {i} contains 20 places.', reviewer_id=f'fixture-r{number}',
                            review_status='DONE')
                evidence[(f'Q{i}', f'C{i}')] = item
            self.reviews.append(dict(CAU_HOI=questions, BANG_CHUNG=evidence))

    def analyze(self, provenance='user-confirmed'):
        return rr.analyze(self.reviews, self.mapping, self.questions, self.chunks, provenance)

    def workbook(self, path, review):
        wb = Workbook()
        wb.remove(wb.active)
        for name, columns in [('CAU_HOI',rr.QUESTION_COLUMNS),('BANG_CHUNG',rr.QREL_COLUMNS)]:
            ws = wb.create_sheet(name)
            ws.append([c[1] for c in columns])
            for row in review[name].values():
                ws.append([row[c[0]] for c in columns])
        wb.save(path)

    def test_agreement_corrects_for_chance_and_constant_labels_are_undefined(self):
        self.assertAlmostEqual(rr.agreement(['a','a','b','b'],['a','b','b','b'])['cohen_kappa'],.5)
        self.assertIsNone(rr.agreement(['a'],['a'])['cohen_kappa'])
        self.assertIsNone(rr.agreement([],[])['cohen_kappa'])

    def test_perfect_agreement_does_not_finalize_refusal_scope(self):
        summary, questions, evidence, issues = self.analyze()
        self.assertEqual(summary['categorical_agreement']['behavior']['cohen_kappa'],1.)
        self.assertEqual(summary['issues_by_type']['CORPUS_REFUSAL_NOT_ESTABLISHED'],1)
        self.assertFalse(summary['benchmark_ready'])
        self.assertFalse(summary['human_independence_certified_by_code'])
        self.assertTrue(all(r['annotation_status']=='REVIEW_RECORDED_NOT_FROZEN' for r in questions+evidence))
        self.assertNotIn('INDEPENDENCE_NOT_CONFIRMED',summary['issues_by_type'])

    def test_unconfirmed_provenance_is_kept_separate_from_matching_labels(self):
        summary,_,_,_=self.analyze('unconfirmed')
        self.assertEqual(summary['issues_by_type']['INDEPENDENCE_NOT_CONFIRMED'],3)

    def test_distinct_reviewer_ids_are_required(self):
        for sheet in self.reviews[1].values():
            for row in sheet.values(): row['reviewer_id']='fixture-r1'
        with self.assertRaisesRegex(ValueError,'distinct reviewer'):
            self.analyze()

    def test_disagreement_requires_adjudication_without_overwriting_labels(self):
        self.reviews[1]['CAU_HOI']['Q0']['behavior_label']='correct'
        summary,questions,_,_=self.analyze()
        self.assertEqual(summary['issues_by_type']['QUESTION_ADJUDICATION_REQUIRED'],1)
        row=next(r for r in questions if r['review_id']=='Q0')
        self.assertEqual(row['consensus_behavior'],'')
        self.assertEqual((row['reviewer_1_label'],row['reviewer_2_label']),('answer','correct'))

    def test_literal_quotes_are_checked_against_the_identified_passage(self):
        self.reviews[0]['BANG_CHUNG'][('Q0','C0')]['evidence_quote']='Invented evidence'
        summary,_,_,_=self.analyze()
        self.assertEqual(summary['invalid_quote_checks'],1)
        self.assertEqual(summary['issues_by_type']['QUOTE_NOT_IN_PASSAGE'],1)

    def test_refusal_conflicts_with_grade2(self):
        for review in self.reviews:
            review['BANG_CHUNG'][('Q2','C2')]['relevance_grade']='2'
        summary,_,_,_=self.analyze()
        self.assertEqual(summary['issues_by_type']['REFUSAL_CONFLICTS_WITH_GRADE2'],1)

    def test_ambiguous_revisions_are_not_silently_applied(self):
        for review in self.reviews:
            review['CAU_HOI']['Q0'].update(question_decision='REVISE',query_revision='Thông tin tại vị trí cần xác minh trong đoạn trích là gì?')
        summary,questions,_,_=self.analyze()
        self.assertEqual(summary['issues_by_type']['REVISION_MISSING_REFERENT'],1)
        self.assertEqual(next(q for q in questions if q['review_id']=='Q0')['original_query'],'Question 0')

    def test_changed_readonly_fields_and_membership_are_rejected(self):
        changed=copy.deepcopy(self.reviews[0])
        changed['CAU_HOI']['Q0']['query']='Changed original question'
        with self.assertRaisesRegex(ValueError,'Immutable'):
            rr.match_original(self.reviews[0],changed)
        changed=copy.deepcopy(self.reviews[0]);changed['CAU_HOI'].pop('Q0')
        with self.assertRaisesRegex(ValueError,'membership'):
            rr.match_original(self.reviews[0],changed)

    def test_row_order_does_not_affect_matching(self):
        reordered={sheet:dict(reversed(list(rows.items()))) for sheet,rows in self.reviews[0].items()}
        rr.match_original(self.reviews[0],reordered)

    def test_workbook_reader_rejects_formulas_without_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'review.xlsx'
            self.workbook(path,self.reviews[0])
            loaded=rr.read_workbook(path)
            rr.match_original(self.reviews[0],loaded)
            self.reviews[0]['CAU_HOI']['Q0']['reference_answer']='=1+1'
            self.workbook(path,self.reviews[0])
            with self.assertRaisesRegex(ValueError,'Formula'):
                rr.read_workbook(path)

    def test_neighbor_keyword_matches_stay_unlabeled(self):
        self.questions['B2']['source_url']='https://example.org/source'
        for row in self.chunks.values(): row['canonical_url']='https://example.org/source'
        self.chunks['C1']['text_content']='Học sinh được miễn phí trong ví dụ giả lập.'
        scopes,candidates,summary=rr.audit_refusal_sources(
            [dict(review_id='Q2',benchmark_id='B2',consensus_behavior='refuse')],self.questions,self.chunks)
        self.assertEqual(summary['source_neighbor_passages_checked'],3)
        self.assertEqual(len(candidates),1)
        self.assertEqual(candidates[0]['label_status'],'UNLABELED')
        self.assertEqual(candidates[0]['relevance_grade'],'')
        self.assertFalse(summary['automatic_relabeling'])

    def test_no_keyword_match_is_not_a_verified_absence_label(self):
        self.questions['B2']['source_url']='https://example.org/source'
        for row in self.chunks.values(): row['canonical_url']='https://example.org/source'
        _,candidates,summary=rr.audit_refusal_sources(
            [dict(review_id='Q2',benchmark_id='B2',consensus_behavior='refuse')],self.questions,self.chunks)
        self.assertEqual(candidates,[])
        self.assertFalse(summary['automatic_relabeling'])
        self.assertIn('No keyword match does not establish corpus-wide unanswerability.',summary['limitations'])


if __name__=='__main__':
    unittest.main()
