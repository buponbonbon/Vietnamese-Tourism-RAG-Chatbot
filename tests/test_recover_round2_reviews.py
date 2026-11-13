"""Recovery must not transfer judgments to a changed question or passage."""
import copy
import unittest

from research.recover_round2_reviews import bind_review, compare


class RecoverReviewsTests(unittest.TestCase):
    def setUp(self):
        self.questions=[dict(benchmark_id='F_answer',query='When is the festival?',
                             anchor_evidence='On 12 June.',source_url='https://example.org/a')]
        self.chunks={'c1':dict(text_content='On 12 June.',url='https://example.org/a')}
        q=dict(benchmark_id='opaque',query='When is the festival?',anchor_evidence='On 12 June.',
               source_url='https://example.org/a',behavior_label='answer',question_decision='KEEP',
               task_type='visit_timing',review_status='DONE',query_revision='',reviewer_id='R1')
        p=dict(benchmark_id='opaque',query=q['query'],chunk_id='c1',candidate_evidence='On 12 June.',
               source_url='https://example.org/a',reviewer_id='R1',review_status='DONE',
               relevance_grade='2',evidence_quote='12 June')
        self.review={'CAU_HOI':{'opaque':q},'BANG_CHUNG':{('opaque','c1'):p}}

    def test_unique_identity_recovers_without_using_opaque_id_pattern(self):
        qs,ps,rid=bind_review(self.review,self.questions,self.chunks)
        self.assertEqual(set(qs),{'F_answer'})
        self.assertEqual(set(ps),{('F_answer','c1')})
        self.assertEqual(qs['F_answer']['original_review_id'],'opaque')
        self.assertEqual(rid,'R1')

    def test_changed_or_ambiguous_query_blocks_transfer(self):
        self.questions[0]['query']='Where is the festival?'
        with self.assertRaisesRegex(ValueError,'unique exact'):
            bind_review(self.review,self.questions,self.chunks)
        self.questions[0]['query']='When is the festival?'
        with self.assertRaisesRegex(ValueError,'unique exact'):
            bind_review(self.review,self.questions*2,self.chunks)

    def test_modified_anchor_source_and_passage_each_block(self):
        for sheet,key,field,value in [
            ('CAU_HOI','opaque','anchor_evidence','On 13 June.'),
            ('CAU_HOI','opaque','source_url','https://example.org/b'),
            ('BANG_CHUNG',('opaque','c1'),'candidate_evidence','On 13 June.'),
            ('BANG_CHUNG',('opaque','c1'),'source_url','https://example.org/b'),
            ('BANG_CHUNG',('opaque','c1'),'query','When is the parade?')]:
            review=copy.deepcopy(self.review);review[sheet][key][field]=value
            with self.subTest(field=field), self.assertRaises(ValueError):
                bind_review(review,self.questions,self.chunks)

    def bound_pair(self):
        other=copy.deepcopy(self.review)
        for sheet in other.values():
            for row in sheet.values():row['reviewer_id']='R2'
        return [bind_review(r,self.questions,self.chunks) for r in (self.review,other)]

    def test_agreement_does_not_finalize_or_certify_independence(self):
        summary,issues=compare(self.bound_pair(),self.questions)
        self.assertEqual(issues,[])
        self.assertEqual(summary['agreement']['relevance_grade']['cohen_kappa'],None)
        self.assertFalse(summary['strict_review_gate_passed'])
        self.assertIn('NOT_ESTABLISHED',summary['independence'])

    def test_disagreement_and_nonliteral_quote_remain_unresolved(self):
        bound=self.bound_pair()
        bound[1][1]['F_answer','c1']['relevance_grade']='1'
        bound[1][1]['F_answer','c1']['evidence_quote']='invented quote'
        summary,issues=compare(bound,self.questions)
        codes={r['issue_type'] for r in issues}
        self.assertIn('DISAGREEMENT_GRADE',codes)
        self.assertIn('QUOTE_NOT_LITERAL',codes)
        self.assertFalse(summary['strict_review_gate_passed'])

    def test_partial_membership_is_not_silently_called_complete(self):
        bound=self.bound_pair()
        questions=self.questions+[dict(benchmark_id='F_correct')]
        summary,issues=compare(bound,questions)
        self.assertEqual(summary['questions_without_returned_review'],1)
        self.assertIn('NO_RETURNED_QUESTION_REVIEW',{r['issue_type'] for r in issues})
        bound[1][1].clear()
        with self.assertRaisesRegex(ValueError,'memberships differ'):
            compare(bound,questions)


if __name__=='__main__':
    unittest.main()
