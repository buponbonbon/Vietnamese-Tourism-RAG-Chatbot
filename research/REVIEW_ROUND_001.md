# Returned review round: recorded, not yet frozen

The two returned workbooks match the original full export: **300 questions and
492 question/passage pairs per reviewer**, rather than the later 30-question pilot.
All original IDs, question text, passages and URLs match after NFC/whitespace
normalization. Each workbook contains a consistent reviewer ID, the IDs differ,
and every review row is marked DONE. The user explicitly confirmed that two people
reviewed independently before discussion. That provenance statement is recorded;
software cannot independently establish the human process.

## Observed agreement and decisions

| Item | Observed result |
|---|---|
| Behavior agreement | 300/300; nominal Cohen kappa = 1.0. |
| Question-quality decision agreement | 300/300; nominal Cohen kappa = 1.0. |
| Passage-grade agreement | 492/492; nominal Cohen kappa = 1.0. |
| Consensus decisions | 134 KEEP, 152 REVISE, 14 REJECT. |
| Recorded behaviors | 100 answer, 100 correct, 100 refuse; these include questions proposed for revision/rejection. |
| Recorded passage grades | 187 grade 0, 105 grade 1, 200 grade 2. |
| Literal quote checks | All grade-1/2 quotes in both files match the identified corpus passages after normalization. This checks text, not semantic sufficiency. |

All supplied free-text responses, revisions and review notes also match after
normalization. This observation is preserved in the audit. The agreement values
are descriptive before adjudication; they do not establish correctness, exhaustive
qrels or journal novelty. There are no categorical disagreements requiring a third
reviewer in the returned files.

## Outstanding work found during reconciliation

- The 152 REVISE questions need final wording and checks that evidence judgments
  still apply. **93** proposed rewrites refer to a missing excerpt/position;
  **59** contain an ellipsis in the proposed assertion. Those are review flags,
  not automatically applied replacements.
- The 14 REJECT proposals must be archived and reflected in a new membership
  version, preserving the original frozen bundle. They belong to seven families.
- All 100 refusal records cite source-level review without a reviewed corpus-wide
  search log. They require broader BM25/dense/RRF/source-neighbor evidence review
  before the corpus-unanswerability claim is frozen.
- A lexical check of **5,020 same-URL source passages** found two additional
  student/benefit keyword passages for one refusal question. Those passages remain
  **unlabeled**; school-pupil benefits and university-student eligibility are not
  automatically equivalent, and keyword co-occurrence is not sufficient evidence.
- 266 distinct questions have at least one recorded work item. The other 34 have
  no flags under these checks; this is not a certificate of benchmark readiness.

## Executed artifacts and validation

Latest coordinator outputs are under the ignored
`data/research_v2/reconciliation_002/` directory:

- `Doi_chieu_va_xu_ly_tham_dinh.xlsx`: matched judgments, Vietnamese work queue and
  additional source passages for review.
- `question_consensus.csv`, `qrel_consensus.csv`, `issues.csv`: both judgments and
  their mapping retained; status `REVIEW_RECORDED_NOT_FROZEN`.
- `review_summary.json`, `output_manifest.json`, `REPORT.md`: agreement, provenance,
  hashes and limitations.
- `inputs/`: byte-preserving copies of the returned workbooks.
- `Ket_qua_doi_chieu_tham_dinh.zip`: coordinator report bundle.

40 tests passed (27 existing preparation/protocol tests + 13 workbook/reconciliation
tests). Checks include chance-corrected agreement, undefined kappa for constant
labels, immutable membership, row-order independence, formula rejection, distinct
reviewer IDs, literal quotes, refusal/grade-2 conflicts, unfinalized revisions and
unlabeled lexical cues. The original bundle and uploaded workbooks remain unchanged.
No final benchmark was exported and no new training was started.

The code and this aggregate report can be reviewed in Git. Returned files,
reviewer identifiers, raw judgments and scraped passages stay in ignored local
artifacts; they are not published by this change.
