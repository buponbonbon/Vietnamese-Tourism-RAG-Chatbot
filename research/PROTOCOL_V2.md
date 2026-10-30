# Protocol V2 — evidence selection and tourism decisions

Status: executable preparation protocol, **not preregistered, human-validated, or submission-ready**.
This file records prospective choices. Freeze a reviewed version before confirmatory
experiments; changes after looking at outcomes must be logged as exploratory.

## Research question and contributions to test

When does adversarial pairwise training improve evidence selection and decisions
in Vietnamese tourism information tasks, and when does it fail to transfer to
real retrieval candidate pools?

Potential contributions, conditional on literature review and results:

1. A corpus-anchored, human-validated benchmark covering answer, correction of a
   false premise, and insufficient-evidence refusal in tourism tasks.
2. A controlled account of pairwise-to-candidate-set transfer, separating extra
   data volume, negative source, retrieval availability, ranking and calibration.
3. An evidence-based answer/correct/refuse policy, with downstream tourism-task
   validation. A new loss or a larger architecture is not a presumed contribution.

Do not describe this as the first such study or as novel relative to existing work
until the literature matrix is reviewed. A journal quartile is not an acceptance rule.

## Historical results and V2 lineage

- Keep the original train/VAL/TEST, checkpoints, correctness masks and reports.
- Do not replace the original 1,136-row TEST with a cleaned file under the same name.
- Report historical two-candidate metrics and all quality limitations as historical.
- The training loader reads each record's `margin`; changing `default_margin`
  does not override an existing value. Detaching the sigmoid weight does not remove it.
- Preserve the effective margins, per-example weights, exact optimizer-step count,
  seeds, padding/precision and checkpoint hashes for every future run.
- Record **added examples / final training size** and the actual counts. Historic
  moc_6/moc_13 labels do not mean 6.85%/13.7% of the final training set.

## Source identity, exposure and splits

`benchmark_v2.py build` canonicalizes HTTP(S) source URLs: lowercase host, remove
`www`, fragment, trailing slash and known tracking parameters; retain content query
parameters. It joins sources that share a doc_id or an identical NFC/whitespace-
normalized chunk. Connected components stay together. Missing URLs are quarantined.

Known legacy training exposure is recovered from positive **and negative** passage
text and corpus-mappable doc_ids. Exposed components are forced to train. Known
legacy VAL/TEST exposure without train exposure is excluded as `legacy_evaluation`.
Unknown external-pseudo-document provenance remains a limitation; exact matching
does not prove those sources have never been seen.

Remaining components use a seeded hash to assign approximately 60% train, 15%
validation, 10% calibration and 15% test. Fractions are component proportions, not
equal chunk counts. The initial batch selects 25 eligible components per partition,
with three proposed behaviors per family: **300 annotation drafts total**. This is
an annotation pilot, not a powered final sample or a final 300-question TEST.

An optional province holdout assigns entire connected components to test; a
holdout component with known train/evaluation exposure is quarantined. A geography
generalization experiment needs a separately frozen specification; do not choose
the held-out provinces after inspecting model outcomes.

Run a token-shingle near-duplicate audit and review its flagged cross-split pairs.
Exact identity grouping is not a semantic decontamination claim. Confirmed duplicate
sources must be merged into a **new bundle version** before model training and
annotation finalization. Source URLs and all corpus/input hashes belong in releases.
The implemented lexical audit uses exact token-five-gram Jaccard verification at
0.85 for the selected anchors against all corpus chunks; its candidate filter
preserves recall at that threshold. It is not a semantic duplicate detector.
Completed source-link decisions require two reviewers and adjudication when needed;
`build --source-links` records those inputs and joins confirmed duplicate sources.

## Annotation and meaningful tourism tasks

Automatic question templates are drafting aids. They are not real tourist queries.
Rewrite them for naturalness, a single clear information need and an explicit
historical/time scope where appropriate. Do not invent a current price or schedule
from an old news article. Numeric/date perturbations require plausibility and a
human check that the changed claim is actually false in the defined scope.

Choose `task_type` from a documented task taxonomy: destination choice, visit timing,
price/eligibility, access/logistics, or cultural/event information. Review relevance
to real tourism decisions; discard administrative material and source-page boilerplate.

Behavior definitions:

| Label | Required evidence |
|---|---|
| answer | Sufficient corpus evidence to answer a factual question whose premise needs no correction. |
| correct | Evidence establishes a specific false premise; the reference response corrects it and answers only what is supported. |
| refuse | The frozen corpus does not provide sufficient evidence after pooled review. This is corpus-snapshot unanswerability, not a claim that no real-world answer exists. |

The refusal question generator deliberately produces **proposals**, not verified
unanswerable questions. A missing fact in the anchor chunk is insufficient. Search
BM25 and dense/RRF pools plus source-linked neighboring passages, record the search
procedure, and revise the label if sufficient evidence is found. Separate corpus
unanswerability from retrieval failure and model refusal.

Two independent reviewers record labels before adjudication. Use IDs, not invented
identities; do not put sensitive reviewer details in Git. A third reviewer adjudicates
disagreement with a note. Report agreement before adjudication (Cohen's kappa for
two categorical raters, or a justified alternative) and raw disagreement counts.
Automated code can inspect review records, not verify that humans actually reviewed.
Use the separate reviewer packets to hide automatic proposals and the other
reviewer's labels. A mechanical pair-audit ledger is supplied for the legacy data;
it does not export a newly validated training set or replace the historical TEST.

In `benchmark_draft.csv`, fill `target_behavior`, the final `query`, `reference_answer`,
`task_type`, evidence-search notes and reviewer fields. Set `annotation_status` to
`VERIFIED` only after review. For refusals, use `refusal_scope=corpus_snapshot`.

Multiple relevant passages are represented in `qrels_review.csv`:

- 0: unrelated or not useful evidence for this question;
- 1: relevant context but insufficient to support the required factual answer/correction;
- 2: sufficient evidence supporting an answer or correction (possibly one of multiple passages).

Grade 1/2 rows require a literal evidence quote from the identified corpus chunk.
Grade 2 evidence must belong to the question's source partition to support the
held-out-answer-source claim. Partial topical relevance may cross partitions and
must not be mislabeled as zero just to simplify a split. `refuse` may have grades
0/1, but conflicts with grade 2 evidence. Define multi-hop evidence sets explicitly
in a later schema version; this first batch does not claim multi-hop completeness.

The build freezes identities, families and partitions. Human text/label edits are
expected, but dropping families or changing sources requires a new version directory.
Candidate mining does not assign negative labels. Review false negatives before
any training export; never export validation/calibration/test candidates as training data.

## Controlled training comparisons

Use identical architecture, initialization policy, input handling, optimizer,
max-length, precision and **total update budget**. Equal epochs with unequal dataset
sizes are not an equal-step control. Keep batch/accumulation and selection criteria fixed.

| Condition | Additional data, after review |
|---|---|
| B0 | No added examples; clean baseline. |
| B1 | Ordinary examples matched in number to adversarial additions. |
| B2 | Synthetic counterfactual/adversarial negatives. |
| B3 | Natural hard negatives mined from the actual retriever. |
| B4 | A prespecified mixture of synthetic and natural negatives. |

Match addition counts and query/source composition where feasible. Choose the
feasible addition budget after annotation availability is known, using train data
only, and record it before evaluating test. Use seeds 42, 123 and 999 initially;
do not select the favorable seed. A secondary fixed-data experiment can compare
unweighted hinge, the original weighted objective and detached weighting. Explicitly
set/log actual per-record margins. Do not simultaneously change negatives, loss,
epochs and filtering and attribute the result to one component.

Old B/C/D checkpoints are useful historical comparators; they are not substitutes
for newly controlled V2 training. A public unfine-tuned BGE checkpoint is not the
historical B checkpoint. run_1/run_2 remain required to reconstruct all nine old models.

## Evidence availability, ranking, generation and calibration

Use identical retrieval pools across reranker comparisons. Fix the primary pool
size on validation, with prespecified sensitivity sizes (e.g. 10/20/50). Retain
the complete candidate lists, retrieval scores, rank/logit scores and qrels. Compare
BM25, dense and RRF separately; BM25-only pilots do not validate hybrid retrieval.

Report candidate gold availability first; distinguish missing evidence from
reranking failure conditional on available evidence. Report multi-relevance
Recall/Hit, MRR and nDCG with the precise cutoff, graded-label convention and
unjudged-passage policy. These are not transforms of pairwise accuracy on the new
multi-candidate task. Report ordinary and counterfactual subsets separately.

Proposed decision-policy experiment: jointly inspect query relevance and whether
retrieved evidence supports or contradicts the question's premise, then choose
answer/correct/refuse. Relevance logits are not factuality probabilities. Compare
against the same reranker/generator without this policy and a validation-tuned
threshold baseline. Never assume that the raw-logit threshold 0 is calibrated.

The calibration partition is separate from training, model-selection validation
and final test. Fit any temperature/Platt/isotonic calibration and choose thresholds
without test labels. Freeze primary risk definition (unsupported/incorrect released
answers), coverage definition, false-refusal cost and the decision rule before test.
Report risk–coverage, behavior confusion matrix, false refusal and correction success,
with answer quality as judged independently. An always-refuse system must be visibly
penalized. Do not claim distribution-free guarantees without checking their assumptions.

Use the same frozen generator, prompt and decoding settings across conditions.
Keep an NLI/LLM judge as an auxiliary measure and validate it against a blinded
human subset. Count generation/network failures explicitly, not as correct refusals.

## Statistical and tourism-task evidence

All three variants of a family share a source; rows are not independent. Use paired
contrasts and source-component/family-cluster resampling, report effect sizes and
confidence intervals, and prespecify primary vs secondary outcomes and multiplicity
handling. Three seeds do not justify treating every seed×query row as independent.
Select the final annotation/user-study size using a power/precision calculation
based on a pilot; do not promise that 200 or 300 examples establish journal readiness.

For a subsequent user study, use counterbalanced tasks and identical interfaces.
Primary endpoint: correctness of a concrete tourism decision. Secondary endpoints:
source verification, completion time and reliance on wrong answers. Distinguish
appropriate trust from a high subjective trust rating. Prepare consent, recruitment,
data handling and required institutional review before recruiting. No recruitment
or human participation has occurred in this implementation task.

## Literature and claim gates

Complete `literature_matrix.csv` with verified sources covering adversarial/domain
retrieval, false-premise QA, abstention/calibration, tourism RAG and tourism user
decisions. Record the closest papers' actual tasks, datasets, baselines, limitations
and the remaining difference. Foundational BM25/RRF/RAG citations alone do not
establish the gap. Current publisher access is restricted; no fresh literature
search or bibliographic priority claim has been completed here.

Before submission require: reviewed benchmark and near-duplicate/source audit;
controlled training runs; appropriate downstream evaluation; verified comparative
literature; statistical analysis; clear data/code access terms; and claims matching
the observed results. Passing the software annotation gate alone is insufficient.
