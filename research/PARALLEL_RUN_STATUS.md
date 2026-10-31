# Preparation executed while independent reviewers are being recruited

These are technical preparation and execution checks on draft questions. They are
not a new benchmark result, a full nine-checkpoint reconstruction, or evidence of
a contribution sufficient for any journal quartile.

## Executed results

| Work | Observed result |
|---|---|
| Full draft BM25 mining | 300 queries × top 50 = **15,000 unlabeled candidates**, covering 4,519 distinct chunks. |
| Train-only annotation queue | **2,850 query/candidate pairs** have both the query and candidate in train; no candidate is automatically labeled negative. |
| V2 source-partition audit | 0 source components span partitions; the 225 held-out draft anchors have 0 known legacy train/evaluation exposure flags. This covers observable IDs and exact text, not all semantic reuse. |
| Expanded lexical overlap check | 100 selected anchors against all 56,984 chunks: **0 cross-partition token-five-gram Jaccard matches >= 0.65**. This is not a corpus-wide semantic decontamination certificate. |
| Legacy reuse audit | All three historical train sets still share passage text with VAL/TEST across roles, despite zero query/doc-ID intersections. For TEST, 999 positives occur in train negatives and 793 negatives occur in train positives; these are risk signals, not an estimate of causal bias. |
| Independent annotation pilot | **30 train questions / 10 whole families / 165 passage reviews** per reviewer. Each proposed behavior contributes 10 drafts; these proposals are hidden in Excel and are not validated labels. |
| Real reranker execution | Historical D, seed 42: **150 candidates / 30 train questions**, CPU FP32, max length 512, fixed padding, batch 4. Tokenization/inference took **55.68 s**, excluding retrieval/generation and model load. |
| Candidate preservation | All 150 candidates retained, including 25 negative logits; no raw-logit threshold or quality metric was applied. |
| Controlled comparison plan | 15 data-arm rows + 9 loss-control rows; potentially **21 distinct runs** if the identical B4 unweighted control is shared. This is a blocked plan, not completed training. |

The pilot is selected only from train before inspecting scores. It can be used to
improve annotation instructions and wording without spending held-out evaluation
queries. Two people receive the same questions and passage pairs in independently
shuffled Excel files, with opaque question IDs and blank judgments. The mapping
stays with the coordinator outside the reviewer ZIP. Reviewers should use
`unclear` / `NEEDS_REVIEW` when the small evidence pool is insufficient.

## Artifacts

All paths below are under the ignored `data/research_v2/` directory:

| Directory | Contents |
|---|---|
| `bm25_all_300_top50_001/` | Frozen candidate pool and provenance hashes. |
| `parallel_audit_001/` | Exact legacy reuse audit, V2 exposure/partition counts and train-only candidate annotation queue. |
| `near_duplicate_audit_065_001/` | Expanded lexical check with selected-anchor scope recorded. |
| `pilot30_packets_001/` | Coordinator-facing pilot CSVs and selection manifest. |
| `pilot30_excel_001/` | Two reviewer workbooks, distributable ZIP and private mapping. |
| `pilot30_D42_cpu_001/` | Raw candidate scores and exact inference/model provenance. Coordinator use only; keep model scores out of independent annotation. |
| `experiment_plan_001/` | Provisional comparison matrix and explicit outstanding freeze requirements. |

## What remains pending

- Independent judgments, wording revisions and adjudication; then regenerate
  retrieval pools for changed questions and freeze the reviewed version.
- Dense/RRF and source-neighbor pooling before approving corpus-unanswerability.
  No local BGE-M3 encoder or FAISS corpus index was available for this run; no
  dense retrieval or generated-answer experiment was executed.
- Train-only approved base/addition banks, matched addition count/source mix,
  common optimizer-update budget, pinned initialization and precision. The plan
  deliberately leaves those unavailable values unset. Proposed shared settings
  (including constant margin 1.0) are not claims of tuned optimality.
- An executable controlled trainer and the planned experiments. The three loss
  conditions must explicitly implement unweighted, differentiable weighted and
  detached weighted objectives; detached weighting is not alpha removal.
- Historical B/C checkpoints for a complete reconstruction of the old comparison.

## Verification

27 protocol/preparation tests passed (18 existing + 9 new). New checks reject
candidate hash mismatches, stale query wording, incorrect eligibility flags and
held-out draft inference. They also check same-set blank pilot packets, whole-family
selection, preservation of negative logits and the deliberately blocked plan.
The Excel exporter independently checked row counts, blank judgments, opaque IDs,
dropdowns, absence of hidden labels and ZIP integrity. Historical data and model
files are inputs only; no V2 training or environment publication occurred.
