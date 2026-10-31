# V2 preparation: first executed batch

For the subsequent full draft mining, train-only 30-question pilot, real CPU
reranker execution and blocked control matrix, see
[PARALLEL_RUN_STATUS.md](PARALLEL_RUN_STATUS.md). The inventory below records the
initial preparation batch; its full 300-question packets remain available.

This is the preparation phase of the research plan, not evidence that novelty,
system-level improvements or journal readiness have been established.

## Completed and executed

- Added `research/benchmark_v2.py`: source grouping/exposure-aware splits, drafting,
  review gates, near-duplicate audit, unlabeled BM25 mining, separate reviewer
  packets and a non-destructive legacy pair-quality ledger.
- Added the prospective experimental protocol and a six-topic literature matrix.
  Bibliographic fields remain unverified; no nearest-paper novelty claim is made.
- Created **300 drafts / 100 source families**: 100 proposed answers, 100 proposed
  corrections and 100 proposed refusals. Each of train/validation/calibration/test
  has 75 questions from 25 source families. These are drafting allocations, not
  final labels, a powered sample, or a 300-question held-out test.
- Grouped 56,984 chunks / 4,122 canonical source URLs into **3,135 components**.
  Components with known legacy training exposure stay in train; known legacy
  evaluation-only exposure is excluded from the fresh partitions.
- Checked the 100 selected anchors against all 56,984 chunks: **0 cross-partition
  matches at token-five-gram Jaccard >= 0.85**. Semantic paraphrases and all possible
  future mined training passages are not covered by that result.
- Mined **200 UNLABELED BM25 candidates for 20 draft questions**. This is an
  annotation pilot, not hybrid-retrieval validation or a negative-labeled dataset.
- Prepared two independent reviewer packets, each with **300 question records
  and 492 distinct question/passage relevance records** (anchors plus mined pilot).
  Automatic proposals, split and other-reviewer labels are hidden.
- Recorded **230 mechanical-quality flags across five legacy datasets**: baseline
  0, C 71, D 124, VAL 27, TEST 8. These counts include the same underlying examples
  appearing in nested training files; they are not 230 distinct source documents.
  Source files are preserved; no replacement clean TEST or training set was exported.

## Verification

- 18 meaningful tests passed on synthetic fixtures. They cover source aliases,
  exact-copy grouping, legacy negative exposure, repeatable builds, immutable
  membership, independent review/adjudication, evidence quotes, partial vs sufficient
  qrels, refusal handling, near-duplicate blocking and blinded packets.
- The real 300-row batch passes structure-only validation.
- The strict CLI **correctly rejects** that batch because its labels are still TODO.
- SHA-256 checks confirm all five original train/VAL/TEST files match the uploaded
  originals. Existing application source and historical results were not changed.
- No V2 model was trained or scored as a confirmatory experiment. No human review,
  new user study, environment publication, or configuration update occurred.

## Current artifacts

| Artifact | Local location |
|---|---|
| Benchmark drafts | `data/research_v2/batch_001/benchmark_draft.csv` |
| Source/split manifest | `data/research_v2/batch_001/corpus_manifest.csv` |
| Multi-passage qrel ledger | `data/research_v2/batch_001/qrels_review.csv` |
| Build provenance | `data/research_v2/batch_001/build_manifest.json` |
| Lexical near-duplicate audit | `data/research_v2/near_duplicate_audit_001/` |
| BM25 pilot | `data/research_v2/bm25_pilot_001/` |
| Reviewer packets (latest) | `data/research_v2/reviewer_packets_002/` |
| Legacy issues for review | `data/research_v2/legacy_quality_audit_001/quarantine_review.csv` |
| Real-data integration check | `data/research_v2/integration_check_001.json` |

Generated material is in the already ignored `data/` directory. It exists in this
workspace; it has not been published or committed as a public benchmark.

## Outstanding prerequisites and next experiments

1. Two real, independent reviewers must rewrite/assess question naturalness,
   classify behavior and grade evidence; disagreements require adjudication.
   The software cannot replace this with invented human identities or labels.
2. Add dense/RRF and source-neighbor pooled relevance review before approving
   corpus-unanswerability. BM25-only review is insufficient for that claim.
3. Resolve semantic source overlap/provenance limitations and finalize sample-size
   justification; freeze a reviewed benchmark/protocol version.
4. Obtain historical run_1/run_2 checkpoints for all nine-model reconstruction.
   The existing run_3 checkpoint is not a substitute for those controls.
5. Complete nearest-literature verification, then run the matched-budget training
   comparisons and the separate evidence-policy/calibration experiment specified
   in the protocol. Keep test outcomes out of tuning.
6. Prepare and conduct the tourism decision study only after the necessary human
   study arrangements and a validated system comparison are available.
