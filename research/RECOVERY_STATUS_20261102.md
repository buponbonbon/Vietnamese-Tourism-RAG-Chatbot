# Research recovery status — 2 November 2026 (Asia/Ho_Chi_Minh)

> Historical report recovered from `codex/tourism-final-review-20261101`.
> For the 13 November code/results branch, manuscript sources and archives are
> intentionally excluded and remain local. References below to manuscript
> publication, PDF builds and prior executed checks describe the earlier branch;
> they are not new executions or publication instructions for this branch.

This is a recovery and author-review branch, **not a completed confirmatory study**.
The previous execution environment failed to start with an executor-readiness
timeout. Its local commits and generated outputs were not on GitHub. Their
absence from this snapshot is not proof that earlier work was never performed.
No conclusions are reconstructed from conversation notes alone.

## Verified in the accessible workspace

| Material | Recovery status |
|---|---|
| Repository | Published `main` at `fa306fb3eb5f475b241ba291865dae6d5ee6d716` |
| Manuscript and source notes | Recovered from the preserved 31 October ZIP |
| Corpus | 56,984 chunks; original source CSV retained |
| Original train/VAL/TEST | Retained; hashes and pair defects checked separately |
| Original D, seed 42 | Local checkpoint remains available; no new inference implied |
| Two latest uploaded round-2 files | Re-downloaded and matched to the preserved wording/anchors/URLs |
| Review recovery | 268 questions and 1,688 passage rows per reviewer, no passage-text mismatch |
| Revised benchmark snapshot | 286 draft questions; not frozen for confirmatory evaluation |
| Manuscript citations | 13 published records, including seven from 2026; no preprint-only citations |

The round-2 opaque-ID export mapping is absent. Recovery therefore requires a
unique normalized exact query match and checks the anchor and source URL, then
checks each passage's query, corpus text and source URL. It blocks changed or
ambiguous identities. It does not infer identity from similar wording.
The original workbooks are copied byte-for-byte into a new ignored audit folder.

R01/R02 agree on 268 behavior labels and 1,662/1,688 passage grades. Their raw
question decisions differ on six rows; task categories differ on 36. These
describe the **returned files at this recovery point**. Later assistant edits
and resolutions are unavailable, so these counts must not be presented as newly
discovered errors requiring the user to repeat work. The latest R02 was returned
after feedback; no new blind reliability estimate is claimed.

## Missing from this snapshot

The following were mentioned in prior session notes but cannot presently be
verified from files here: consolidated `batch_003`/`batch_004`, the final
11-question/185-passage confirmation packet, restored B/C/D checkpoints, complete
BGE-M3 index, all-model candidate scores, Qwen generation outputs, and the new
nine-checkpoint CPU replay. Earlier notes about 101 tests are not current test
evidence. Current checks are reported separately after execution.

The supplied Drive URL is already known; no new upload is requested. Recover
the old workspace if it becomes accessible, or re-download checkpoint assets
and resume only computations whose valid complete outputs cannot be found.
Do not overwrite original weights or data. Do not label a partial index complete.
Do not infer generated-answer success from model-server availability.

## Scientific assessment

**Dynamic Loss.** The implemented objective is weighted hinge loss
`max(0, m_i - delta_i) * (1 + sigmoid(-delta_i))`. For an active hinge, writing
`alpha = sigmoid(-delta)`, the differentiable derivative is
`-(1 + alpha) - (m - delta) * alpha * (1 - alpha)`. With the weight detached,
the derivative is `-(1 + alpha)`; unweighted hinge gives `-1`. Thus detaching the
weight does not remove it. A sample's stored margin also overrides the loader's
default. Legacy runs that change only that default are not constant-margin
ablations. These are mathematical/code observations, not evidence of a new loss
or a gain from it. B/C/D also vary data count, margin mix and potentially update
budget; their comparison does not identify an adversarial-hardness effect.

**C-TDQ.** No recoverable definition or implementing source was found. Do not
assign a formula, expansion, result or novelty claim to this acronym.

**Retrieval–Decision Gap.** A useful empirical question is whether supplied-pair
success transfers to selecting sufficient evidence and then answering correctly.
Measure candidate availability, selection conditional on availability, and
generation conditional on sufficient selected evidence on the same reviewed
queries. A decision outcome additionally needs an operational task definition.
The existing two-candidate TEST and 80-query VAL diagnostic use different items
and incomplete qrels; subtracting their metrics does not estimate such a gap.
Prior work already distinguishes relevance from evidence utility (CoRM-RAG) and
examines harms from synthesized negatives (CausalNeg). A renamed decomposition
alone is not a new algorithm.

**Candidate extensions.** Candidate-set/listwise training, retrieval-mined
negative controls and evidence-sufficiency calibration are testable extensions,
not presently demonstrated novel methods. First establish a valid fixed-pool
baseline. Any training comparison must match initialization, data volume,
effective margins and optimizer-update budget; calibrate thresholds on a separate
partition, count unknown qrels as unknown, and use source-group uncertainty.
GPU fine-tuning requires approval; none is launched by this recovery branch.

For IT&T, the defensible direction remains a source-grounded tourism evidence
resource and a carefully controlled application study. Vietnamese tourism RAG
already has published precedent (ViTravelRAG). Audit rigor and multiple runs
improve credibility, but do not alone establish novelty or Q1 readiness. No
acceptance or quartile guarantee is made.

## Continue without inventing completion

1. Preserve this branch and the recovered manuscript/review archive remotely.
2. Recover unavailable source and result files with hash checks; retain any
   genuine completed runs, including unfavorable results.
3. Reconstruct the latest question/label lineage before asking for any further
   targeted human input. Do not repeat the entire annotation round by default.
4. Freeze the supported task/pools and actual human-reviewed versions before
   held-out quality evaluation. Keep train diagnostics separate.
5. Complete same-pool model comparisons and the fixed-generator evaluation,
   then update tables only from verified manifests and score/output files.

Private audit location: `data/research_v2/recovery_20261102_001/round2_audit/`.
`recover_round2_reviews.py` never sets VERIFIED or passes the strict freeze gate.

## Checks executed for this recovery

- 46 software tests passed, including six new exact-identity recovery checks.
- Six original dataset hashes match the protected source hashes. TEST still has
  seven identical positive/negative pairs and one empty negative; nothing was deleted.
- Recomputing means, population SDs and seed-42 exact McNemar probabilities from
  the reported counts agrees with the archived tables. This is an arithmetic
  check, not a checkpoint rerun or validation of original labels.
- All four PDFs compile: numeric and named author-year manuscripts (22 pages
  each), anonymous Springer manuscript (17 pages), and title page (one page).
  Final logs have no undefined references/citations or overfull-box warnings.
  The anonymous PDF text and metadata contain none of the author identities.
- Official Springer and elsarticle dependency archives pass pinned SHA-256
  checks, and the reproducible bootstrap command was executed successfully.
- Public source staging excludes original data, reviewer workbooks, weights,
  generated model outputs and credentials. No fine-tuning or environment
  publication was performed.
