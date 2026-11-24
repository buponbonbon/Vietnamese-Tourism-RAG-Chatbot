# Condition-sensitive tourism retrieval checks

This directory records the research runs reported in Supplementary Sections
S12–S20. The saved-score verifier for S20 uses the checked-in data and the
Python standard library.

```bash
python research/condition_checks_20261123/tourism_reader_target_20261123/verify_saved_reader_scores.py
```

The verifier reproduces grade-2 Hit@1, failures conditional on availability,
and results after replacing the fifth candidate on the four unavailable
pools. It checks the rankings for all 24 questions and seven checkpoint score
files, plus the BM25 baseline. The cumulative cohort index lists 60 case IDs
across separate source panels and identifies the 50 cases evaluated in original
top-five pools.

The archived logit JSON, result summary and ranking CSV for S20 retain their
original bytes. `reader_pools_s20.json` projects the frozen benchmark onto the
candidate IDs and grades required to verify metrics. `reader_cohort_index.json`
projects the internal source ledger onto case and cohort IDs. The manifest
`reader_public_sha256.json` checks these exports. The original text-based training
overlap check is recorded separately in `reader_verification.json`.

## Archived experiment directories

| Evidence | Directory |
| --- | --- |
| Schedule-derived stop reversal and equal-stop controls (S12) | `travel_constraint_swap_20261121/` |
| Public reader pilot and operator-sourced evidence (S13–S14) | `vietnam_travel_natural_20261121/`, `tourism_vietnamese_operators_20261121/` |
| Native-pool reader evaluation (S15) | `tourism_stage4_20261121/` |
| Additional reader cases and schedule scope checks (S16–S17) | `tourism_stage4_stage5_20261122/` |
| Reader extension and date/direction transfer (S18–S19) | `tourism_reader_scale_20261122/` |
| Archived Vietnamese reader evaluation (S20) | `tourism_reader_target_20261123/` |

The archived inference and corpus scripts use the layout recorded at the time
of the experiments. A full replay takes the original frozen local inputs,
source snapshots, run manifests and matching checkpoints. Some scripts use
absolute paths to those original assets. To replay, copy the archived inputs
and scripts to a separate analysis directory and adapt the asset paths there.
The S20 saved-score command above works directly from a checkout.

Question and passage text, workbooks, model weights and manuscript files are
retained in the original project archives. Source downloads and reuse remain
subject to the sources' terms and the data notes in the
[recovery directory](../recovery_20261113/DATA_TERMS.md).
