# V2 research workflow

Run from `/workspace/Vietnamese-Tourism-RAG-Chatbot` with `.venv/bin/python`.
Read [PROTOCOL_V2.md](PROTOCOL_V2.md) before finalizing annotations or training.
No automatic step supplies validated human labels or a novelty/acceptance claim.

The build requires the original `train_moc_13.json`, `cross_encoder_VAL.json` and
`cross_encoder_TEST.json` under `data/processed`, in addition to the corpus CSV.
These pair datasets are local inputs and are not published by this change.
BM25 mining also needs the existing inverted index and `underthesea` dependency.
The protocol tests use only synthetic fixtures and the Python standard library.

```bash
# Freeze corpus/source partitions and create 300 drafts (25 families per split).
.venv/bin/python -m research.benchmark_v2 build --out data/research_v2/batch_001

# Validate structure while labels remain drafts.
.venv/bin/python -m research.benchmark_v2 validate \
  --bundle data/research_v2/batch_001 --allow-draft

# Check selected anchors against the entire corpus for lexical near-duplicates.
.venv/bin/python -m research.benchmark_v2 near-duplicates \
  --bundle data/research_v2/batch_001 --threshold 0.85 \
  --out data/research_v2/near_duplicate_audit_001

# Mine UNLABELED BM25 candidates before including them in reviewer packets.
.venv/bin/python -m research.benchmark_v2 mine \
  --bundle data/research_v2/batch_001 --allow-draft \
  --sample 20 --top-k 10 --out data/research_v2/bm25_pilot_001

# Create separate reviewer packets, hiding automatic behavior proposals and splits.
.venv/bin/python -m research.benchmark_v2 packets \
  --bundle data/research_v2/batch_001 \
  --candidates data/research_v2/bm25_pilot_001/candidate_review.csv \
  --out data/research_v2/reviewer_packets_002

# Retain mechanical-quality issues for review; do not rewrite historical data.
.venv/bin/python -m research.benchmark_v2 audit-pairs \
  --out data/research_v2/legacy_quality_audit_001

# After independent review of benchmark and pooled qrels, require the strict gate.
.venv/bin/python -m research.benchmark_v2 validate \
  --bundle data/research_v2/batch_001

.venv/bin/python -m unittest discover -s tests -p test_benchmark_v2.py -v
```

Existing output directories cause a failure instead of an overwrite. Use a new
version directory for a different seed, source split, membership or corpus. Generated
data lives under the existing ignored `data/` directory; preparation code and the
protocol can be reviewed independently of private annotations and scraped text.
Nothing here publishes an environment or modifies historical checkpoints/TEST.

Bundle files:

| File | Purpose |
|---|---|
| corpus_manifest.csv | Every chunk's canonical source, connected component, partition and known legacy exposure. |
| benchmark_draft.csv | Three proposed behaviors per source family; query wording and labels require review. |
| qrels_review.csv | Editable multi-passage relevance ledger; initial anchor grades are deliberately blank. |
| build_manifest.json | Corpus/input/output hashes, seed, membership identity and limitations. |
| candidate_review.csv (mining output) | Unlabeled candidates with query/candidate partitions and separate review fields. |

`proposed_behavior` is a drafting aid; `target_behavior` is the adjudicated label.
Fill reviewer labels separately and adjudicate disagreement. Use distinct IDs;
the software checks records, not whether real people performed the work. Candidate
grade labels can be transferred into additional qrel rows after review using the
question ID and chunk ID; retain the original reviewer decisions and quotes.
Do not copy `candidate_text` into an arbitrary gold reference without review.

The validator checks source/split and evidence identities, reviewer records,
correction anchors, refusal scope and sufficient qrels. It blocks draft inputs
by default. For completed reviews, it also rechecks token-five-gram Jaccard >=0.85
between selected anchors and other partitions. Unresolved flags block the gate.
Resolve flags with independent reviewers in the near-duplicate CSV, then build a
new bundle using `--source-links path/to/reviewed_links.csv`. Confirmed duplicate
sources are joined; independent decisions are retained with provenance hashes.
This check does not cover semantic paraphrases or all future mined training passages.

`review_gate_passed=true` means those input checks passed; it does not
certify annotation truth, exhaustive relevance coverage, statistical power or novelty.

The initial batch is an annotation pilot (75 questions in each of four partitions),
not a full 300-query held-out test. BM25-only candidates must be supplemented with
dense/RRF and source-neighbor review before final unanswerability labels.

The two reviewer packets hide `proposed_behavior`, `target_behavior`, split and
other reviewers' decisions. Each packet has a questions CSV and a separate
passage-relevance CSV; `--candidates` adds mined passages without importing labels.
They do not implement access controls. Reconcile any
query revisions before final review, then transfer the independent labels into the
bundle without inventing reviewers, gold references or relevance grades.

## Excel files for independent reviewers

The coordinator-facing CSV question IDs retain historic behavior suffixes. For
blind review, distribute the Excel exports: they replace those IDs with opaque
codes, shuffle row order separately for each reviewer, and keep the reverse mapping
outside the reviewer ZIP. No hidden worksheet contains proposed labels.

```bash
# Install only the optional workbook-export dependency in the chosen interpreter.
python -m pip install -r research/requirements-export.txt
python research/export_review_workbooks.py \
  --packets data/research_v2/reviewer_packets_002 \
  --out data/research_v2/excel_exports_002
```

Give each reviewer only their own `.xlsx`. The workbook includes Vietnamese
instructions, question and passage-review sheets, dropdown labels and a progress
sheet. Yellow cells are editable review fields. Completed files can be uploaded
back for reconciliation; the coordinator must retain the private ID mapping.
Existing output directories are not overwritten.

## Work that can run while human review is pending

Executed results and limitations are recorded in
[PARALLEL_RUN_STATUS.md](PARALLEL_RUN_STATUS.md). These commands prepare unlabeled
pools, a train-only pilot and a provisional comparison plan. They do not approve
labels, choose an addition budget or start training.

```bash
.venv/bin/python -m research.benchmark_v2 mine \
  --bundle data/research_v2/batch_001 --allow-draft \
  --sample 300 --top-k 50 --out data/research_v2/bm25_all_300_top50_001

.venv/bin/python -m research.parallel_preflight audit \
  --bundle data/research_v2/batch_001 \
  --candidates data/research_v2/bm25_all_300_top50_001/candidate_review.csv \
  --out data/research_v2/parallel_audit_001

# Same 30 train questions for both reviewers, 10 whole source families.
.venv/bin/python -m research.parallel_preflight pilot \
  --bundle data/research_v2/batch_001 \
  --candidates data/research_v2/bm25_all_300_top50_001/candidate_review.csv \
  --out data/research_v2/pilot30_packets_001

python -m research.export_review_workbooks \
  --packets data/research_v2/pilot30_packets_001 \
  --out data/research_v2/pilot30_excel_001

.venv/bin/python -m research.parallel_preflight plan \
  --bundle data/research_v2/batch_001 --out data/research_v2/experiment_plan_001

# Complete local checkpoint required; no remote download or held-out draft scoring.
.venv/bin/python -m research.score_pilot_pool \
  --bundle data/research_v2/batch_001 \
  --candidates data/research_v2/bm25_all_300_top50_001/candidate_review.csv \
  --pilot-packets data/research_v2/pilot30_packets_001 \
  --model models/run_3/vietnam_tourism_reranker_moc_13_seed_42 \
  --out data/research_v2/pilot30_D42_cpu_001

.venv/bin/python -m unittest discover -s tests -p test_parallel_preflight.py -v
```

The pilot evidence pool uses the anchor and top-5 BM25 candidates, deduplicated per
question. It supports an instruction/quality pilot; it cannot establish that a
question is unanswerable throughout the corpus. Distribute each reviewer only their
own workbook from `pilot30_excel_001`, and keep scores and coordinator manifests
separate. Changed questions need newly mined pools before final relevance review.

## Reconcile returned workbooks

Use the export that matches the returned question and passage IDs. The importer
checks the original workbook hashes, opaque ID mapping and immutable fields;
row order can differ. It reads data without executing workbook formulas and
records both judgments without overwriting the benchmark or assigning VERIFIED.
It requires the optional workbook dependency from `requirements-export.txt`.

```bash
python -m research.reconcile_reviews \
  --first /path/to/Nguoi_tham_dinh_01_completed.xlsx \
  --second /path/to/Nguoi_tham_dinh_02_completed.xlsx \
  --export-dir data/research_v2/excel_exports_002 \
  --bundle data/research_v2/batch_001 \
  --independence user-confirmed \
  --out data/research_v2/reconciliation_002

python -m unittest discover -s tests -p test_reconcile_reviews.py -v
```

Use `--independence user-confirmed` only when the user has actually confirmed two
independent reviewers; otherwise omit it. Matching labels or distinct ID strings
are not a substitute for that provenance. Returned data is still subject to
question-quality, evidence-scope and versioning checks. The executed round is
documented in [REVIEW_ROUND_001.md](REVIEW_ROUND_001.md).
