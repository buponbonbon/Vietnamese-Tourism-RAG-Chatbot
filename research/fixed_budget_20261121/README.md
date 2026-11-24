# Fixed-budget tourism retrieval experiments

This snapshot preserves the code and protocols used for the November 21
tourism, Travel Stack Exchange, and MLQA evaluations. It includes aggregate
results for all three settings. The Vietnamese development analysis uses
single-passage sufficiency judgements. The independent travel forum and
MLQA experiments use accepted-answer and official-paragraph references.

- The tourism development replay compares lexical and dense retrieval,
  including symmetric BM25 at five candidates.
- The travel forum evaluation separates missing references from
  misranking and tests a fixed-size reference substitution.
- The Vietnamese MLQA evaluation checks sensitivity to the lexical baseline.

## Verify the archived results

Extract the verified research artifact ZIP to a local directory. Run the
standard-library verifier against that directory with Python 3:

    python research/fixed_budget_20261121/verify_exported_counts.py /path/to/extracted/artifacts

The verifier checks archive hashes and recalculates the tourism bounds,
forum reference counts and substitution outcomes, and MLQA retrieval and
reranking counts.

## Re-run inference

The experiment directories preserve the verified scripts and protocol files.
A full rerun takes the pinned model and tokenizer files, the original
public source downloads and the recovered local tourism inputs identified
in the protocols. Keep the documented directory layout when replaying the
archived runs. The scripts read the archive and write new outputs alongside
their own protocol. Set the supported local root variables as needed,
including TOURISM_REPO and TOURISM_DENSE_OUTPUT for the MLQA script.
The result verifier above runs independently of inference dependencies.

The Git snapshot contains code, protocols and aggregate results. Keep local
manuscripts, source dumps, raw annotations and model checkpoints in the
local research artifact directories.
