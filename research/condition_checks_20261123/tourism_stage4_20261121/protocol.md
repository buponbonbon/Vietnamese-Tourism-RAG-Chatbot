# Stage 4. Native retrieval and evidence selection on Vietnamese reader questions

Initial design recorded before new model inference on 21 November 2026. The seven earlier Vietnamese-only cases and the original nine-case pilot are previous work. This stage uses new question decisions.

## Sampling frame

- Natural first-person travel questions from independent public reader articles, public comments, discussion threads and public submission archives. Require a locatable author-authored body or comment. Use only the text needed for the travel decision, exact apart from whitespace or removal of an unrelated identifier. If a comment needs context, use the original published title as a distinct line. Preserve question provenance and source date.
- Require that at least one independently published official Vietnamese passage resolves the material scope of the question. Official operators, airport authorities, embassies and service authorities are eligible within their actual remit. Keep clear which operator or regulatory domain is relevant. Exclude individual arrival-time, visa and service guarantees, missing decisive carrier or route context, and date-specific rules without applicable evidence. For generic stable questions, distinguish the verified document snapshot from retrospective advice.
- Aim for 60–100 cases after discovery. Enforce genuine separate decisions, record actual yield and reason for exclusions, and cluster questions from shared articles/threads or repeated policy targets. Do not fill the cohort with rephrased or generated material. Report service group counts and language conditions. Do not infer traveler identity from a public account.
- Freeze the reviewed queries, source snapshots, raw corpus construction and evaluation rules before viewing reranker scores. Record the qrel evidence clauses and unknowns. Reader evidence is a single documented assistant-led source audit.

## Natural retrieval and annotation

- Build one common closed corpus from Vietnamese public operator/authority pages. Retain the material policy sections of the archived pages, not just gold excerpts. Use deterministic passage IDs, exact source text spans, metadata and snapshot hashes. Every question is retrievable over the same corpus. Populate each corpus group with plausible same-operator distractors and other operators' actual passages.
- Rank the complete corpus with Unicode NFC casefold unigram BM25, with symmetric passage/query tokenization. K=5, ties broken by stable passage ID. Record exact rank and audited labels for every top-five slot. Grading is in the context of the query's published decision, operator, time, direction and passenger conditions.
- Availability is presence of at least one sufficient native candidate. Where ambiguity cannot be resolved, retain unknown and report lower/upper counts. Use strict applicable sufficiency. Where alternatives are genuinely sufficient count all, and reuse checked labels for the same case. The common corpus is bounded and selected from accessible public official pages. These rates do not estimate open-web recall.

## Selection

- Rerank the native BM25 top 5 with pretrained BGE and M0/M1 seeds 42, 123 and 999. Compare with native raw BM25. For confidently absent native pools, substitute the verified anchor for native rank 5 without changing the candidate budget. Reuse positive and negative scores across comparisons. Keep oracles separate from native performance.
- Compare a fixed sufficient reference with the most lexically ranked audited insufficient competitor, and evaluate strict top rank in the full reviewed five-passage set. If multiple passages qualify, use any verified sufficient top rank and report K2-to-K5 loss at the query level. Unknown selections carry uncertainty. Record every tie. No training or hyperparameter search on this cohort.
- Report top-five availability, native rank-one sufficiency, rank one conditional on availability and after substitution, and cases that pass the two-passage test but fail in the full pool. Show counts by model, seed, source group and condition. Use thresholds and denominators explicitly. Diversity and source dependence matter for conclusions.

After source discovery and before new checkpoint scoring, add a separate bridge evaluation of the seven prior all-Vietnamese reader cases, using the same native corpus protocol. This comparison measures a new outcome on previously studied questions. It is reported separately from newly sourced cases and is excluded from the independent sample count. Native source judgement remains separated from all checkpoint outputs.
