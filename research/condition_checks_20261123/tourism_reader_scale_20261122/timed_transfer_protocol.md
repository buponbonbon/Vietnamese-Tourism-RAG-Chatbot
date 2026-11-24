# Held-out transfer of text-based itinerary verification

Use LA Metro, Montreal STM, and SEPTA GTFS rail. These networks and their
archived service dates, 23 and 29 November 2026, were outside both earlier
route experiments. The work is a schedule-based diagnostic with generated
Vietnamese prompts. Use published snapshot values, without equating them
to actual real-time operating status.

One stable opposing-direction endpoint family per eligible route and date.
Use the natural top-five BM25 ranking from the combined two-date window
corpus. A separate five-candidate control contains one sufficient window,
one opposite direction on the query date, one same-direction window on
the other date, one same-route window without an endpoint, and one other-route
window. Candidate selections use only lexical scores and reference
conditions. Windows represent regular boarding and alighting for both
query endpoints. A reference traversal must use a single active trip with
ordered event times. Candidate labels are frozen before scoring.

Provide only the visible question and candidate text to the verifier.
Compare the raw rank, route and unordered-stop check, that check with
the document date, textual order, and schedule order from the printed
departure/arrival times. Assess distinct renderings of the same source
conditions, including snippets where chronological order differs from
mention order. Quantify demotion of truly sufficient passages and compare
abstaining against demoting when an extractor cannot verify a field.
The partial-extraction control deletes an entire field from the verifier's
view at a fixed rate. The scoring inputs and labels stay fixed.

Retain identical BM25 pools, model scores and reranking cost in every
post-ranking comparison. Report successful top one, rank-five availability,
repairs, regressions and false demotion. Include per-network counts and
three augmented and matched-budget baseline seeds. All scoring uses frozen
checkpoints on CPU and the same visible passage for all models.
