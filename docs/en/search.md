# Search Behaviour

*English · [Português (Brasil)](../pt-BR/search.md)*

A search embeds at most **15 s** of query audio. Scoring uses the
best-matching pair of windows, so a long query would allow an arbitrary
fragment to determine the result. Files within the limit are used whole;
longer files require `--start-time` and `--end-time` (or a selection in the
interface), and are refused rather than truncated. Only the selected frames
are decoded.

Filters are applied before scoring, so results are ranked within the filtered
set rather than filtered afterwards. Running the same query again with
different filters reuses the cached embeddings and returns immediately.

## Ranking

Every active model scores every candidate, but the results are ranked by one
model only: Perch v2, unless `search --rank-by`, *Rank by* on the Search page
or `search.rank_by` in the [configuration](configuration.md) names another.
Each model's scores have a baseline and spread of their own, so a 0.6 from one
is not a 0.6 from another, and they are never combined. Instead, each result
also shows the other models' scores and the rank each gave the item: an item
near the top of both rankings is one the models agree on. A minimum score is
read on the ranking model's scale.
