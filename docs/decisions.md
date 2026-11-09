# Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-11-09 | Drop raw `author` by default; only create `author_hash` when an explicit salt is supplied. | Avoid retaining usernames or producing reversible unsalted hashes. |
| 2026-11-09 | Use a small starter profanity lexicon with exact normalized-word matching. | Keep the metric transparent and reproducible; review the list and 20 hand checks before analysis. |
| 2026-11-09 | Add neutral as its own group when evaluating Ekman or sentiment mappings. | The official mapping files omit neutral; keeping it separate lets grouped evaluation cover all 28 labels. Verify this choice before evaluating the real test split. |
| 2026-11-09 | For P3, sample 2,000 rows from simplified `train.tsv`, fit on 1,600, and predict on the other 400. | Keeps the prototype self-contained and avoids using dev/test data; the fixed 0.5 threshold is only a smoke test, not the final tuned model. |
| 2026-11-09 | Use one-vs-rest Multinomial Naive Bayes as the B4 comparison model. | It is the lightweight comparison option named in the whiteboard and provides probabilities for dev-set threshold tuning. |
| 2026-11-09 | Preserve collapsed human emotion means as `human_<emotion>` when B5 writes model probabilities into the original emotion columns. | Keeps both human and model scores available for later comparison analyses. |
| 2026-11-09 | Define H3 mean non-neutral share as `1 - mean(neutral model score)` and group H5 agreement by each comment's human-score dominant emotion. | Makes the summaries explicit and keeps model outputs separate from original-rater means. |
