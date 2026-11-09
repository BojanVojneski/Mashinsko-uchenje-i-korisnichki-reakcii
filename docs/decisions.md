# Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-11-09 | Drop raw `author` by default; only create `author_hash` when an explicit salt is supplied. | Avoid retaining usernames or producing reversible unsalted hashes. |
| 2026-11-09 | Use a small starter profanity lexicon with exact normalized-word matching. | Keep the metric transparent and reproducible; review the list and 20 hand checks before analysis. |
| 2026-11-09 | Add neutral as its own group when evaluating Ekman or sentiment mappings. | The official mapping files omit neutral; keeping it separate lets grouped evaluation cover all 28 labels. Verify this choice before evaluating the real test split. |
