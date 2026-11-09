# Hypotheses and operational definitions

These are transcribed from section 6 of `project_whiteboard.md`. Confirm them
before analysis; do not treat them as results or predictions.

## H1 — Profanity and valence

**Question:** To what extent does vulgarity go with negative emotions, and
does it also appear with high-energy positive emotions?

**Measures:** `vulg`, `vulg_any`, `grupa`, `glavna_emocija`, and the 28 emotion
scores.

**Planned tests:** Kruskal–Wallis of `vulg` across emotion groups; chi-square of
`vulg_any` by group; Spearman correlations of `vulg` with each emotion score.
Report epsilon-squared, Cramer's V, and correlations as appropriate.

## H2 — Text length and emotional disclosure

**Question:** Do comments with negative emotions tend to be longer than
positive or neutral comments?

**Measures:** `dol` and `grupa`, with subreddit considered as a covariate.

**Planned tests:** Kruskal–Wallis by group and, if retained, regression
`log(dol) ~ grupa + subreddit`. Report epsilon-squared for the rank test.

## H3 — Subreddit differences

**Question:** Do subreddit emotion-group distributions differ?

**Measures:** `subreddit`, `grupa`, and the `neutral` score; include only
subreddits with at least 200 comments.

**Planned tests:** Chi-square of subreddit by group with Cramer's V; rank
subreddits by mean non-neutral share.

## H4 — Neutrality baseline

**Question:** How large is the neutral share compared with all other emotions?

**Measures:** Neutral as the dominant model emotion and neutral's share of
total model score mass, compared with the same two measures from human
rater means.

## H5 — Human/model disagreement

**Question:** Where do human raters disagree, and does the model struggle on
the same comments?

**Measures:** Human agreement `soglasnost`, model confidence `uverenost`,
emotion, subreddit, rater count `n_rater`, and per-emotion F1.

**Planned tests:** Spearman correlation of `soglasnost` and `uverenost`; mean
agreement by emotion and subreddit; compare per-emotion F1 with mean human
agreement. Check the `n_rater` caveat.

## Human-panel questions

- **V1:** Is model agreement with the panel majority comparable to a typical
  panel member's agreement and to original-rater agreement?
- **V2:** How closely do the panel and original raters agree at group level,
  and where do they differ?
- **V3:** Does ambiguity replicate, measured by the correlation between panel
  and original-rater agreement across the sampled comments?

## Interpretation limits

Use association language, report effect sizes with p-values, and state that
these labels capture perceived emotion. Finalize the tests before looking at
their results.
