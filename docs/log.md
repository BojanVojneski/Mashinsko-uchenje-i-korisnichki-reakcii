# Work log

Add three short lines at the end of each work session:

- **Done:**
- **Problems:**
- **Next:**

Researcher-owned reading, prediction, manual-check, and approval tasks are
tracked in `user_todo.md`.

## 2026-11-09

- **Done:** Created the project skeleton, added the verified GoEmotions mapping files, and ran the P1 exploration notebook. The raw files contain 211,225 rater rows and 58,011 unique comments; simplified splits match the expected sizes and have no ID overlap.
- **Also done:** Implemented B1 data preparation, B2 text metrics and histograms, and the B3 TF-IDF/logistic-regression tuning/evaluation functions. Generated the processed table and panel sample; 12 unit tests pass on synthetic fixtures.
- **Also done:** Ran the P3 notebook on 2,000 training-split comments: 1,600 fit / 400 held out, fixed threshold 0.5, macro-F1 0.044, and a prediction-count chart. No dev/test rows or hypothesis tests were used. The starter lexicon matched 4.8% of this prototype sample.
- **Also done:** Implemented the B4 Multinomial Naive Bayes comparison pipeline, dev-set tuning, combined evaluation-table helper, and reproducible misclassification sampler; synthetic tests cover these functions.
- **Also done:** Implemented B5 five-fold out-of-fold predictions for labeled comments and fold-averaged predictions for unlabeled comments, plus dominant emotion, confidence, margin, sentiment group, and human agreement outputs. Synthetic tests only; this has not been run on the full master table.
- **Also done:** Added H1-H5 analysis helpers and the planned figures for H1-H5, including effect sizes, optional `razlika` filtering, subreddit-size filtering, and the `n_rater` summary. These have only been exercised on synthetic data; no real-data hypothesis tests were run.
- **Problems:** The prototype threshold is untuned and its small-sample score is not a final result. The 20-comment manual metric/lexicon reviews, predictions, and any applicable school/questionnaire decisions are in `docs/user_todo.md`. The notebook's first run selected a system kernel without scikit-learn; it now has a project-venv kernel.
- **Next:** Decide later whether to include the optional human-panel module; otherwise continue pipeline orchestration and report-output code. Predictions remain required before formal hypothesis tests or questionnaire-response review, not before implementation.
