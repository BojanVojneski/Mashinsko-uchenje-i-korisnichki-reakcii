# Your remaining project tasks

This file gathers researcher-owned work from the whiteboard. None of these
planning or manual-review tasks blocks implementation of the code. Predictions
must still be written and committed before formal hypothesis tests or review
of questionnaire responses. Do not ask AI to invent predictions, human
ratings, questionnaire responses, personal reflections, or interpretations.

## Project-level G0 and research gates

- [ ] **Predictions:** Complete `predictions.md` for H1-H5 and V1-V3. For each,
  write what you expect, why, and what result would change your mind. Commit
  the file before any formal hypothesis test or before viewing questionnaire
  responses. This is not a prerequisite for writing or testing code.
- [ ] **If required for the school project:** Ask your teacher for the grading rubric,
  required report format and length, report language, code-demo expectations,
  and rules about libraries and data. Record the answers in your project
  planning notes. This does not block coding.
- [ ] **If required for the final project:** Replace `[DATE]` for the final
  deadline and defense date at the top of `project_whiteboard.md`.
- [ ] **Questionnaire scope:** Decide whether to include the human-panel
  questionnaire in the final project. If included, confirm school/teacher
  rules for consent, minors, anonymous data, and online recruitment, and get
  approval before distributing it.

## Researcher review before later analyses

- [ ] **20 metric checks:** Complete `metric_sanity_checks.md` yourself using
  20 comment IDs, including sarcasm, short comments, and neutral examples.
  Compare your reading with polarity, subjectivity, and vulgarity metrics.
  Keep potentially identifying comment text out of public docs.
- [ ] **Profanity lexicon:** Review `data/profanity_lexicon.txt`; test 20
  examples and record false positives, misses, and whether the observed
  zero-heavy distribution seems reasonable. The prototype matched 4.8% of its
  2,000 comments; that is a check to review, not a conclusion.
- [ ] **Sample privacy:** Review the private panel sample for personal
  information or extreme content. Record any removed comments without
  publishing identifying text.
- [ ] **GoEmotions paper:** Read the paper and verify or correct the starter
  notes in `goemotions_notes.md`, especially annotation, sampling, quality
  checks, and limitations.
- [ ] **Literature:** Complete `literature_notes.md` in your own words and
  verify each source before citing it.
- [ ] **Learn-back:** Replace the prompt in `ml_concepts.md` with your own
  explanation of the concepts and record your teach-back in `docs/log.md`.
- [ ] **Defense prep:** Write your own two-sentence answers in `defense_qa.md`.
- [ ] **Analysis plan:** Confirm the operational definitions and planned tests
  in `hypotheses.md` before results are examined.
- [ ] **Margin filter:** Choose and record the `razlika` cutoff before the
  planned with/without-filter analyses; the whiteboard does not specify a
  cutoff.
- [ ] **Model error review:** After model comparison, read and tag 30
  misclassified comments and record the error categories.
- [ ] **Model write-up:** After choosing a final model, write the methods,
  benchmark comparison, results table, and error-analysis discussion.
- [ ] **Optional feedback:** Ask a psychology teacher or school counselor to
  review the hypotheses and record any feedback.
- [ ] **Extensions:** Decide whether to attempt any optional extension; it is
  fine to choose none.

## Already completed in code

- [x] Data downloaded and explored.
- [x] The 2,000-comment prototype notebook runs end to end.
- [x] Repository skeleton, requirements, README, ignore rules, and work log
  are in place.

When ready, provide the updated prediction file and any school-project or
questionnaire decisions that apply, along with completed review worksheets
and notes.
