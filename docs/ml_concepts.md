# Core ML concepts — study guide

Use these explanations to study, then write your own short explanation in
`docs/log.md` as required by the whiteboard.

- **Train/dev/test:** Fit the model on the training data, use the development
  data to choose settings, and reserve the test data for the final evaluation.
- **TF-IDF:** Convert text into word-based numerical features. A word gets more
  weight when it is frequent in a comment but uncommon across the collection.
- **Logistic regression:** Learn weights for those text features and turn them
  into estimated probabilities. In this project, one-vs-rest models handle the
  multiple emotion labels.
- **Precision:** Of the labels predicted as present, the share that is correct.
- **Recall:** Of the labels that are truly present, the share the model finds.
- **F1:** The harmonic mean of precision and recall. Macro-F1 computes F1 per
  emotion and averages the emotions equally, so rare labels matter too.

## My explanation

Replace this section with your own explanation after reviewing the concepts.
