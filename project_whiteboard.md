# WHITEBOARD: Project Concept and Full Plan (v8)

## 0. How to Use This Document
* This file is the single source of truth. When a decision changes, update it and add a line to the **Decision Log** (section 19).
* **Pace:** step by step, with no fixed durations. A step is finished only when its **Done when** test passes; don't move on before that. Because AI speeds up the coding, spend the saved effort on understanding, checking, and writing.
* **Final deadline:** `[DATE]` | **Defense date:** `[DATE]`

---

## 1. Official Title and Aim
> **"Machine Learning and User Reactions"**

**Aim in one sentence:** Build an emotion-detection model, score about 58,000 Reddit comments with it, and test how emotion relates to vulgarity, text length, and online community, including how much *humans themselves* agree about what a comment feels like.

**Two audiences, one project**
* *Technical school:* complete pipeline, clean modular code, model comparison, reproducibility.
* *Psychology admissions:* questions rooted in psychology, honest limitations, human validation, careful language.

---

## 2. What Makes This Project Stand Out (simple core, five "amazing" touches)
1. **No data leakage.** Every comment is scored by a model that never trained on it (out-of-fold scoring).
2. **Humans vs. machine (H5).** Using the rater data to ask *where people disagree about emotion* and whether the model struggles in the same places.
3. **Predictions written before results.** You commit your predictions to Git before running any test (a real scientific habit).
4. **Your own human panel study.** A short anonymous questionnaire where several people rate the same 100 comments, giving you a human rating to compare with the model and with the original GoEmotions raters, plus an independent replication of H5.
5. **Honest limitations and careful wording.** Associations, never causes.

Everything else is deliberately simple: one dataset, one baseline model, standard tests, readable charts.

---

## 3. Non-Negotiable Rules
1. **Leakage rule:** a comment is never scored by a model trained on it.
2. **Predictions first:** write `docs/predictions.md` and commit it before running any hypothesis test or looking at any questionnaire response.
3. **Careful language:** write "tends to go with," never "causes" or "proves."
4. **Reproducible:** random seed `42` everywhere, `requirements.txt`, one `main.py` that runs the whole pipeline.
5. **Privacy:** never keep or publish real usernames (see 4.5).
6. **Keep a log:** 3 lines at the end of each session in `docs/log.md` (what I did, what broke, what's next).
7. **Finished beats ambitious:** the core pipeline comes first; extensions only if every step so far is done and checked.
8. **AI is a tool, not the author:** you must be able to explain every line of code and every result in your own words (rules in section 13).
9. **Verify everything AI gives you:** code, numbers, and references (see section 13).

---

## 4. Data

### 4.1 The two versions of GoEmotions
| Version | What it is | Role |
|---|---|---|
| **Raw** (`goemotions_1/2/3.csv`) | One row per *rater per comment* (about 211k rows, about 58k unique comments). Has metadata (`author`, `subreddit`, `created_utc`, ...) and 28 binary emotion columns per rater. | **Master analysis set.** Collapse to one row per comment `id`. |
| **Simplified** (`train.tsv`, `dev.tsv`, `test.tsv`) | 43,410 / 5,426 / 5,427 comments, **no header row**: `text`, comma-separated emotion ids, comment `id`. Filtered by rater agreement. | **Model training, tuning, and official evaluation.** Emotion ids follow the order in `emotions.txt`. |
| `emotions.txt`, `ekman_mapping.json`, `sentiment_mapping.json` | Label names and official groupings (verify exact filenames in the repo). | Source for `grupa` and for coarser evaluation. |

Both versions come from the same comments, so they join on the comment `id`.

### 4.2 Why the 3.7k "filtered-out" comments matter
The simplified version keeps only comments where raters agreed. The roughly 3.7k comments left out are, by definition, the ones humans disagreed about. If you analyze only the simplified comments, agreement is artificially high and **H5 is biased**. So the analysis set is **all ~58k comments**, with the unlabeled ones scored by the average of the fold models (see 5.4).

### 4.3 Preprocessing: raw to one row per comment
* Group by `id`:
  * 28 emotion columns -> **mean across raters** (human soft scores, 0-1)
  * `text`, `subreddit`, `created_utc`, `link_id`, `parent_id` -> first value
  * `n_rater` -> number of raters
  * `author` -> replaced by a hash (see 4.5)
* If the raw file has an `example_very_unclear` flag, count how many comments are flagged and decide (and log) whether to keep them.

### 4.4 Variables (Macedonian terms, Latin script)
* **Text metrics**
  * `pol` Polarity, -1 to 1 (TextBlob)
  * `sub` Subjectivity, 0 to 1 (TextBlob)
  * `dol` Length = word count (`len(text.split())`)
  * `vulg` Vulgarity = share of words found in a profanity lexicon (0-1; document the list)
  * `vulg_any` = 1 if the comment contains at least one lexicon word, else 0
* **Model outputs**
  * 28 emotion scores (probabilities, 0-1)
  * `glavna_emocija` dominant emotion = `idxmax` of the 28 scores
  * `uverenost` model confidence = top score
  * `razlika` margin = top score minus second score
  * `grupa` positive / negative / ambiguous / neutral (from the sentiment mapping)
* **Human-signal variables**
  * `soglasnost` human agreement = highest of the 28 rater-mean columns (1.0 means all raters picked the same emotion; low means raters scattered)
  * `n_rater`

### 4.5 Ethics and privacy
* Replace `author` with a salted hash, or drop it. Never publish usernames.
* In the report, don't quote comments that could identify a person; use short, non-identifying examples or paraphrase.
* Some comments contain offensive language: add a short content note to the report.
* The dataset is released under the Apache 2.0 license (repository). Cite the GoEmotions paper (Demszky et al., 2020).

### 4.6 Final Output Schema (Output CSV)
`text` | `id` | `author_hash` | `subreddit` | `link_id` | `parent_id` | `created_utc` | `n_rater` | `pol` | `sub` | `dol` | `vulg` | `vulg_any` | `admiration` ... `neutral` *(model scores)* | `glavna_emocija` | `uverenost` | `razlika` | `grupa` | `soglasnost`

---

## 5. Model Plan (the technical core)

### 5.1 Baseline
**TF-IDF + one-vs-rest Logistic Regression** (multi-label), via a scikit-learn `Pipeline`. `predict_proba` gives a continuous 0-1 score per emotion.
* Suggested settings: `TfidfVectorizer(sublinear_tf=True, min_df=2)`, `LogisticRegression(max_iter=1000)`.
* Do **not** use `class_weight="balanced"` for the final scoring model: it distorts probabilities. (You may test it for F1 and mention it.)

### 5.2 Tuning (on `dev.tsv` only)
* Small grid: `C` in {0.5, 1, 2, 5}; `ngram_range` in {(1,1), (1,2)}.
* Choose **one decision threshold** (try 0.2 to 0.5) that maximizes macro-F1 on `dev`, to turn probabilities into labels for F1.

### 5.3 Evaluation (on `test.tsv`, once, at the end)
* Macro-F1 and per-emotion F1 at three levels: **28 emotions**, **Ekman grouping**, **sentiment grouping** (a group counts as present if any emotion in it is present).
* **Benchmark:** the GoEmotions paper reports macro-F1 about .46 (28), .64 (Ekman), .69 (sentiment) for fine-tuned BERT. A simple model will probably score lower; explaining why (no context, no sarcasm, rare classes, short text) is part of the project.
* **Comparison model:** one alternative (Linear SVM with calibration, or Naive Bayes). One results table, one paragraph of discussion.
* **Error analysis:** read 30 misclassified comments and tag each (sarcasm / very short / ambiguous / rare emotion / other).

### 5.4 Out-of-fold scoring (the leakage fix)
```
labeled   = comments with agreed labels (~54k, from train+dev+test)
unlabeled = the rest (~3.7k, no agreed label)
folds     = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in folds.split(labeled):
    pipeline.fit(text[train_idx], Y[train_idx])           # TF-IDF is refit inside each fold
    scores_labeled[val_idx]  = pipeline.predict_proba(text[val_idx])
    scores_unlabeled        += pipeline.predict_proba(text_unlabeled) / 5
```
* **Sanity check:** macro-F1 of the out-of-fold scores should be close to the test-set F1. If it is much higher, there is a leakage bug.
* **Simple fallback** (if time runs out): analyze only comments never used for training (dev + test + unlabeled, about 14k).

### 5.5 Optional extension
Fine-tuned DistilBERT on Colab. Only if every core milestone is done early.

---

## 6. Research Questions and Hypotheses (operational definitions)

*All phrased "to what extent." Effect sizes always accompany p-values (with this much data almost everything is "significant"). Rules of thumb for interpretation: Spearman |rho| < .1 negligible, .1-.3 small, .3-.5 medium, > .5 large; epsilon-squared about .01 small, .06 medium, .14 large; Cramer's V about .1 small, .3 medium, .5 large. State that these are conventions.*

### H1 [CORE] Profanity and Valence
* **Question:** To what extent does vulgarity go with negative emotions, and does it also appear with high-energy positive emotions (excitement, amusement)?
* **Psychology link:** swearing as emotional expression (Jay, 2009); valence-arousal model (Russell, 1980).
* **Variables:** `vulg`, `vulg_any` vs. `grupa`, `glavna_emocija`, the 28 scores.
* **Tests:** Kruskal-Wallis of `vulg` across `grupa` (epsilon-squared); chi-square of `vulg_any` by `grupa` (Cramer's V); Spearman of `vulg` with each emotion score.
* **Figure:** bar chart of `vulg_any` rate per emotion group; Spearman heatmap.
* **Caution:** GoEmotions was curated to reduce offensive content, so expect many zeros. Describe findings as "mild vulgarity in a filtered dataset."

### H2 [CORE] Text Length and Emotional Disclosure
* **Question:** Do comments with negative emotions tend to be longer than positive or neutral ones?
* **Psychology link:** emotional disclosure and expressive writing (Pennebaker).
* **Variables:** `dol` vs. `grupa`; control for `subreddit`.
* **Tests:** Kruskal-Wallis (epsilon-squared); regression `log(dol) ~ grupa + subreddit` (statsmodels) to separate emotion effect from community effect.
* **Figure:** box plot of `dol` by `grupa`.
* **Caution:** the dataset consists of short comments (the repository lists a maximum sequence length of 30; check the paper for the exact filtering), so `dol` has a narrow range. Expect small effects and say so.

### H3 [CORE] Subreddit Differences
* **Question:** Do some subreddits skew toward strong emotions while others stay near a neutral baseline?
* **Psychology link:** group polarization and online communities (Moscovici and Zavalloni, 1969; Sunstein, 2002).
* **Variables:** `subreddit` x `grupa` (and `neutral` score).
* **Tests:** chi-square of subreddit by `grupa` (Cramer's V), only subreddits with at least 200 comments; rank subreddits by mean "non-neutral" share.
* **Figure:** stacked bar of emotion groups for the 10-15 largest subreddits.
* **Caution:** subreddit may also reflect topic, not "polarization." Don't claim a mechanism.

### H4 [CORE] Neutrality Baseline
* **Question:** How large is the neutral share compared with all other emotions combined?
* **Psychology link:** emotion regulation and self-presentation online.
* **Tests:** (a) share of comments where `neutral` is dominant; (b) neutral's share of total score mass; both for the **model** and the **human rater means**. Report agreement between the four numbers.
* **Figure:** bar chart of `glavna_emocija` counts, next to the human version.
* **Caution:** `neutral` is one of the classes and the dataset was balanced during curation, so this describes the dataset, not Reddit in general.

### H5 [CORE, signature question] Where Do Humans and the Model Disagree?
* **Question:** Which emotions, subreddits, and kinds of comments make human raters disagree (low `soglasnost`), and does the model struggle on the same comments (low `uverenost`)?
* **Psychology link:** emotion is not always readable from text alone; perceived emotion depends on context and the perceiver.
* **Tests:** Spearman between `soglasnost` and `uverenost`; mean `soglasnost` per emotion and per subreddit; per-emotion F1 next to per-emotion mean `soglasnost`.
* **Figure:** hexbin of `soglasnost` vs. `uverenost`; paired bars of F1 and agreement per emotion.
* **Caution:** `soglasnost` depends on `n_rater` (fewer raters push agreement up). Check by `n_rater` or restrict to comments with the same number of raters.

### V1-V3 [CORE] Human Panel Validation
Three extra questions answered with your own questionnaire; full design in section 8.1.

### Extensions (only if on schedule after M2)
* **E1 Pseudo-objectivity:** do comments with `sub` near 0 still carry negative emotion? (Caution: low `sub` may only mean TextBlob found no opinion words.)
* **E2 Polarity and subjectivity:** Spearman of |`pol`| with `sub`; discuss outliers.
* **E3 Time of day:** emotion mix by hour of `created_utc` (exploratory only; UTC is not local time).

---

## 7. Visualizations (all saved to `outputs/figures/`)
| # | File | Content | Supports |
|---|---|---|---|
| 1 | `fig01_emotion_counts.png` | Count of `glavna_emocija`, model vs. human | H4 |
| 2 | `fig02_vulgarity_by_group.png` | `vulg_any` rate and mean `vulg` per `grupa` | H1 |
| 3 | `fig03_vulgarity_heatmap.png` | Spearman of `vulg` vs. 28 scores | H1 |
| 4 | `fig04_length_by_group.png` | Box plot of `dol` by `grupa` | H2 |
| 5 | `fig05_subreddits.png` | Emotion groups across top subreddits | H3 |
| 6 | `fig06_agreement_confidence.png` | Hexbin `soglasnost` vs. `uverenost` | H5 |
| 7 | `fig07_f1_vs_agreement.png` | Per-emotion F1 next to mean `soglasnost` | H5 |
| 8 | `fig08_model_comparison.png` | Model results table/figure | Model |
| 9 | `fig09_pol_sub_hexbin.png` | Hexbin `pol` vs. `sub` | E1, E2 |
| 10 | `fig10_panel_agreement.png` | Agreement with the panel: model vs. a single human vs. original raters (with confidence intervals) | V1, V2 |
| 11 | `fig11_ambiguity_replication.png` | Panel `soglasnost` vs. original `soglasnost` per comment | V3 |

**Style rules:** one color scheme throughout, title, axis labels in plain language, units, readable font size, and a one-line takeaway under each figure in the report. Use `grupa` (4 groups) in plots; keep the 28 emotions in tables.

---

## 8. Validation and Limitations

### 8.1 Human panel study (questionnaire)
**Idea:** run a short, anonymous questionnaire in which several people rate the same comments. Averaging their votes gives a "human rating" built the same way GoEmotions itself was built, so your panel can be compared directly with the original raters and with the model.

**Questions it answers**
* **V1: Is the model as good as an extra human?** Compare the model's agreement with the panel majority to (a) a single panel member's agreement with the majority of the *other* panel members and (b) the original raters' agreement with the panel majority.
* **V2: Do my raters and the original raters perceive emotion alike?** Group-level agreement between the panel majority and the original rater majority, and where they differ.
* **V3: Does ambiguity replicate?** Spearman correlation between panel `soglasnost` and original `soglasnost` over the 100 comments (an independent replication of the measure used in H5).

**Design**
1. **Sample:** 100 comments, stratified by original-rater group (25 positive, 25 negative, 25 ambiguous, 25 neutral), random within each group, seed 42. Manually remove comments with personal information (names, links) or extreme content, and record what you removed.
2. **Blocks:** split the 100 into **10 blocks of 10 comments**, each as balanced across the four groups as possible (2-3 per group). Each block is its own short form with its own link; a person may answer one or several blocks. Target **at least 5 valid raters per comment** (so at least 50 valid block responses overall), ideally 8 per block. When computing kappa, randomly keep the same number of raters per comment.
3. **Per comment (show only the comment text, nothing else; no subreddit, no model output):**
   * **Required, one answer (multiple choice):** *"How does the writer of this comment seem to feel?"* Positive / Negative / Ambiguous / Neutral / I can't tell. This is the question V1-V3 use.
   * **Optional, several answers (checkboxes):** *"Which emotions do you see? Tick up to 3."* Used descriptively only (too few votes per emotion for tests). Drop it if the form feels too long.
   * Comment order within a block is fixed by the sampling script (seeded shuffle), because Google Forms can't shuffle comments without separating each comment's two questions.
4. **Attention check:** one obvious comment per block (not part of the 100). Exclude respondents who fail it and record how many.
5. **Optional background questions (coarse only):** first language; self-rated English level; where they found the survey (friends / Reddit / Discord / other). No names, emails, or other personal data; collect age only if your school approves.
6. **First page:** purpose, voluntary, anonymous, can stop any time, content note (some comments contain offensive language), contact person, consent checkbox, and an "I am 18 or older" confirmation (public communities include minors; change this only if your school approves).
7. **Instructions to respondents:** answer alone, don't discuss with others, don't look the comment up.
8. **Pilot** with a few people first, fix confusing wording, and exclude pilot answers from the analysis.
9. **Researcher excluded:** you don't count as a rater in the main analysis (you know the study).
10. **Tool:** Google Forms or similar; check that it needs no account from respondents and that its settings do what you expect.
11. **Recruitment in small batches (10 comments per form):** short forms get more completions. Post blocks in psychology-related Discord servers and subreddits, plus your own network.
    * Read each community's rules first; ask moderators before posting in a Discord server; never spam or repost the same block in the same community.
    * Spread each block across several communities (don't give one community a whole block), so a block's results don't just reflect one community's raters.
    * Use a neutral post: what it is, that it is a school research project, short and anonymous, how the answers will be used, a content note, the link. Don't mention the model or your hypotheses (avoids priming).
    * Track the number of valid raters per block and stop a block once it has enough.

**Questionnaire layout (Google Forms; one form per block).** *Multiple choice* = one answer; *Checkboxes* = several answers.
* **Page 1, consent:** two separate required single-option Checkboxes questions: "I am 18 or older" and "I understand this is anonymous and voluntary, and I agree to take part." (Two separate questions force both to be ticked.)
* **Page 2, instructions:** answer alone; some comments contain offensive language; the group definitions below.
  * **Positive:** e.g. joy, gratitude, admiration, amusement.
  * **Negative:** e.g. anger, sadness, disgust, disappointment.
  * **Ambiguous:** feelings that can go either way depending on context: surprise, curiosity, confusion, realization.
  * **Neutral:** no clear emotion; factual or informational.
  * **I can't tell:** the comment is too unclear to judge.
* **Page 3, comments (10 + 1 attention check):** for each comment put its text in the question description and a code in the title (for example `[B3-C07]`), then:
  * Q-a (Multiple choice, required): the group question.
  * Q-b (Checkboxes, optional): "Tick up to 3," using response validation "select at most 3" if available. List the emotions in group order, with no commas in the labels:
    * Positive: admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief
    * Negative: anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness
    * Ambiguous: confusion, curiosity, realization, surprise
    * Neutral: neutral
    * (Verify this grouping against the repository's sentiment mapping file.)
* **Page 4, background (optional):** first language; English level (1-5, multiple choice); where they found the survey.
* **Settings:** don't collect emails and don't require a sign-in.
* **Export:** responses go to a Google Sheet or CSV. Checkbox answers arrive in **one cell, separated by commas**, so split them in pandas (`str.split(", ")`). Keep the code-to-comment-id mapping in `panel_sample.csv`.

**Data handling:** save the export in `data/validation/`; keep it anonymous; keep it out of any public repo unless fully anonymized.

**Analysis (decide in advance; predictions first)**
* **Panel scores per comment:** vote share per group among valid votes, ignoring "I can't tell"; `panel_grupa` = group with the most votes (flag ties); `panel_soglasnost` = highest vote share.
* **Panel reliability:** Fleiss' kappa (equal number of raters per comment) or Krippendorff's alpha (handles unequal numbers). Read the numbers cautiously (kappa bands are conventions).
* **V1-V3:** accuracy and Cohen's kappa at group level, with 95% bootstrap confidence intervals (resample the comments 1,000 times).
* **Descriptives:** group distribution of panel vs. original raters vs. model; read the comments where panel and original raters disagree and discuss them.
* **Subgroup (descriptive only):** native vs. non-native English raters.

**How to read the result**
* Averaging votes is a fine "human rating," but always report agreement as well: a mean can hide disagreement.
* The panel is not ground truth; it is another sample of perceivers (small, convenience sample).
* A model that agrees with the panel as well as a typical person does is "as good as an extra human on this sample," not "understands emotion."

### 8.2 Limitations (write all of these)
* The model predicts *perceived* emotion from text, not what the author felt.
* Labels come from human raters (English speakers from India); rater background can shape labels.
* The model has its own error rate; findings describe what the model sees.
* TextBlob misses sarcasm, slang, and context; `sub` near 0 is not proof of objectivity.
* The profanity lexicon is a simplification (misses creative spellings and context).
* Reddit users are not representative of everyone; no demographic information.
* GoEmotions was curated (offensive content reduced, short comments, emotions balanced), limiting H1, H2, H4.
* Observational data: associations only, no causal claims.
* The human panel is a small, self-selected convenience sample (friends plus people from psychology-interested online communities: probably adults, more psychology-literate than average, mixed English levels), not representative; panel results are exploratory.
* Effect sizes matter more than p-values.

---

## 9. Technical Setup and Code Structure

### 9.1 Tools
Python 3.10+, VS Code or Jupyter, Git + private GitHub repo. Libraries: `pandas`, `numpy`, `scikit-learn`, `textblob`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `joblib`. Pin versions in `requirements.txt`.

### 9.2 Folder structure
```
project/
├── data/
│   ├── raw/                 # goemotions_1/2/3.csv (not committed)
│   ├── simplified/          # train.tsv, dev.tsv, test.tsv, emotions.txt, mappings
│   ├── processed/           # comments.csv (one row per comment)
│   └── validation/          # panel_sample.csv, anonymous questionnaire responses
├── src/
│   ├── data.py              # load, collapse raw, hash authors, join splits
│   ├── metrics.py           # pol, sub, dol, vulg, vulg_any
│   ├── model.py             # build_pipeline, tune, evaluate, out_of_fold_scores
│   ├── analysis.py          # one function per hypothesis (H1..H5)
│   ├── panel.py             # sample, clean responses, panel scores, agreement stats
│   └── plots.py             # one function per figure
├── tests/                   # a few simple asserts (data shapes, metric ranges)
├── notebooks/               # exploration only
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── final_scored_comments.csv
├── docs/
│   ├── log.md  hypotheses.md  predictions.md  decisions.md  ai_use.md  defense_qa.md
├── report/
├── main.py                  # runs: data -> metrics -> model -> scoring -> analysis -> plots
├── requirements.txt
├── README.md
└── .gitignore               # data/raw, big files
```

### 9.3 Function list (keeps code short and modular)
* `data.py`: `load_raw()`, `collapse_raw(df)`, `hash_authors(df)`, `load_split(path)`, `labels_to_matrix(ids)`, `build_master()`
* `metrics.py`: `add_pol_sub(df)`, `add_dol(df)`, `add_vulg(df, lexicon)`
* `model.py`: `build_pipeline(C, ngram)`, `tune(dev)`, `evaluate(model, test, level)`, `out_of_fold_scores(master)`
* `analysis.py`: `h1_profanity()`, `h2_length()`, `h3_subreddits()`, `h4_neutrality()`, `h5_agreement()`
* `panel.py`: `make_sample()`, `clean_responses()`, `panel_scores()`, `agreement_stats()`, `bootstrap_ci()`
* `plots.py`: `fig01()` ... `fig11()`

### 9.4 Git habits
Commit at the end of every session with a clear message. Tag milestones: `M1-model`, `M2-results`, `M3-draft`, `M4-final`. Commit `docs/predictions.md` **before** running tests.

---

## 10. Master Plan and Gates
| Phase | Steps | Gate at the end |
|---|---|---|
| **Planning** | P1-P4 | **G0** Go/No-go: everything below checked |
| **Build A: data and model** | B1-B4 | **M1** model report page done |
| **Build B: scoring and analysis** | B5-B8 | **M2** all results and validation done |
| **Build C: writing and defense** | B9-B12 | **M3** full draft (end B10), **M4** submitted (end B12) |

---

## 11. Planning Stage (P1-P4): step by step

### P1: Setup and explore the data
* Install Python, VS Code/Jupyter, Git; create the private repo and folder skeleton.
* Download the raw files, simplified files, `emotions.txt`, and both mapping files into `data/`.
* Load everything with pandas and answer 10 questions in an exploration notebook: row counts; unique comment ids; raters per comment; columns and missing values; top 15 subreddits; length distribution; label counts; share of multi-label comments; how many raw comments are not in the simplified files; any `example_very_unclear` count.
* Read the GoEmotions paper: abstract, taxonomy, data collection, limitations. Write 1 page of notes.
* Learn the basics: train/test split, TF-IDF, logistic regression, precision/recall/F1 (scikit-learn docs, tutorials, or AI explanations). Then explain each concept back in your own words in `docs/log.md`.
* **Done when:** exploration notebook with 10 answers plus 1 page of paper notes.

### P2: Psychology reading and hypotheses
* Read abstracts and introductions of the sources in section 17. For each, write 5 lines: finding, method, relevance, limitation.
* Finalize the operational definitions of H1-H5 (section 6) into `docs/hypotheses.md`.
* Write `docs/predictions.md`: for each hypothesis **and for the panel questions V1-V3 (section 8.1)**, your prediction, why, and what result would change your mind. **Commit it now.**
* Decide which extensions (if any) you would try.
* **Done when:** `hypotheses.md` and `predictions.md` exist and are committed.

### P3: Prototype on 2,000 comments
* Quick-and-dirty end-to-end run on a 2,000-comment sample: collapse raw, compute `pol`/`sub`/`dol`/`vulg`, train the baseline, predict, make one chart.
* Test TextBlob on 20 hand-picked comments (include sarcasm) and keep the examples for the limitations section.
* Choose the profanity lexicon; test it on 20 examples; check how zero-inflated `vulg` is.
* Log every problem and decision.
* **Done when:** prototype notebook runs top to bottom; list of problems and decisions written.

### P4: Finalize the plan and pass gate G0
* **Meet your teacher:** grading rubric, required format and length, report language, deadline, whether a code demo is expected, any rules about libraries and data.
* Optional: ask a psychology teacher or school counselor to read your hypotheses and give feedback.
* **Questionnaire permission:** ask your teacher or school about the rules for surveying people (consent, whether minors may take part, how to handle anonymous data, posting in public online communities) and get an OK before sharing anything.
* Put the real deadline and defense dates at the top of this file.
* Set up the final repo skeleton, `requirements.txt`, README stub, `.gitignore`, and `docs/log.md`.
* **G0 checklist:**
  - [x] Data downloaded and explored
  - [x] Prototype runs end to end
  - [ ] Hypotheses and predictions committed
  - [ ] Rubric known, dates in this file
  - [ ] Questionnaire plan approved by teacher/school
  - [x] Repo skeleton ready
* **Done when:** all G0 boxes are checked.

Code implementation may continue while project-planning items are pending.
Commit researcher-authored predictions before formal hypothesis tests or
viewing questionnaire responses; get approval before collecting panel data.

---

## 12. Building Stage (B1-B12): step by step

### Build A: Data and Model (B1-B4)

**B1: `data.py`**
* Implement `collapse_raw`, `hash_authors`, `load_split`, `labels_to_matrix`, `build_master`.
* Checks (use `assert`): unique ids about 58k; split sizes 43,410 / 5,426 / 5,427; no id appears in two splits; no NaN in emotion means.
* Save `data/processed/comments.csv`.
* Draw the 100-comment panel sample (8.1) with seed 42 and save `data/validation/panel_sample.csv`.
* **Done when:** processed file exists, checks pass, notebook shows basic statistics.

**B2: `metrics.py`**
* Add `pol`, `sub`, `dol`, `vulg`, `vulg_any` for all comments.
* Sanity tests on 20 hand examples; histogram of every metric; count zeros.
* A few asserts in `tests/test_metrics.py` (ranges: pol in [-1,1], sub in [0,1], vulg in [0,1]).
* Build the questionnaire from the panel sample (8.1), pilot it with a few people, and fix confusing parts.
* **Done when:** metrics added, distribution figure saved, tests pass.

**B3: Baseline model**
* Convert id lists to a 28-column matrix; fit TF-IDF on training text only.
* Tune on `dev` (grid in 5.2) and choose the threshold.
* Evaluate once on `test` at all three levels; save per-emotion F1 to `outputs/tables/model_baseline.csv`.
* **Launch the questionnaire** in small blocks of 10 and keep sharing it while you work on the later steps.
* **Done when:** results table saved; one-paragraph interpretation written.

**B4: Comparison model, error analysis, Milestone M1**
* Train the comparison model; make the results table.
* Error analysis: read and tag 30 misclassified comments.
* Choose the final model; log why.
* Write the model part of the methods (about 1 page) while it is fresh.
* **M1 (tag `M1-model`):** one-page model report with table, benchmark comparison, and error categories.

### Build B: Scoring and Analysis (B5-B8)

**B5: Out-of-fold scoring**
* Implement `out_of_fold_scores` (5.4); derive `glavna_emocija`, `uverenost`, `razlika`, `grupa`.
* Checks: all scores in [0,1]; no NaN; out-of-fold macro-F1 close to the test F1 (if much higher, find the leak).
* Save `outputs/final_scored_comments.csv` (schema in 4.6).
* **Done when:** output CSV exists and checks pass.

**B6: H1 and H2**
* Run the tests and make the figures (fig02, fig03, fig04); run with and without the `razlika` filter.
* Write one paragraph per hypothesis: result, effect size, caveat; compare to your prediction (no editing the predictions file).
* **Done when:** two figures, two tables, two paragraphs.

**B7: H3 and H4**
* Run the tests and make fig05 and fig01; compute model and human neutrality numbers.
* One paragraph each, with caveats.
* **Done when:** same as B6.

**B8: H5, human panel analysis, Milestone M2**
* Close the questionnaire once every comment has at least 5 valid raters; clean the responses (attention checks, "I can't tell" votes).
* Run the panel analysis (8.1): panel scores, reliability (kappa or alpha), V1-V3 with bootstrap confidence intervals (fig10, fig11).
* Run H5 on the full data (fig06, fig07); check the `n_rater` caveat.
* Compare all results (H1-H5, V1-V3) with `predictions.md` and write 1 page: "what surprised me."
* **M2 (tag `M2-results`):** all core figures, tables, and validation done.
* **If every step so far is done and checked:** pick one extension (E1, E2, or E3). If not: skip all extensions.

### Build C: Writing, Packaging, Defense (B9-B12)

**B9: Writing 1**
* Draft: abstract placeholder, introduction, psychological background (short paragraph per hypothesis), data, methods (model, metrics, tests), ethics note.
* Build from your existing notes; don't start from a blank page.
* **Done when:** sections 1-4 of the report drafted.

**B10: Writing 2, Milestone M3**
* Draft: results (model, H1-H5, validation), discussion, limitations, conclusion, abstract.
* Finalize all figures with captions and consistent style.
* Give the full draft to your teacher for feedback.
* **M3 (tag `M3-draft`):** complete draft, no empty sections.

**B11: Revise and package**
* Apply teacher feedback; proofread; check every number in the text against the tables.
* Code cleanup: docstrings, remove dead code, consistent names.
* **Reproducibility test:** clone the repo into a fresh folder and run `main.py` from scratch.
* Write the README (how to download data, set up, and run).
* Draft the slides (section 16) and the one-page summary.
* **Done when:** fresh-folder run succeeds and README is complete.

**B12: Defense and buffer, Milestone M4**
* Rehearse the presentation three times (once to a technical person, once to a non-technical one, once timed).
* Write and practice answers in `docs/defense_qa.md`.
* Final proofreading, final export (PDF), tag `M4-final`, submit.
* **M4:** submitted.

---

## 13. Work Routine and Working With AI

### 13.1 Work routine
* **Each work session:** (1) look at the current step and its "Done when" test, (2) work on that one step, (3) write 3 lines in `docs/log.md` and commit.
* **After each step:** tick its "Done when." If it fails, fix it before moving on.
* **If you fall behind:** don't squeeze everything; use the cut rules in section 14.

### 13.2 AI rules (the project will be built with heavy AI help, so these protect you)
1. **Understand before you keep.** For every function AI writes, be able to explain what it does and why. If you can't, ask AI to explain it, then write the explanation yourself in the log.
2. **Verify, never trust.** AI-written ML code often has silent mistakes (data leakage, thresholds tuned on the test set, wrong axis, mixed-up labels). Use the checks in this plan: asserts on data shapes, the out-of-fold sanity check (5.4), hand-checking 20 examples for every metric.
3. **Your own thinking stays yours.** Write the hypotheses, predictions, human labels (8.1), interpretation of results, and limitations yourself. AI may critique them, not author them. The human panel must consist of real people: never use AI to fill in or generate questionnaire responses.
4. **Verify every reference.** AI can invent papers. Confirm each source exists and read at least its abstract before citing it.
5. **Keep an AI log** in `docs/ai_use.md`: what you asked, what you kept, what you changed or rejected.
6. **Check your school's rules on AI** and include a short, honest AI-use statement in the report if required or appropriate.
7. **Spend the time AI saves** on understanding, checking, and writing: that is what the committees grade and what the defense tests.

---

## 14. Risks and Cut Rules

### 14.1 Risk register
| Risk | Likelihood | Plan |
|---|---|---|
| Falling behind | High | Check each step's "Done when"; use the cut rules below |
| Out-of-fold scoring is confusing | Medium | Start from the pseudocode in 5.4; use the simple fallback if stuck |
| Model F1 is much lower than the paper | High (expected) | Not a failure; explain why; it's part of the discussion |
| `vulg` is almost all zeros | High | Use `vulg_any` and rank tests; state the limitation |
| Results are "boring" or unclear | Medium | Report them honestly; small effects are real results |
| Can't install a library | Low | Check in P3; use a simple lexicon list as a fallback |
| Big files slow the laptop | Low | Work on samples while developing; full run only at the end |
| Data loss | Low | Git push after every session |
| Can't explain AI-written code in the defense | High | AI rules in 13.2; explain each function in the log; practice the defense questions |
| AI makes a silent error in the pipeline | Medium | Asserts, hand-checks, and the out-of-fold sanity check |
| Too few questionnaire responses | Medium | Share early and widely; shrink the comment list (not the raters per comment); keep at least 5 raters per comment |
| Panel raters influence each other | Medium | Separate links, ask people not to discuss, and state it as a limitation |
| Posts removed or ignored in communities | Medium | Follow each community's rules, ask moderators first, post in several communities, keep forms short |

### 14.2 Cut rules (in this order)
1. Drop all extensions (E1-E3, DistilBERT).
2. Drop the regression in H2 and the second model's depth (keep a simple comparison).
3. Drop H3.
4. Use the simple fallback for scoring (5.4).
**Never cut:** the core pipeline, H1, H2, H5, the human panel (you may shrink the comment list, not the raters per comment), limitations, the leakage rule.

---

## 15. Report Outline (about 15-20 pages; language and format per your teacher)
1. **Abstract** (150 words)
2. **Introduction:** question, why it matters, what you did (1 page)
3. **Background:** emotion in text, psychology of H1-H5 (1-1.5 pages)
4. **Data:** GoEmotions, raw vs. simplified, preprocessing, privacy
5. **Methods:** metrics, model, out-of-fold scoring, statistics
6. **Results:** model performance; H1-H5; human validation
7. **Discussion:** what the results suggest, what surprised you, predictions vs. results
8. **Limitations**
9. **Conclusion and future work**
10. **References**
11. **Appendix:** full per-emotion F1, lexicon description, extra figures

*Content note: some quoted comments may contain offensive language.*

---

## 16. Defense and Admissions Package

### 16.1 Slides (10-12)
1. Title and question
2. Why emotion in text? (psychology motivation)
3. The data and the leakage problem
4. Pipeline diagram (raw -> metrics -> model -> scores -> tests)
5. Model results and benchmark
6. H1 and H2 results
7. H3 and H4 results
8. H5: humans vs. machine (the signature result)
9. My human panel study and the replication of H5
10. Limitations
11. What I learned and what's next

### 16.2 Defense questions (write 2-sentence answers in `docs/defense_qa.md`)
**Technical**
* Why logistic regression? What does TF-IDF do? What does `predict_proba` return?
* What is F1, and why macro-F1? Why is your score lower than the paper's?
* What is data leakage? What does out-of-fold mean, and why did you use it?
* Why train/dev/test? Why a fixed seed?
* How is `vulg` computed? What are its weaknesses?
* What would you do with more time?
* How did you use AI, what did it get wrong, and what did you check yourself?
* What is kappa (or alpha)? Why report agreement and not just the average of the votes?

**Psychology**
* What does this project say about emotions, and what does it *not* say?
* Why can't you claim that anger *causes* longer comments?
* What is "perceived" vs. "felt" emotion?
* Why did humans disagree on some comments? What does that mean for emotion research?
* Who were your questionnaire raters, and how might that affect your results?
* What would a follow-up study with real participants look like?
* What do Pennebaker's (or Russell's) findings say, in your own words?

### 16.3 Admissions extras
* **One-page summary:** question, method, 3 key findings, 1 limitation, link to the repo (private link or PDF).
* **Motivation-letter hook:** one paragraph on what the human-vs-machine result taught you about emotion and why you want to study psychology.

---

## 17. Reading List (titles from memory; AI can also invent references, so confirm each one exists in a library or Google Scholar before citing)
* Demszky et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions.* ACL.
* Pennebaker & Beall (1986). *Confronting a traumatic event: Toward an understanding of inhibition and disease.* Journal of Abnormal Psychology.
* Pennebaker, Mehl & Niederhoffer (2003). *Psychological aspects of natural language use.* Annual Review of Psychology.
* Russell (1980). *A circumplex model of affect.* Journal of Personality and Social Psychology.
* Jay (2009). *The utility and ubiquity of taboo words.* Perspectives on Psychological Science.
* Moscovici & Zavalloni (1969). *The group as a polarizer of attitudes.* Journal of Personality and Social Psychology.
* Sunstein (2002). *The law of group polarization.* Journal of Political Philosophy.
* Ekman (1992). *An argument for basic emotions.* Cognition and Emotion.
* Fleiss (1971). *Measuring nominal scale agreement among many raters.* Psychological Bulletin.
* Krippendorff (2011). *Computing Krippendorff's alpha-reliability.* (technical note)
* Landis & Koch (1977). *The measurement of observer agreement for categorical data.* Biometrics.
* (For E3 only) Golder & Macy (2011). *Diurnal and seasonal mood vary with work, sleep, and daylength across diverse cultures.* Science.

---

## 18. Final Checklist
- [ ] `predictions.md` committed before any test was run
- [ ] Leakage rule applied; out-of-fold F1 sanity check passed
- [ ] Output CSV matches the schema; no real usernames anywhere
- [ ] Core figures (1-8, 10, 11) done with captions
- [ ] Questionnaire approved, run, and closed with at least 5 valid raters per comment
- [ ] Panel analysis (V1-V3) done with confidence intervals; anonymous data handled correctly
- [ ] Every number in the report matches the tables
- [ ] Fresh-folder reproducibility run succeeded
- [ ] README, requirements, seed in place
- [ ] Slides, one-page summary, defense Q&A ready
- [ ] AI-use log complete; school's AI rules checked; AI-use statement in the report if required
- [ ] Teacher's rubric checked against the finished work

---

## 19. Decision Log (fill in as you go)
| Date | Decision | Why |
|---|---|---|
| | | |

---

> **Note on Implementation:** Keep the Python code clean, modular, and concise. Prioritize simple and effective Pandas and scikit-learn operations without unnecessary complexity or redundant loops. A finished simple project beats an unfinished ambitious one: complete the core pipeline first, then add extensions.
