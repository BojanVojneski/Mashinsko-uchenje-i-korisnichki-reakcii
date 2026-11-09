# Machine Learning and User Reactions

This project studies how emotion scores from a text classifier relate to
vulgarity, comment length, subreddit, and disagreement among human raters.
Results describe associations in the GoEmotions dataset; they do not establish
causes or reveal what an author actually felt.

## Project status

Data preparation, text metrics, and baseline model code are implemented. The
P1 exploration notebook and P3 2,000-comment prototype have run. The prototype
uses only a held-out portion of the training split; full model evaluation,
out-of-fold scoring, and hypothesis tests have not been run. Follow
`project_whiteboard.md` one milestone at a time, and do not run hypothesis
tests until the researcher has written and committed `docs/predictions.md`.
Researcher-owned tasks are collected in `docs/user_todo.md`.

## Data

- Raw GoEmotions CSVs: `data/raw/`
- Simplified train/dev/test data and `emotions.txt`: `data/simplified/`
- Collapsed data: `data/processed/`
- Human-panel data: `data/validation/` (keep private and anonymous)

Verified Ekman and sentiment mapping files are in `data/simplified/`.

## Setup and run

Use Python 3.10 or later. `requirements.txt` pins compatible dependencies for
Python 3.10–3.13 and Python 3.14+. Install them with
`python -m pip install -r requirements.txt`. Run `notebooks/explore_data.ipynb`
for P1 and `notebooks/prototype.ipynb` for the training-only P3 prototype;
select the project virtual environment as the notebook kernel. `main.py` is
still a scaffold, not an end-to-end pipeline.

Run the unit tests with `python -m pytest -q`.

See `project_whiteboard.md` for the source of truth, milestones, checks, and
ethics requirements.
