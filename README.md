# Machine Learning and User Reactions

This project studies how emotion scores from a text classifier relate to
vulgarity, comment length, subreddit, and disagreement among human raters.
Results describe associations in the GoEmotions dataset; they do not establish
causes or reveal what an author actually felt.

## Project status

The project structure and available dataset files are in place. The analysis
pipeline is not implemented yet. Follow `project_whiteboard.md` one milestone
at a time; do not run hypothesis tests until `docs/predictions.md` has been
written by the researcher and committed.

## Data

- Raw GoEmotions CSVs: `data/raw/`
- Simplified train/dev/test data and `emotions.txt`: `data/simplified/`
- Collapsed data: `data/processed/`
- Human-panel data: `data/validation/` (keep private and anonymous)

The Ekman and sentiment mapping files are not currently present in the
workspace. Add verified copies to `data/simplified/` before relying on those
groupings; do not infer or recreate them from memory.

## Setup and run

Use Python 3.10 or later. `requirements.txt` pins compatible dependencies for
Python 3.10–3.13 and Python 3.14+. Install them with
`python -m pip install -r requirements.txt`. The end-to-end entry point will be
`main.py` once the pipeline milestones are implemented.

See `project_whiteboard.md` for the source of truth, milestones, checks, and
ethics requirements.
