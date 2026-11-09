"""Draw the 100-comment panel sample for the questionnaire (plan section 8.1).

Input : data/processed/comments.csv  (one row per comment). Needs the columns
        id, text, and the 28 emotion rater-mean columns named after the emotions.
        Optional (kept in the output if present): soglasnost, n_rater.
Output: data/validation/panel_sample.csv       full sample, KEEP (has orig_group, soglasnost)
        data/validation/panel_forms_input.csv  block, code, text only (upload this to Google Drive)

Usage:  python make_sample.py
        python make_sample.py --exclude id1,id2   # redraw after removing problem comments
"""
import argparse
from pathlib import Path

import pandas as pd

SEED = 42
PER_GROUP = 25
N_BLOCKS = 10

GROUPS = {
    "positive": ["admiration", "amusement", "approval", "caring", "desire", "excitement",
                 "gratitude", "joy", "love", "optimism", "pride", "relief"],
    "negative": ["anger", "annoyance", "disappointment", "disapproval", "disgust",
                 "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"],
    "ambiguous": ["confusion", "curiosity", "realization", "surprise"],
    "neutral": ["neutral"],
}
RISKY = r"http|www\.|@|\bu/|\br/"   # links, mentions, usernames: review these by hand


def draw_sample(df, exclude=()):
    df = df[~df["id"].isin(exclude)].copy()
    mass = pd.DataFrame({g: df[cols].sum(axis=1) for g, cols in GROUPS.items()})
    df["orig_group"] = mass.idxmax(axis=1)          # group with the most rater votes

    parts = []
    for offset, group in zip([0, 5, 0, 5], GROUPS):  # offsets give every block 10 comments
        part = df[df["orig_group"] == group].sample(PER_GROUP, random_state=SEED).reset_index(drop=True)
        part["block"] = (part.index + offset) % N_BLOCKS + 1
        parts.append(part)

    sample = pd.concat(parts).sample(frac=1, random_state=SEED)       # shuffle order
    sample = sample.sort_values("block", kind="stable").reset_index(drop=True)
    sample["code"] = [f"B{b:02d}-C{i:02d}"
                      for b, i in zip(sample["block"], sample.groupby("block").cumcount() + 1)]
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/comments.csv")
    ap.add_argument("--out_dir", default="data/validation")
    ap.add_argument("--exclude", default="", help="comma-separated comment ids to leave out")
    args = ap.parse_args()

    exclude = [x for x in args.exclude.split(",") if x]
    sample = draw_sample(pd.read_csv(args.src), exclude)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keep = [c for c in ["block", "code", "id", "text", "orig_group", "soglasnost", "n_rater"] if c in sample]
    sample[keep].to_csv(out / "panel_sample.csv", index=False)
    sample[["block", "code", "text"]].to_csv(out / "panel_forms_input.csv", index=False)

    print("Comments per block and group:")
    print(pd.crosstab(sample["block"], sample["orig_group"]))
    risky = sample[sample["text"].str.contains(RISKY, case=False, regex=True)]
    print(f"\nReview by hand ({len(risky)} comments with links, mentions or usernames):")
    print(risky[["code", "id", "text"]].to_string(index=False))
    print("\nTo drop a comment, rerun with: --exclude <id>,<id>")


if __name__ == "__main__":
    main()
