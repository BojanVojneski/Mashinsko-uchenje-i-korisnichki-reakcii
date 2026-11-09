"""Generate project figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_COLUMNS = ("pol", "sub", "dol", "vulg", "vulg_any")
COLOR = "#4C78A8"


def plot_metric_distributions(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> Path:
    """Save histograms for the five text metrics and return the figure path."""
    missing = set(METRIC_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing metric columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Cannot plot metric distributions for an empty DataFrame")
    if df[list(METRIC_COLUMNS)].isna().any().any():
        raise ValueError("Metric columns contain missing values")
    if not df["vulg_any"].isin([0, 1]).all():
        raise ValueError("vulg_any must contain only 0 and 1")

    destination_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "outputs" / "figures"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "metric_distributions.png"

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for column, ax in zip(("pol", "sub", "dol", "vulg"), axes.flat):
        ax.hist(df[column], bins=30, color=COLOR, edgecolor="white")
        ax.set_title(column)
        ax.set_xlabel("Value")
        ax.set_ylabel("Comments")
        ax.grid(axis="y", alpha=0.2)

    counts = df["vulg_any"].value_counts().reindex([0, 1], fill_value=0)
    axes.flat[4].bar(["No", "Yes"], counts.to_numpy(), color=COLOR)
    axes.flat[4].set_title("vulg_any")
    axes.flat[4].set_xlabel("At least one lexicon word")
    axes.flat[4].set_ylabel("Comments")
    axes.flat[4].grid(axis="y", alpha=0.2)
    axes.flat[5].set_visible(False)

    fig.suptitle("Text metric distributions")
    fig.tight_layout()
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destination
