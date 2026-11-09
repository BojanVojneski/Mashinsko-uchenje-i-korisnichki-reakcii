"""Generate project figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_COLUMNS = ("pol", "sub", "dol", "vulg", "vulg_any")
COLOR = "#4C78A8"


def _output_directory(output_dir: str | Path | None) -> Path:
    directory = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "outputs" / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


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

    destination_dir = _output_directory(output_dir)
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


def plot_h1_profanity(
    results: dict[str, pd.DataFrame],
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Save H1 vulgarity-rate and emotion-correlation figures."""
    required = {"group_summary", "spearman"}
    if not required <= set(results):
        raise ValueError(f"H1 results are missing: {sorted(required - set(results))}")
    groups = results["group_summary"]
    correlations = results["spearman"]
    if groups.empty or correlations.empty:
        raise ValueError("H1 results must not be empty")
    if not {"grupa", "vulg_any_rate"} <= set(groups.columns):
        raise ValueError("H1 group summary lacks vulgarity rates")
    if not {"emotion", "rho"} <= set(correlations.columns):
        raise ValueError("H1 correlations lack emotion coefficients")

    destination_dir = _output_directory(output_dir)
    rate_path = destination_dir / "fig02_vulgarity_by_group.png"
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(groups["grupa"].astype(str), groups["vulg_any_rate"], color=COLOR)
    axis.set_title("Comments with a lexicon match by emotion group")
    axis.set_xlabel("Dominant emotion group")
    axis.set_ylabel("Share with at least one match")
    axis.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(rate_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    heatmap_path = destination_dir / "fig03_vulgarity_heatmap.png"
    coefficients = pd.to_numeric(correlations["rho"], errors="coerce").to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(max(10, len(correlations) * 0.45), 3.5))
    image = axis.imshow(coefficients[np.newaxis, :], cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(correlations)))
    axis.set_xticklabels(correlations["emotion"].astype(str), rotation=90)
    axis.set_yticks([])
    axis.set_title("Spearman correlation: vulgarity share vs. emotion scores")
    fig.colorbar(image, ax=axis, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"group_rates": rate_path, "emotion_correlations": heatmap_path}


def plot_h2_length(
    frame: pd.DataFrame,
    output_dir: str | Path | None = None,
    minimum_margin: float | None = None,
) -> Path:
    """Save the H2 word-count box plot, optionally filtering low-margin rows."""
    required = {"dol", "grupa"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("H2 plot data must be non-empty and complete")
    data = frame
    if minimum_margin is not None:
        if not 0 <= minimum_margin <= 1 or "razlika" not in frame.columns:
            raise ValueError("A valid minimum_margin and 'razlika' column are required")
        data = frame.loc[frame["razlika"] >= minimum_margin]
    if data.empty:
        raise ValueError("No rows remain for the H2 plot")

    groups = sorted(data["grupa"].unique())
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.boxplot(
        [data.loc[data["grupa"] == group, "dol"].to_numpy() for group in groups],
        tick_labels=[str(group) for group in groups],
    )
    axis.set_title("Comment length by emotion group")
    axis.set_xlabel("Dominant emotion group")
    axis.set_ylabel("Whitespace-separated word count")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    destination = _output_directory(output_dir) / "fig04_length_by_group.png"
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_h3_subreddits(
    results: dict[str, pd.DataFrame],
    output_dir: str | Path | None = None,
    top_n: int = 15,
) -> Path:
    """Save stacked emotion-group shares for the largest eligible subreddits."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    required = {"subreddit_summary", "group_shares"}
    if not required <= set(results):
        raise ValueError(f"H3 results are missing: {sorted(required - set(results))}")
    summary = results["subreddit_summary"]
    shares = results["group_shares"]
    if summary.empty or shares.empty:
        raise ValueError("H3 results must not be empty")
    if not {"subreddit", "n"} <= set(summary.columns) or "subreddit" not in shares.columns:
        raise ValueError("H3 results lack subreddit counts or group shares")

    selected = summary.nlargest(top_n, "n")["subreddit"]
    group_columns = [column for column in shares.columns if column != "subreddit"]
    plot_data = shares.set_index("subreddit").reindex(selected).fillna(0)
    if not group_columns:
        raise ValueError("H3 group shares contain no emotion groups")

    destination_dir = _output_directory(output_dir)
    destination = destination_dir / "fig05_subreddits.png"
    axis = plot_data[group_columns].plot(
        kind="bar",
        stacked=True,
        figsize=(max(10, len(plot_data) * 0.7), 6),
        colormap="tab20",
    )
    axis.set_title("Emotion-group shares in the largest subreddits")
    axis.set_xlabel("Subreddit")
    axis.set_ylabel("Share of comments")
    axis.legend(title="Emotion group", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.tick_params(axis="x", labelrotation=45)
    axis.figure.tight_layout()
    axis.figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(axis.figure)
    return destination


def plot_h4_neutrality(
    frame: pd.DataFrame,
    emotions: tuple[str, ...] | list[str],
    output_dir: str | Path | None = None,
) -> Path:
    """Save paired model/human dominant-emotion counts for H4."""
    human_columns = [f"human_{emotion}" for emotion in emotions]
    required = {*emotions, *human_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"DataFrame is missing emotion columns: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("H4 plot data must be non-empty and complete")

    human = frame[human_columns].copy()
    human.columns = emotions
    counts = pd.DataFrame(
        {
            "model": frame[list(emotions)].idxmax(axis=1).value_counts().reindex(emotions, fill_value=0),
            "human": human.idxmax(axis=1).value_counts().reindex(emotions, fill_value=0),
        }
    )
    axis = counts.plot(kind="bar", figsize=(max(10, len(emotions) * 0.45), 6), color=[COLOR, "#F58518"])
    axis.set_title("Dominant emotion counts: model and human ratings")
    axis.set_xlabel("Emotion")
    axis.set_ylabel("Comments")
    axis.tick_params(axis="x", labelrotation=90)
    axis.legend(title="Source")
    axis.figure.tight_layout()
    destination = _output_directory(output_dir) / "fig01_emotion_dominance.png"
    axis.figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(axis.figure)
    return destination


def plot_h5_disagreement(
    frame: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Save H5 agreement-confidence and per-emotion comparison figures."""
    required = {"correlation", "emotion_summary"}
    if not required <= set(results):
        raise ValueError(f"H5 results are missing: {sorted(required - set(results))}")
    if frame.empty or not {"soglasnost", "uverenost"} <= set(frame.columns):
        raise ValueError("H5 scatter data must contain agreement and confidence")
    emotions = results["emotion_summary"]
    if emotions.empty or not {"emotion", "mean_soglasnost", "model_f1"} <= set(emotions.columns):
        raise ValueError("H5 emotion summary is missing agreement or F1 values")

    destination_dir = _output_directory(output_dir)
    scatter_path = destination_dir / "fig06_agreement_confidence.png"
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.hexbin(
        frame["soglasnost"],
        frame["uverenost"],
        gridsize=30,
        mincnt=1,
        cmap="Blues",
    )
    axis.set_title("Human agreement and model confidence")
    axis.set_xlabel("Original-rater agreement")
    axis.set_ylabel("Model confidence")
    fig.colorbar(image, ax=axis, label="Comments")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    comparison_path = destination_dir / "fig07_f1_vs_agreement.png"
    values = emotions.set_index("emotion")[["model_f1", "mean_soglasnost"]]
    axis = values.plot(
        kind="bar",
        figsize=(max(10, len(values) * 0.45), 6),
        color=[COLOR, "#F58518"],
    )
    axis.set_title("Model F1 and original-rater agreement by emotion")
    axis.set_xlabel("Emotion")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", labelrotation=90)
    axis.legend(["Model F1", "Mean human agreement"])
    axis.figure.tight_layout()
    axis.figure.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(axis.figure)
    return {"agreement_confidence": scatter_path, "f1_vs_agreement": comparison_path}
