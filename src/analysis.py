"""Statistical analyses for hypotheses H1-H5."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, kruskal, spearmanr


def _analysis_frame(
    frame: pd.DataFrame,
    required: Sequence[str],
    minimum_margin: float | None = None,
) -> pd.DataFrame:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Analysis data is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cannot analyze an empty DataFrame")
    if frame[list(required)].isna().any().any():
        raise ValueError("Analysis data contains missing values")

    result = frame
    if minimum_margin is not None:
        if not 0 <= minimum_margin <= 1:
            raise ValueError("minimum_margin must be between 0 and 1")
        if "razlika" not in frame.columns:
            raise ValueError("Margin filtering requires the 'razlika' column")
        margins = frame["razlika"].to_numpy(dtype=float)
        if not np.isfinite(margins).all() or ((margins < 0) | (margins > 1)).any():
            raise ValueError("'razlika' must contain finite values in [0, 1]")
        result = frame.loc[frame["razlika"] >= minimum_margin]
        if result.empty:
            raise ValueError("The margin filter removed every row")
    return result.copy()


def _epsilon_squared(statistic: float, groups: int, observations: int) -> float:
    if observations <= groups:
        raise ValueError("Kruskal-Wallis effect size requires more observations than groups")
    return max(0.0, (statistic - groups + 1) / (observations - groups))


def _cramers_v(statistic: float, table: pd.DataFrame) -> float:
    rows, columns = table.shape
    if min(rows, columns) < 2:
        raise ValueError("Cramer's V requires at least two observed categories per variable")
    return float(np.sqrt(statistic / (table.to_numpy().sum() * min(rows - 1, columns - 1))))


def _kruskal_result(values: pd.Series, groups: pd.Series) -> dict[str, float | int]:
    samples = [values[groups == group].to_numpy() for group in sorted(groups.unique())]
    if len(samples) < 2:
        raise ValueError("Kruskal-Wallis requires at least two observed groups")
    statistic, p_value = kruskal(*samples)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "epsilon_squared": _epsilon_squared(float(statistic), len(samples), len(values)),
        "n": int(len(values)),
        "groups": len(samples),
    }


def h1_profanity(
    frame: pd.DataFrame,
    emotions: Sequence[str],
    minimum_margin: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Test vulgarity differences by emotion group and correlations by emotion."""
    emotion_names = list(emotions)
    if not emotion_names or len(emotion_names) != len(set(emotion_names)):
        raise ValueError("Emotion columns must be non-empty and unique")
    data = _analysis_frame(
        frame,
        ["vulg", "vulg_any", "grupa", *emotion_names],
        minimum_margin,
    )
    if not data["vulg_any"].isin([0, 1]).all():
        raise ValueError("'vulg_any' must contain only 0 and 1")
    vulg = data["vulg"].to_numpy(dtype=float)
    if not np.isfinite(vulg).all() or ((vulg < 0) | (vulg > 1)).any():
        raise ValueError("'vulg' must contain finite values in [0, 1]")

    kw = _kruskal_result(data["vulg"], data["grupa"])
    contingency = pd.crosstab(data["grupa"], data["vulg_any"])
    if min(contingency.shape) < 2:
        raise ValueError("Chi-square requires both vulgarity outcomes and multiple groups")
    chi_square, chi_p, _, _ = chi2_contingency(contingency)
    tests = pd.DataFrame(
        [
            {
                "test": "kruskal_wallis_vulg_by_group",
                "statistic": kw["statistic"],
                "p_value": kw["p_value"],
                "effect_name": "epsilon_squared",
                "effect_size": kw["epsilon_squared"],
                "n": kw["n"],
            },
            {
                "test": "chi_square_vulg_any_by_group",
                "statistic": float(chi_square),
                "p_value": float(chi_p),
                "effect_name": "cramers_v",
                "effect_size": _cramers_v(float(chi_square), contingency),
                "n": int(contingency.to_numpy().sum()),
            },
        ]
    )

    correlations = []
    for emotion in emotion_names:
        if data["vulg"].nunique() < 2 or data[emotion].nunique() < 2:
            correlations.append(
                {
                    "emotion": emotion,
                    "rho": pd.NA,
                    "p_value": pd.NA,
                    "n": len(data),
                    "status": "undefined: constant input",
                }
            )
            continue
        correlation, p_value = spearmanr(data["vulg"], data[emotion])
        correlations.append(
            {
                "emotion": emotion,
                "rho": float(correlation),
                "p_value": float(p_value),
                "n": len(data),
                "status": "ok",
            }
        )

    group_summary = (
        data.groupby("grupa", observed=True)
        .agg(
            n=("vulg", "size"),
            mean_vulg=("vulg", "mean"),
            vulg_any_rate=("vulg_any", "mean"),
        )
        .reset_index()
    )
    return {
        "tests": tests,
        "group_summary": group_summary,
        "spearman": pd.DataFrame(correlations),
    }


def h2_length(
    frame: pd.DataFrame,
    minimum_margin: float | None = None,
    include_subreddit: bool = False,
) -> dict[str, pd.DataFrame]:
    """Compare comment length across groups, optionally controlling for subreddit."""
    required = ["dol", "grupa"]
    if include_subreddit:
        required.append("subreddit")
    data = _analysis_frame(frame, required, minimum_margin)
    lengths = data["dol"].to_numpy(dtype=float)
    if not np.isfinite(lengths).all() or (lengths < 0).any():
        raise ValueError("'dol' must contain finite non-negative word counts")

    kw = _kruskal_result(data["dol"], data["grupa"])
    tests = pd.DataFrame(
        [
            {
                "test": "kruskal_wallis_length_by_group",
                "statistic": kw["statistic"],
                "p_value": kw["p_value"],
                "effect_name": "epsilon_squared",
                "effect_size": kw["epsilon_squared"],
                "n": kw["n"],
            }
        ]
    )
    group_summary = (
        data.groupby("grupa", observed=True)
        .agg(
            n=("dol", "size"),
            mean_dol=("dol", "mean"),
            median_dol=("dol", "median"),
        )
        .reset_index()
    )

    regression = pd.DataFrame()
    if include_subreddit:
        if (data["dol"] <= 0).any():
            raise ValueError("Regression of log(dol) requires positive word counts")
        if data["subreddit"].nunique() < 2:
            raise ValueError("Subreddit regression requires at least two subreddits")
        regression_data = data.assign(log_dol=np.log(data["dol"].astype(float)))
        fitted = smf.ols("log_dol ~ C(grupa) + C(subreddit)", data=regression_data).fit()
        intervals = fitted.conf_int()
        regression = pd.DataFrame(
            {
                "term": fitted.params.index,
                "coefficient": fitted.params.to_numpy(),
                "standard_error": fitted.bse.to_numpy(),
                "p_value": fitted.pvalues.to_numpy(),
                "ci_low": intervals[0].to_numpy(),
                "ci_high": intervals[1].to_numpy(),
            }
        )
    return {"tests": tests, "group_summary": group_summary, "regression": regression}


def h3_subreddits(
    frame: pd.DataFrame,
    minimum_comments: int = 200,
) -> dict[str, pd.DataFrame]:
    """Compare emotion groups across sufficiently large subreddits."""
    if minimum_comments < 1:
        raise ValueError("minimum_comments must be positive")
    data = _analysis_frame(frame, ["subreddit", "grupa", "neutral"])
    neutral_scores = data["neutral"].to_numpy(dtype=float)
    if not np.isfinite(neutral_scores).all() or (
        (neutral_scores < 0) | (neutral_scores > 1)
    ).any():
        raise ValueError("'neutral' scores must be finite values in [0, 1]")
    counts = data.groupby("subreddit", observed=True).size()
    selected = data.loc[data["subreddit"].map(counts) >= minimum_comments]
    if selected.empty:
        raise ValueError("No subreddits meet the minimum comment count")

    contingency = pd.crosstab(selected["subreddit"], selected["grupa"])
    if min(contingency.shape) < 2:
        raise ValueError("Chi-square requires multiple subreddits and emotion groups")
    statistic, p_value, _, _ = chi2_contingency(contingency)
    test = pd.DataFrame(
        [
            {
                "test": "chi_square_subreddit_by_group",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "effect_name": "cramers_v",
                "effect_size": _cramers_v(float(statistic), contingency),
                "n": int(contingency.to_numpy().sum()),
            }
        ]
    )
    summary = (
        selected.groupby("subreddit", observed=True)
        .agg(
            n=("grupa", "size"),
            neutral_score_mean=("neutral", "mean"),
        )
        .assign(non_neutral_share=lambda table: 1 - table["neutral_score_mean"])
        .sort_values("non_neutral_share", ascending=False)
        .reset_index()
    )
    group_shares = contingency.div(contingency.sum(axis=1), axis=0).reset_index()
    return {"tests": test, "subreddit_summary": summary, "group_shares": group_shares}


def h4_neutrality(
    frame: pd.DataFrame,
    emotions: Sequence[str],
    neutral: str = "neutral",
) -> pd.DataFrame:
    """Compare neutral dominance and score mass for model and human labels."""
    emotion_names = list(emotions)
    if neutral not in emotion_names:
        raise ValueError(f"Neutral label {neutral!r} is not in the emotion columns")
    model_columns = emotion_names
    human_columns = [f"human_{emotion}" for emotion in emotion_names]
    data = _analysis_frame(frame, [*model_columns, *human_columns])
    results = []
    for source, columns in (("model", model_columns), ("human", human_columns)):
        scores = data[columns].to_numpy(dtype=float)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError(f"{source} emotion scores must be finite values in [0, 1]")
        totals = scores.sum(axis=1)
        if (totals <= 0).any():
            raise ValueError(f"{source} score mass must be positive for every row")
        neutral_index = emotion_names.index(neutral)
        dominant = np.argmax(scores, axis=1)
        results.append(
            {
                "source": source,
                "dominant_neutral_share": float(np.mean(dominant == neutral_index)),
                "neutral_score_mass_share": float(scores[:, neutral_index].sum() / totals.sum()),
                "n": len(data),
            }
        )
    return pd.DataFrame(results)


def h5_disagreement(
    frame: pd.DataFrame,
    model_metrics: pd.DataFrame,
    emotions: Sequence[str],
    minimum_raters: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Relate human agreement to model confidence and per-emotion F1."""
    emotion_names = list(emotions)
    if not emotion_names or len(emotion_names) != len(set(emotion_names)):
        raise ValueError("Emotion columns must be non-empty and unique")
    human_columns = [f"human_{emotion}" for emotion in emotion_names]
    data = _analysis_frame(
        frame,
        ["soglasnost", "uverenost", "subreddit", "n_rater", *human_columns],
    )
    if minimum_raters is not None:
        if minimum_raters < 1:
            raise ValueError("minimum_raters must be positive")
        data = data.loc[data["n_rater"] >= minimum_raters]
        if data.empty:
            raise ValueError("The rater-count filter removed every row")

    for column in ("soglasnost", "uverenost"):
        values = data[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"'{column}' must contain finite values in [0, 1]")
    if (data["n_rater"] < 1).any():
        raise ValueError("'n_rater' must be positive")

    if data["soglasnost"].nunique() < 2 or data["uverenost"].nunique() < 2:
        correlation = pd.DataFrame(
            [
                {
                    "rho": pd.NA,
                    "p_value": pd.NA,
                    "n": len(data),
                    "status": "undefined: constant input",
                }
            ]
        )
    else:
        rho, p_value = spearmanr(data["soglasnost"], data["uverenost"])
        correlation = pd.DataFrame(
            [
                {
                    "rho": float(rho),
                    "p_value": float(p_value),
                    "n": len(data),
                    "status": "ok",
                }
            ]
        )

    human_scores = data[human_columns].copy()
    human_scores.columns = emotion_names
    data = data.assign(human_dominant=human_scores.idxmax(axis=1))
    agreement = (
        data.groupby("human_dominant", observed=True)
        .agg(n=("soglasnost", "size"), mean_soglasnost=("soglasnost", "mean"))
        .rename_axis("emotion")
        .reset_index()
    )

    required_metric_columns = {"level", "label", "f1"}
    if not required_metric_columns <= set(model_metrics.columns):
        raise ValueError("Model metrics must contain level, label, and f1 columns")
    f1 = model_metrics.loc[
        (model_metrics["level"] == "28_emotions")
        & model_metrics["label"].isin(emotion_names),
        ["label", "f1"],
    ].rename(columns={"label": "emotion", "f1": "model_f1"})
    if f1["emotion"].duplicated().any() or set(f1["emotion"]) != set(emotion_names):
        raise ValueError("Model metrics must have exactly one 28-emotion F1 per emotion")
    emotion_summary = (
        pd.DataFrame({"emotion": emotion_names})
        .merge(agreement, on="emotion", how="left", validate="one_to_one")
        .merge(f1, on="emotion", how="left", validate="one_to_one")
    )
    if not emotion_summary["model_f1"].between(0, 1).all():
        raise ValueError("Per-emotion F1 values must be between 0 and 1")
    emotion_summary["n"] = emotion_summary["n"].fillna(0).astype(int)

    subreddit_summary = (
        data.groupby("subreddit", observed=True)
        .agg(
            n=("soglasnost", "size"),
            mean_soglasnost=("soglasnost", "mean"),
            mean_uverenost=("uverenost", "mean"),
            mean_n_rater=("n_rater", "mean"),
        )
        .sort_values("mean_soglasnost")
        .reset_index()
    )
    rater_summary = (
        data.groupby("n_rater", observed=True)
        .agg(
            n=("soglasnost", "size"),
            mean_soglasnost=("soglasnost", "mean"),
            mean_uverenost=("uverenost", "mean"),
        )
        .reset_index()
    )
    return {
        "correlation": correlation,
        "emotion_summary": emotion_summary,
        "subreddit_summary": subreddit_summary,
        "rater_summary": rater_summary,
    }
