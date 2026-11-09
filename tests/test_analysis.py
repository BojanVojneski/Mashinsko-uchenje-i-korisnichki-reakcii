import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    h1_profanity,
    h2_length,
    h3_subreddits,
    h4_neutrality,
    h5_disagreement,
)
from src.plots import (
    plot_h1_profanity,
    plot_h2_length,
    plot_h3_subreddits,
    plot_h4_neutrality,
    plot_h5_disagreement,
)


EMOTIONS = ["anger", "joy", "neutral"]


def _analysis_data(size=48):
    rows = []
    groups = ["negative", "positive", "ambiguous", "neutral"]
    for index in range(size):
        emotion_scores = {
            "anger": (index % 7) / 6,
            "joy": ((index + 2) % 7) / 6,
            "neutral": ((index + 4) % 7) / 6,
        }
        rows.append(
            {
                **emotion_scores,
                "vulg": (index % 5) / 10,
                "vulg_any": int(index % 5 == 0),
                "grupa": groups[index % len(groups)],
                "dol": index % 9 + 1,
                "subreddit": f"sub-{index % 3}",
                "neutral": emotion_scores["neutral"],
                "razlika": (index % 7) / 7,
            }
        )
    return pd.DataFrame(rows)


def test_h1_returns_effect_sizes_group_summary_and_emotion_correlations():
    results = h1_profanity(_analysis_data(), EMOTIONS, minimum_margin=0.2)

    assert set(results) == {"tests", "group_summary", "spearman"}
    assert set(results["tests"]["effect_name"]) == {"epsilon_squared", "cramers_v"}
    assert results["tests"]["effect_size"].between(0, 1).all()
    assert len(results["spearman"]) == len(EMOTIONS)
    assert set(results["group_summary"]["grupa"]) == {
        "negative", "positive", "ambiguous", "neutral"
    }


def test_h2_runs_kw_and_optional_subreddit_regression():
    results = h2_length(_analysis_data(), include_subreddit=True)

    assert results["tests"].iloc[0]["test"] == "kruskal_wallis_length_by_group"
    assert results["tests"].iloc[0]["effect_name"] == "epsilon_squared"
    assert {"C(grupa)", "C(subreddit)"} <= {
        term.split("[")[0] for term in results["regression"]["term"]
    }
    assert len(results["group_summary"]) == 4


def test_h3_applies_minimum_subreddit_size_and_ranks_non_neutral_share():
    results = h3_subreddits(_analysis_data(), minimum_comments=12)

    assert results["tests"].iloc[0]["effect_name"] == "cramers_v"
    assert results["subreddit_summary"]["n"].min() >= 12
    assert results["subreddit_summary"]["non_neutral_share"].is_monotonic_decreasing
    assert len(results["group_shares"]) == 3


def test_h4_compares_model_and_human_neutrality():
    frame = pd.DataFrame(
        {
            "anger": [0.8, 0.1, 0.2, 0.1],
            "joy": [0.1, 0.2, 0.1, 0.1],
            "neutral": [0.1, 0.7, 0.7, 0.8],
            "human_anger": [0.6, 0.1, 0.2, 0.1],
            "human_joy": [0.2, 0.1, 0.1, 0.1],
            "human_neutral": [0.2, 0.8, 0.7, 0.8],
        }
    )

    results = h4_neutrality(frame, EMOTIONS)

    assert results["source"].tolist() == ["model", "human"]
    assert results["dominant_neutral_share"].tolist() == [0.75, 0.75]
    assert results["neutral_score_mass_share"].between(0, 1).all()


def test_h1_rejects_empty_margin_filter_result():
    frame = _analysis_data()

    with pytest.raises(ValueError, match="removed every row"):
        h1_profanity(frame, EMOTIONS, minimum_margin=1)


def test_h1_to_h4_plots_save_figures(tmp_path):
    frame = _analysis_data()
    h1_results = h1_profanity(frame, EMOTIONS)
    h3_results = h3_subreddits(frame, minimum_comments=12)
    for emotion in EMOTIONS:
        frame[f"human_{emotion}"] = frame[emotion]

    h1_paths = plot_h1_profanity(h1_results, tmp_path)
    h2_path = plot_h2_length(frame, tmp_path)
    h3_path = plot_h3_subreddits(h3_results, tmp_path)
    h4_path = plot_h4_neutrality(frame, EMOTIONS, tmp_path)

    assert all(path.is_file() for path in h1_paths.values())
    assert h2_path.is_file()
    assert h3_path.is_file()
    assert h4_path.is_file()


def test_h5_reports_rater_caveat_and_saves_figures(tmp_path):
    frame = _analysis_data()
    frame["soglasnost"] = frame[EMOTIONS].max(axis=1)
    frame["uverenost"] = 0.2 + frame["vulg"]
    frame["n_rater"] = frame.index % 3 + 1
    for emotion in EMOTIONS:
        frame[f"human_{emotion}"] = frame[emotion]
    metrics = pd.DataFrame(
        {
            "level": ["28_emotions"] * 4,
            "label": [*EMOTIONS, "macro"],
            "f1": [0.3, 0.4, 0.5, 0.4],
        }
    )

    results = h5_disagreement(frame, metrics, EMOTIONS, minimum_raters=2)
    paths = plot_h5_disagreement(frame, results, tmp_path)

    assert results["correlation"].iloc[0]["status"] == "ok"
    assert len(results["emotion_summary"]) == len(EMOTIONS)
    assert results["rater_summary"]["n_rater"].min() >= 2
    assert all(path.is_file() for path in paths.values())
