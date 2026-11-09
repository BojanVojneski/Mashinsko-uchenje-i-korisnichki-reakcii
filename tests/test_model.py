import json

import pandas as pd

from src.model import (
    build_comparison_pipeline,
    build_pipeline,
    evaluate,
    evaluate_models,
    out_of_fold_scores,
    sample_misclassified_examples,
    tune,
    tune_comparison,
)

EMOTIONS = ["anger", "joy", "neutral"]


def _synthetic_split(texts, labels):
    frame = pd.DataFrame({"text": texts})
    for index, emotion in enumerate(EMOTIONS):
        frame[f"label_{emotion}"] = [row[index] for row in labels]
    return frame


def test_pipeline_fits_multilabel_text_and_returns_probabilities():
    train = _synthetic_split(
        [
            "angry words", "angry words again", "very angry",
            "joyful words", "joyful words again", "very joyful",
            "plain words", "plain words again",
        ],
        [
            [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 1, 0], [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1],
        ],
    )
    model = build_pipeline(C=1, ngram_range=(1, 1))
    model.fit(train["text"].tolist(), train[[f"label_{name}" for name in EMOTIONS]].to_numpy())

    scores = model.predict_proba(["angry words", "joyful words"])

    assert scores.shape == (2, 3)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_tune_uses_only_supplied_train_and_dev_frames():
    train = _synthetic_split(
        [
            "angry words", "angry words again",
            "joyful words", "joyful words again",
            "plain words", "plain words again",
        ],
        [
            [1, 0, 0], [1, 0, 0],
            [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1],
        ],
    )
    dev = _synthetic_split(
        ["very angry", "very joyful"],
        [[1, 0, 0], [0, 1, 0]],
    )

    result = tune(
        train,
        dev,
        EMOTIONS,
        c_values=[1],
        ngram_ranges=[(1, 1)],
        thresholds=[0.3, 0.5],
    )

    assert result.c == 1
    assert result.ngram_range == (1, 1)
    assert result.threshold in {0.3, 0.5}
    assert 0 <= result.dev_macro_f1 <= 1


def test_comparison_model_tunes_on_dev_and_predicts_probabilities():
    train = _synthetic_split(
        [
            "angry words", "angry words again",
            "joyful words", "joyful words again",
            "plain words", "plain words again",
        ],
        [
            [1, 0, 0], [1, 0, 0],
            [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1],
        ],
    )
    dev = _synthetic_split(
        ["very angry", "very joyful", "plain words"],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )

    result = tune_comparison(
        train,
        dev,
        EMOTIONS,
        alphas=[1.0],
        ngram_ranges=[(1, 1)],
        thresholds=[0.3, 0.5],
    )

    assert result.alpha == 1.0
    assert result.ngram_range == (1, 1)
    assert result.threshold in {0.3, 0.5}
    assert 0 <= result.dev_macro_f1 <= 1
    scores = result.model.predict_proba(["angry words", "joyful words"])
    assert scores.shape == (2, len(EMOTIONS))
    assert ((scores >= 0) & (scores <= 1)).all()


def test_evaluate_reports_three_levels_and_adds_neutral_group(tmp_path):
    for filename, mapping in (
        ("ekman_mapping.json", {"anger": ["anger"], "joy": ["joy"]}),
        ("sentiment_mapping.json", {"negative": ["anger"], "positive": ["joy"]}),
    ):
        (tmp_path / filename).write_text(json.dumps(mapping), encoding="utf-8")

    train = _synthetic_split(
        [
            "angry words", "angry words again",
            "joyful words", "joyful words again",
            "plain words", "plain words again",
        ],
        [
            [1, 0, 0], [1, 0, 0],
            [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1],
        ],
    )
    model = build_pipeline(C=1, ngram_range=(1, 1))
    model.fit(train["text"].tolist(), train[[f"label_{name}" for name in EMOTIONS]].to_numpy())
    results = evaluate(model, train, EMOTIONS, threshold=0.5, mapping_dir=tmp_path)

    assert set(results["level"]) == {"28_emotions", "ekman", "sentiment"}
    assert "neutral" in results.loc[results["level"] == "sentiment", "label"].tolist()
    assert results.loc[results["label"] == "macro", "f1"].between(0, 1).all()


def test_evaluate_models_and_sample_errors_for_manual_review(tmp_path):
    for filename, mapping in (
        ("ekman_mapping.json", {"anger": ["anger"], "joy": ["joy"]}),
        ("sentiment_mapping.json", {"negative": ["anger"], "positive": ["joy"]}),
    ):
        (tmp_path / filename).write_text(json.dumps(mapping), encoding="utf-8")

    train = _synthetic_split(
        [
            "angry words", "angry words again",
            "joyful words", "joyful words again",
            "plain words", "plain words again",
        ],
        [
            [1, 0, 0], [1, 0, 0],
            [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1],
        ],
    ).assign(id=[f"comment-{index}" for index in range(6)])
    label_columns = [f"label_{emotion}" for emotion in EMOTIONS]
    y_train = train[label_columns].to_numpy()
    logistic = build_pipeline().fit(train["text"], y_train)
    naive_bayes = build_comparison_pipeline().fit(train["text"], y_train)

    comparison = evaluate_models(
        {"logistic": (logistic, 0.5), "naive_bayes": (naive_bayes, 0.5)},
        train,
        EMOTIONS,
        mapping_dir=tmp_path,
    )
    errors = sample_misclassified_examples(
        logistic, train, EMOTIONS, threshold=0.999999, sample_size=2
    )

    assert set(comparison["model"]) == {"logistic", "naive_bayes"}
    assert len(comparison) > 0
    assert len(errors) == 2
    assert errors["id"].notna().all()
    assert {"text", "true_labels", "predicted_labels", "error_count"} <= set(errors.columns)


def test_out_of_fold_scores_cover_labeled_and_unlabeled_rows(tmp_path):
    for filename, mapping in (
        ("ekman_mapping.json", {"anger": ["anger"], "joy": ["joy"]}),
        ("sentiment_mapping.json", {"negative": ["anger"], "positive": ["joy"]}),
    ):
        (tmp_path / filename).write_text(json.dumps(mapping), encoding="utf-8")

    rows = []
    for emotion_index, emotion in enumerate(EMOTIONS):
        for example_index in range(6):
            labels = [0, 0, 0]
            labels[emotion_index] = 1
            human_scores = [0.1, 0.1, 0.1]
            human_scores[emotion_index] = 0.8
            rows.append(
                {
                    "id": f"{emotion}-{example_index}",
                    "text": f"{emotion} words",
                    **{f"label_{name}": labels[index] for index, name in enumerate(EMOTIONS)},
                    **{name: human_scores[index] for index, name in enumerate(EMOTIONS)},
                }
            )
    for example_index in range(2):
        rows.append(
            {
                "id": f"unlabeled-{example_index}",
                "text": "unlabeled words",
                **{f"label_{name}": float("nan") for name in EMOTIONS},
                **{name: 0.2 for name in EMOTIONS},
            }
        )
    master = pd.DataFrame(rows)

    scored = out_of_fold_scores(
        master,
        build_pipeline(C=1, ngram_range=(1, 1)),
        EMOTIONS,
        n_splits=3,
        mapping_dir=tmp_path,
        output_path=tmp_path / "outputs" / "scored.csv",
    )

    assert len(scored) == len(master)
    assert scored[EMOTIONS].notna().all().all()
    assert scored[EMOTIONS].ge(0).all().all() and scored[EMOTIONS].le(1).all().all()
    assert scored["human_anger"].equals(master["anger"])
    assert scored["soglasnost"].equals(master[EMOTIONS].max(axis=1))
    assert scored["uverenost"].between(0, 1).all()
    assert scored["razlika"].between(0, 1).all()
    assert scored["grupa"].isin({"negative", "positive", "neutral"}).all()
    assert (tmp_path / "outputs" / "scored.csv").is_file()
