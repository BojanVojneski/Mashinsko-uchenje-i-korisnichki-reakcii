import json

import pandas as pd

from src.model import build_pipeline, evaluate, tune

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
