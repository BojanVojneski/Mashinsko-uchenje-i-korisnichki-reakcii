"""Train, tune, and evaluate multi-label emotion classifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from src.data import DEFAULT_DATA_DIR

DEFAULT_C_VALUES = (0.5, 1.0, 2.0, 5.0)
DEFAULT_NGRAM_RANGES = ((1, 1), (1, 2))
DEFAULT_THRESHOLDS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
RANDOM_SEED = 42


@dataclass(frozen=True)
class TuningResult:
    model: Pipeline
    c: float
    ngram_range: tuple[int, int]
    threshold: float
    dev_macro_f1: float


def build_pipeline(
    C: float = 1.0,
    ngram_range: tuple[int, int] = (1, 1),
) -> Pipeline:
    """Create the TF-IDF plus one-vs-rest logistic regression pipeline."""
    if C <= 0:
        raise ValueError("C must be positive")
    if len(ngram_range) != 2 or ngram_range[0] < 1 or ngram_range[0] > ngram_range[1]:
        raise ValueError(f"Invalid ngram_range: {ngram_range}")

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    sublinear_tf=True,
                    min_df=2,
                    ngram_range=ngram_range,
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(max_iter=1000, C=C, random_state=RANDOM_SEED)
                ),
            ),
        ]
    )


def _label_columns(emotions: Sequence[str]) -> list[str]:
    return [f"label_{emotion}" for emotion in emotions]


def _training_arrays(
    frame: pd.DataFrame,
    emotions: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    required = {"text", *_label_columns(emotions)}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Data is missing model columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cannot train or evaluate on an empty data frame")
    if frame["text"].isna().any() or frame[_label_columns(emotions)].isna().any().any():
        raise ValueError("Model data contains missing text or labels")

    texts = frame["text"].astype(str).tolist()
    raw_labels = frame[_label_columns(emotions)].to_numpy()
    if not np.isin(raw_labels, [0, 1]).all():
        raise ValueError("Model labels must be binary")
    labels = raw_labels.astype(np.int8)
    return texts, labels


def tune(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    emotions: Sequence[str],
    c_values: Sequence[float] = DEFAULT_C_VALUES,
    ngram_ranges: Sequence[tuple[int, int]] = DEFAULT_NGRAM_RANGES,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> TuningResult:
    """Choose C, n-gram range, and one global threshold using dev macro-F1."""
    train_texts, y_train = _training_arrays(train, emotions)
    dev_texts, y_dev = _training_arrays(dev, emotions)
    if not c_values or not ngram_ranges or not thresholds:
        raise ValueError("Tuning grids must not be empty")
    if any(not 0 < threshold < 1 for threshold in thresholds):
        raise ValueError("Thresholds must be between 0 and 1")

    best: TuningResult | None = None
    for C in c_values:
        for ngram_range in ngram_ranges:
            model = build_pipeline(C=C, ngram_range=ngram_range)
            model.fit(train_texts, y_train)
            scores = model.predict_proba(dev_texts)
            for threshold in thresholds:
                predictions = (scores >= threshold).astype(np.int8)
                macro_f1 = float(f1_score(y_dev, predictions, average="macro", zero_division=0))
                if best is None or macro_f1 > best.dev_macro_f1:
                    best = TuningResult(
                        model=model,
                        c=float(C),
                        ngram_range=ngram_range,
                        threshold=float(threshold),
                        dev_macro_f1=macro_f1,
                    )

    if best is None:
        raise RuntimeError("Tuning completed without a candidate model")
    return best


def _load_complete_mapping(
    path: str | Path,
    emotions: Sequence[str],
) -> dict[str, list[str]]:
    mapping_path = Path(path)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not all(
        isinstance(group, str)
        and isinstance(labels, list)
        and all(isinstance(label, str) for label in labels)
        for group, labels in mapping.items()
    ):
        raise ValueError(f"Invalid emotion grouping mapping: {mapping_path}")

    complete = {group: list(labels) for group, labels in mapping.items()}
    mapped = [emotion for labels in complete.values() for emotion in labels]
    if "neutral" in emotions and "neutral" not in mapped:
        complete["neutral"] = ["neutral"]
        mapped.append("neutral")
    if set(mapped) != set(emotions) or len(mapped) != len(set(mapped)):
        raise ValueError(f"Emotion grouping does not cover each label exactly once: {mapping_path}")
    return complete


def _group_matrix(
    labels: np.ndarray,
    emotions: Sequence[str],
    mapping: dict[str, list[str]],
) -> np.ndarray:
    emotion_indices = {emotion: index for index, emotion in enumerate(emotions)}
    grouped = [
        labels[:, [emotion_indices[emotion] for emotion in members]].any(axis=1)
        for members in mapping.values()
    ]
    return np.column_stack(grouped).astype(np.int8)


def _level_results(
    level: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
) -> list[dict[str, object]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    rows = [
        {
            "level": level,
            "label": "macro",
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
            "support": pd.NA,
        }
    ]
    rows.extend(
        {
            "level": level,
            "label": label,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    )
    return rows


def evaluate(
    model: Pipeline,
    test: pd.DataFrame,
    emotions: Sequence[str],
    threshold: float,
    mapping_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Evaluate once at 28-emotion, Ekman, and sentiment-group levels."""
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1")
    texts, y_true = _training_arrays(test, emotions)
    scores = model.predict_proba(texts)
    if scores.shape != y_true.shape:
        raise ValueError(f"Model score shape {scores.shape} does not match labels {y_true.shape}")
    y_pred = (scores >= threshold).astype(np.int8)

    root = Path(mapping_dir) if mapping_dir is not None else DEFAULT_DATA_DIR / "simplified"
    ekman = _load_complete_mapping(root / "ekman_mapping.json", emotions)
    sentiment = _load_complete_mapping(root / "sentiment_mapping.json", emotions)
    results = _level_results("28_emotions", y_true, y_pred, emotions)

    for level, mapping in (("ekman", ekman), ("sentiment", sentiment)):
        grouped_true = _group_matrix(y_true, emotions, mapping)
        grouped_pred = _group_matrix(y_pred, emotions, mapping)
        results.extend(
            _level_results(level, grouped_true, grouped_pred, list(mapping))
        )
    return pd.DataFrame(results)
