"""Train, tune, and evaluate multi-label emotion classifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.data import DEFAULT_DATA_DIR

DEFAULT_C_VALUES = (0.5, 1.0, 2.0, 5.0)
DEFAULT_ALPHAS = (0.1, 0.5, 1.0, 2.0)
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


@dataclass(frozen=True)
class ComparisonTuningResult:
    model: Pipeline
    alpha: float
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


def build_comparison_pipeline(
    alpha: float = 1.0,
    ngram_range: tuple[int, int] = (1, 1),
) -> Pipeline:
    """Create the TF-IDF plus one-vs-rest multinomial Naive Bayes pipeline."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
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
                OneVsRestClassifier(MultinomialNB(alpha=alpha)),
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


def _best_dev_candidate(
    train_texts: list[str],
    y_train: np.ndarray,
    dev_texts: list[str],
    y_dev: np.ndarray,
    candidates: Sequence[tuple[float, tuple[int, int], Pipeline]],
    thresholds: Sequence[float],
) -> tuple[Pipeline, float, tuple[int, int], float, float]:
    if not candidates or not thresholds:
        raise ValueError("Tuning grids must not be empty")
    if any(not 0 < threshold < 1 for threshold in thresholds):
        raise ValueError("Thresholds must be between 0 and 1")

    best: tuple[Pipeline, float, tuple[int, int], float, float] | None = None
    for parameter, ngram_range, model in candidates:
        model.fit(train_texts, y_train)
        scores = model.predict_proba(dev_texts)
        if scores.shape != y_dev.shape:
            raise ValueError(
                f"Model score shape {scores.shape} does not match labels {y_dev.shape}"
            )
        for threshold in thresholds:
            predictions = (scores >= threshold).astype(np.int8)
            macro_f1 = float(f1_score(y_dev, predictions, average="macro", zero_division=0))
            if best is None or macro_f1 > best[4]:
                best = (model, float(parameter), ngram_range, float(threshold), macro_f1)

    if best is None:
        raise RuntimeError("Tuning completed without a candidate model")
    return best


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
    candidates = [
        (float(C), ngram_range, build_pipeline(C=C, ngram_range=ngram_range))
        for C in c_values
        for ngram_range in ngram_ranges
    ]
    model, C, ngram_range, threshold, macro_f1 = _best_dev_candidate(
        train_texts, y_train, dev_texts, y_dev, candidates, thresholds
    )
    return TuningResult(model, C, ngram_range, threshold, macro_f1)


def tune_comparison(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    emotions: Sequence[str],
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    ngram_ranges: Sequence[tuple[int, int]] = DEFAULT_NGRAM_RANGES,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> ComparisonTuningResult:
    """Tune Naive Bayes smoothing, n-grams, and threshold on dev macro-F1."""
    train_texts, y_train = _training_arrays(train, emotions)
    dev_texts, y_dev = _training_arrays(dev, emotions)
    candidates = [
        (float(alpha), ngram_range, build_comparison_pipeline(alpha, ngram_range))
        for alpha in alphas
        for ngram_range in ngram_ranges
    ]
    model, alpha, ngram_range, threshold, macro_f1 = _best_dev_candidate(
        train_texts, y_train, dev_texts, y_dev, candidates, thresholds
    )
    return ComparisonTuningResult(model, alpha, ngram_range, threshold, macro_f1)


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


def evaluate_models(
    models: Mapping[str, tuple[Pipeline, float]],
    test: pd.DataFrame,
    emotions: Sequence[str],
    mapping_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Evaluate named models at their chosen thresholds and combine the tables."""
    if not models:
        raise ValueError("At least one model is required")
    results = []
    for name, (model, threshold) in models.items():
        if not name.strip():
            raise ValueError("Model names must not be empty")
        model_results = evaluate(model, test, emotions, threshold, mapping_dir)
        model_results.insert(0, "model", name)
        results.append(model_results)
    return pd.concat(results, ignore_index=True)


def sample_misclassified_examples(
    model: Pipeline,
    frame: pd.DataFrame,
    emotions: Sequence[str],
    threshold: float,
    sample_size: int = 30,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Select a reproducible sample of errors for manual, private review."""
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1")
    if sample_size < 1:
        raise ValueError("sample_size must be positive")

    texts, y_true = _training_arrays(frame, emotions)
    scores = model.predict_proba(texts)
    if scores.shape != y_true.shape:
        raise ValueError(f"Model score shape {scores.shape} does not match labels {y_true.shape}")
    y_pred = (scores >= threshold).astype(np.int8)
    error_counts = np.count_nonzero(y_true != y_pred, axis=1)
    error_indices = np.flatnonzero(error_counts)

    rows = []
    for index in error_indices:
        rows.append(
            {
                "id": frame["id"].iloc[index] if "id" in frame.columns else pd.NA,
                "text": texts[index],
                "true_labels": ", ".join(
                    emotion for emotion, present in zip(emotions, y_true[index]) if present
                ),
                "predicted_labels": ", ".join(
                    emotion for emotion, present in zip(emotions, y_pred[index]) if present
                ),
                "error_count": int(error_counts[index]),
            }
        )
    columns = ["id", "text", "true_labels", "predicted_labels", "error_count"]
    errors = pd.DataFrame(rows, columns=columns)
    if len(errors) > sample_size:
        errors = errors.sample(n=sample_size, random_state=random_seed)
    return errors.reset_index(drop=True)


def out_of_fold_scores(
    master: pd.DataFrame,
    model_template: Pipeline,
    emotions: Sequence[str],
    n_splits: int = 5,
    mapping_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Score comments out of fold, optionally saving the scored master table."""
    if master.empty:
        raise ValueError("Cannot score an empty master table")
    if len(emotions) < 2 or len(set(emotions)) != len(emotions):
        raise ValueError("At least two unique emotion labels are required")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    required = {"id", "text", *emotions, *_label_columns(emotions)}
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"Master table is missing scoring columns: {sorted(missing)}")
    if master["id"].isna().any() or master["id"].duplicated().any():
        raise ValueError("Master table IDs must be present and unique")
    if master["text"].isna().any():
        raise ValueError("Master table contains missing text")

    label_columns = _label_columns(emotions)
    labels_present = master[label_columns].notna()
    has_any_labels = labels_present.any(axis=1)
    has_all_labels = labels_present.all(axis=1)
    if (has_any_labels != has_all_labels).any():
        raise ValueError("Each row must have either all simplified labels or none")

    labeled_mask = has_all_labels.to_numpy()
    labeled = master.loc[labeled_mask]
    unlabeled = master.loc[~labeled_mask]
    if len(labeled) < n_splits:
        raise ValueError(
            f"Need at least n_splits={n_splits} labeled rows; found {len(labeled)}"
        )

    _, human_labels = _training_arrays(labeled, emotions)
    human_scores = master[list(emotions)].to_numpy(dtype=float)
    if not np.isfinite(human_scores).all() or ((human_scores < 0) | (human_scores > 1)).any():
        raise ValueError("Human emotion scores must be finite values in [0, 1]")
    if any(f"human_{emotion}" in master.columns for emotion in emotions):
        raise ValueError("Master table already contains prefixed human emotion columns")

    scores = np.zeros((len(master), len(emotions)), dtype=float)
    labeled_indices = np.flatnonzero(labeled_mask)
    unlabeled_indices = np.flatnonzero(~labeled_mask)
    texts = master["text"].astype(str).to_numpy()
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    for train_indices, validation_indices in folds.split(labeled):
        fold_model = clone(model_template)
        fold_model.fit(
            labeled.iloc[train_indices]["text"].astype(str).tolist(),
            human_labels[train_indices],
        )
        validation_scores = fold_model.predict_proba(
            labeled.iloc[validation_indices]["text"].astype(str).tolist()
        )
        if validation_scores.shape != (len(validation_indices), len(emotions)):
            raise ValueError(
                f"Fold score shape {validation_scores.shape} does not match labels"
            )
        scores[labeled_indices[validation_indices]] = validation_scores
        if len(unlabeled_indices):
            unlabeled_scores = fold_model.predict_proba(texts[unlabeled_indices])
            if unlabeled_scores.shape != (len(unlabeled_indices), len(emotions)):
                raise ValueError(
                    f"Unlabeled score shape {unlabeled_scores.shape} does not match labels"
                )
            scores[unlabeled_indices] += unlabeled_scores

    if len(unlabeled_indices):
        scores[unlabeled_indices] /= n_splits
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Model scores must be finite values in [0, 1]")

    result = master.copy()
    for index, emotion in enumerate(emotions):
        result[f"human_{emotion}"] = human_scores[:, index]
        result[emotion] = scores[:, index]

    top_two = np.sort(scores, axis=1)[:, -2:]
    dominant_indices = np.argmax(scores, axis=1)
    sentiment_path = (
        Path(mapping_dir) if mapping_dir is not None else DEFAULT_DATA_DIR / "simplified"
    ) / "sentiment_mapping.json"
    sentiment = _load_complete_mapping(sentiment_path, emotions)
    group_by_emotion = {
        emotion: group
        for group, members in sentiment.items()
        for emotion in members
    }

    result["glavna_emocija"] = [emotions[index] for index in dominant_indices]
    result["uverenost"] = top_two[:, 1]
    result["razlika"] = top_two[:, 1] - top_two[:, 0]
    result["grupa"] = [
        group_by_emotion[emotion] for emotion in result["glavna_emocija"]
    ]
    result["soglasnost"] = human_scores.max(axis=1)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False)
    return result
