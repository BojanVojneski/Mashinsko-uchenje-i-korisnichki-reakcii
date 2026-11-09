"""Load and prepare GoEmotions data for the analysis pipeline."""

from __future__ import annotations

import hashlib
import hmac
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
EXPECTED_SPLIT_ROWS = {"train": 43_410, "dev": 5_426, "test": 5_427}
RAW_METADATA_COLUMNS = (
    "text",
    "author",
    "subreddit",
    "created_utc",
    "link_id",
    "parent_id",
)


def _is_missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    return isinstance(value, (float, np.floating)) and math.isnan(float(value))


def load_emotions(path: str | Path | None = None) -> list[str]:
    """Read the ordered emotion names used by the simplified dataset."""
    emotion_path = Path(path) if path is not None else DEFAULT_DATA_DIR / "simplified" / "emotions.txt"
    emotions = [line.strip() for line in emotion_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not emotions or len(emotions) != len(set(emotions)):
        raise ValueError(f"Emotion list is empty or contains duplicates: {emotion_path}")
    return emotions


def load_raw(raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Read and combine the three rater-level raw CSV files."""
    directory = Path(raw_dir) if raw_dir is not None else DEFAULT_DATA_DIR / "raw"
    paths = sorted(directory.glob("goemotions_*.csv"))
    if len(paths) != 3:
        raise FileNotFoundError(f"Expected 3 goemotions_*.csv files in {directory}; found {len(paths)}")

    frames = [pd.read_csv(path, low_memory=False) for path in paths]
    columns = frames[0].columns
    if any(not frame.columns.equals(columns) for frame in frames[1:]):
        raise ValueError("Raw GoEmotions CSV files do not have matching columns")
    return pd.concat(frames, ignore_index=True)


def collapse_raw(df: pd.DataFrame, emotions: Sequence[str] | None = None) -> pd.DataFrame:
    """Collapse rater rows to one comment row with mean emotion scores."""
    emotion_names = list(emotions) if emotions is not None else load_emotions()
    required = {"id", "text", *emotion_names}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing required columns: {sorted(missing)}")
    if df["id"].isna().any():
        raise ValueError("Raw data contains missing comment IDs")

    grouped = df.groupby("id", sort=False, dropna=False)
    metadata = [column for column in RAW_METADATA_COLUMNS if column in df.columns]
    first_values = grouped[metadata].first()
    emotion_means = grouped[emotion_names].mean()
    if emotion_means.isna().any().any():
        raise ValueError("One or more comments have missing emotion means")

    if "rater_id" in df.columns:
        n_rater = grouped["rater_id"].nunique().rename("n_rater")
    else:
        n_rater = grouped.size().rename("n_rater")

    result = pd.concat([first_values, n_rater, emotion_means], axis=1)
    if "example_very_unclear" in df.columns:
        unclear = df["example_very_unclear"].astype("string").str.lower().isin(
            {"true", "1", "yes"}
        )
        unclear_counts = unclear.groupby(df["id"], sort=False).sum().rename("n_unclear")
        result = result.join(unclear_counts)
        result["example_very_unclear"] = result["n_unclear"].gt(0)

    result.index.name = "id"
    return result.reset_index()


def hash_authors(df: pd.DataFrame, salt: str | bytes | None = None) -> pd.DataFrame:
    """Remove usernames, or replace them with HMAC-SHA256 hashes when salted."""
    result = df.copy()
    if "author" not in result.columns:
        return result
    if salt is None:
        return result.drop(columns="author")

    salt_bytes = salt.encode("utf-8") if isinstance(salt, str) else salt
    if not salt_bytes:
        raise ValueError("Author hash salt must not be empty")

    def digest(author: object) -> str | None:
        if _is_missing_scalar(author):
            return None
        message = str(author).encode("utf-8")
        return hmac.new(salt_bytes, message, hashlib.sha256).hexdigest()

    result["author_hash"] = result["author"].map(digest)
    return result.drop(columns="author")


def labels_to_matrix(
    label_ids: Iterable[str | Iterable[int] | float | None],
    emotions: Sequence[str] | None = None,
) -> np.ndarray:
    """Convert comma-separated emotion IDs to a binary multilabel matrix."""
    emotion_names = list(emotions) if emotions is not None else load_emotions()
    values = list(label_ids)
    matrix = np.zeros((len(values), len(emotion_names)), dtype=np.int8)

    for row_index, value in enumerate(values):
        if _is_missing_scalar(value):
            continue
        if isinstance(value, str):
            if not value.strip():
                continue
            try:
                indices = [int(item.strip()) for item in value.split(",") if item.strip()]
            except ValueError as exc:
                raise ValueError(f"Invalid emotion ID list at row {row_index}: {value!r}") from exc
        elif isinstance(value, (int, np.integer, float, np.floating)):
            indices = [int(value)]
        else:
            try:
                indices = [int(item) for item in value]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid emotion ID list at row {row_index}: {value!r}") from exc

        for label_index in set(indices):
            if not 0 <= label_index < len(emotion_names):
                raise ValueError(
                    f"Emotion ID {label_index} at row {row_index} is outside "
                    f"the valid range 0..{len(emotion_names) - 1}"
                )
            matrix[row_index, label_index] = 1
    return matrix


def load_split(path: str | Path, emotions: Sequence[str] | None = None) -> pd.DataFrame:
    """Load a headerless simplified split with prefixed binary label columns."""
    split_path = Path(path)
    emotion_names = list(emotions) if emotions is not None else load_emotions()
    frame = pd.read_csv(
        split_path,
        sep="\t",
        header=None,
        names=["text", "emotion_ids", "id"],
        dtype="string",
    )
    if frame[["text", "emotion_ids", "id"]].isna().any().any():
        raise ValueError(f"Simplified split contains missing fields: {split_path}")
    if frame["id"].duplicated().any():
        raise ValueError(f"Simplified split contains duplicate comment IDs: {split_path}")

    labels = labels_to_matrix(frame["emotion_ids"].tolist(), emotion_names)
    label_frame = pd.DataFrame(
        labels,
        columns=[f"label_{emotion}" for emotion in emotion_names],
        index=frame.index,
    )
    return pd.concat([frame[["text", "id"]], label_frame], axis=1).assign(
        split=split_path.stem
    )


def build_master(
    data_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    author_salt: str | bytes | None = None,
) -> pd.DataFrame:
    """Build the one-row-per-comment master table and optionally save it."""
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    emotions = load_emotions(root / "simplified" / "emotions.txt")
    comments = collapse_raw(load_raw(root / "raw"), emotions)
    comments = hash_authors(comments, salt=author_salt)

    split_frames = []
    for name, expected_rows in EXPECTED_SPLIT_ROWS.items():
        frame = load_split(root / "simplified" / f"{name}.tsv", emotions)
        if len(frame) != expected_rows:
            raise ValueError(
                f"{name}.tsv has {len(frame)} rows; expected {expected_rows}"
            )
        split_frames.append(frame)

    labeled = pd.concat(split_frames, ignore_index=True)
    if labeled["id"].duplicated().any():
        raise ValueError("A comment ID appears in more than one simplified split")
    label_columns = [f"label_{emotion}" for emotion in emotions]
    master = comments.merge(
        labeled[["id", *label_columns, "split"]],
        on="id",
        how="left",
        validate="one_to_one",
    )

    if master["id"].duplicated().any():
        raise ValueError("Master table contains duplicate comment IDs")
    if master[emotions].isna().any().any():
        raise ValueError("Master table contains missing human emotion means")

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        master.to_csv(destination, index=False)
    return master
