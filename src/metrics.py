"""Compute sentiment, length, and lexicon-based vulgarity metrics."""

from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Collection

import pandas as pd
from textblob import TextBlob

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEXICON_PATH = PROJECT_ROOT / "data" / "profanity_lexicon.txt"
WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
EDGE_PUNCTUATION = string.punctuation + "’"


def _texts(df: pd.DataFrame, text_column: str) -> list[str]:
    if text_column not in df.columns:
        raise ValueError(f"DataFrame is missing text column {text_column!r}")
    if df[text_column].isna().any():
        raise ValueError(f"Text column {text_column!r} contains missing values")
    return df[text_column].astype(str).tolist()


def _normalize_word(word: str) -> str:
    return word.casefold().strip(EDGE_PUNCTUATION)


def load_lexicon(path: str | Path | None = None) -> set[str]:
    """Load a one-word-per-line profanity lexicon, ignoring blank/comment lines."""
    lexicon_path = Path(path) if path is not None else DEFAULT_LEXICON_PATH
    words = {
        _normalize_word(line.split("#", 1)[0].strip())
        for line in lexicon_path.read_text(encoding="utf-8").splitlines()
    }
    words.discard("")
    if not words:
        raise ValueError(f"Profanity lexicon is empty: {lexicon_path}")
    if any(re.search(r"\s", word) for word in words):
        raise ValueError(f"Profanity lexicon entries must be single words: {lexicon_path}")
    return words


def add_pol_sub(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Add TextBlob polarity (`pol`) and subjectivity (`sub`) columns."""
    texts = _texts(df, text_column)
    scores = [TextBlob(text).sentiment for text in texts]
    result = df.copy()
    result["pol"] = [float(score.polarity) for score in scores]
    result["sub"] = [float(score.subjectivity) for score in scores]
    return result


def add_dol(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Add `dol`, the number of whitespace-separated words in each comment."""
    texts = _texts(df, text_column)
    result = df.copy()
    result["dol"] = [len(text.split()) for text in texts]
    return result


def add_vulg(
    df: pd.DataFrame,
    lexicon: Collection[str],
    text_column: str = "text",
) -> pd.DataFrame:
    """Add lexicon word share (`vulg`) and any-match flag (`vulg_any`)."""
    words = {_normalize_word(word) for word in lexicon}
    words.discard("")
    if not words:
        raise ValueError("Profanity lexicon must contain at least one word")

    texts = _texts(df, text_column)
    shares = []
    any_matches = []
    for text in texts:
        tokens = [token for token in (_normalize_word(item) for item in WORD_PATTERN.findall(text)) if token]
        matched = sum(token in words for token in tokens)
        shares.append(matched / len(tokens) if tokens else 0.0)
        any_matches.append(int(matched > 0))

    result = df.copy()
    result["vulg"] = shares
    result["vulg_any"] = any_matches
    return result
