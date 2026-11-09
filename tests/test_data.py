import pandas as pd
import pytest

from src.data import collapse_raw, hash_authors, labels_to_matrix, load_split


def test_collapse_raw_averages_emotions_and_counts_raters():
    raw = pd.DataFrame(
        {
            "id": ["a", "a", "b"],
            "text": ["first", "first", "second"],
            "author": ["alice", "alice", "bob"],
            "rater_id": [1, 2, 3],
            "example_very_unclear": [False, True, False],
            "anger": [1, 0, 0],
            "neutral": [0, 1, 1],
        }
    )

    collapsed = collapse_raw(raw, emotions=["anger", "neutral"]).set_index("id")

    assert collapsed.loc["a", "anger"] == 0.5
    assert collapsed.loc["a", "neutral"] == 0.5
    assert collapsed.loc["a", "n_rater"] == 2
    assert collapsed.loc["a", "n_unclear"] == 1
    assert collapsed.loc["a", "example_very_unclear"]


def test_hash_authors_drops_usernames_or_uses_salted_hmac():
    frame = pd.DataFrame({"author": ["alice", "bob"]})

    dropped = hash_authors(frame)
    hashed = hash_authors(frame, salt="local-test-salt")

    assert "author" not in dropped
    assert "author" not in hashed
    assert hashed["author_hash"].str.len().eq(64).all()
    assert hashed["author_hash"].iloc[0] != hash_authors(
        frame, salt="different-salt"
    )["author_hash"].iloc[0]


def test_labels_to_matrix_handles_ids_and_rejects_out_of_range_values():
    actual = labels_to_matrix(["0,2", [1], None, float("nan")], ["anger", "joy", "neutral"])

    assert actual.tolist() == [[1, 0, 1], [0, 1, 0], [0, 0, 0], [0, 0, 0]]
    with pytest.raises(ValueError, match="outside the valid range"):
        labels_to_matrix(["3"], ["anger", "joy", "neutral"])
    with pytest.raises(ValueError, match="must be an integer"):
        labels_to_matrix([1.5], ["anger", "joy", "neutral"])


def test_load_split_preserves_text_and_prefixes_label_columns(tmp_path):
    split_path = tmp_path / "train.tsv"
    split_path.write_text("A short comment\t0,2\tid-1\n", encoding="utf-8")

    split = load_split(split_path, emotions=["anger", "joy", "neutral"])

    assert split.loc[0, "text"] == "A short comment"
    assert split.loc[0, "split"] == "train"
    assert split.loc[0, "label_anger"] == 1
    assert split.loc[0, "label_neutral"] == 1
