import pandas as pd
import pytest

from src.metrics import add_dol, add_pol_sub, add_vulg, load_lexicon


def test_add_dol_counts_whitespace_separated_words():
    actual = add_dol(pd.DataFrame({"text": ["two words", "  one\tword  ", ""]}))

    assert actual["dol"].tolist() == [2, 2, 0]


def test_add_pol_sub_stays_within_documented_ranges():
    actual = add_pol_sub(pd.DataFrame({"text": ["I love this!", "I hate this.", "Just text."]}))

    assert actual["pol"].between(-1, 1).all()
    assert actual["sub"].between(0, 1).all()


def test_add_vulg_normalizes_case_and_punctuation_and_handles_empty_text():
    frame = pd.DataFrame({"text": ["Damn, this is crap!", "ordinary words", "!!!"]})

    actual = add_vulg(frame, {"damn", "crap"})

    assert actual["vulg"].tolist() == [0.5, 0.0, 0.0]
    assert actual["vulg_any"].tolist() == [1, 0, 0]


def test_load_lexicon_ignores_comments_and_blank_lines(tmp_path):
    lexicon_path = tmp_path / "lexicon.txt"
    lexicon_path.write_text("# comment\nDamn\n\ncrap # note\n", encoding="utf-8")

    assert load_lexicon(lexicon_path) == {"damn", "crap"}


def test_metrics_reject_missing_text_and_empty_lexicon():
    with pytest.raises(ValueError, match="missing values"):
        add_dol(pd.DataFrame({"text": [None]}))
    with pytest.raises(ValueError, match="at least one word"):
        add_vulg(pd.DataFrame({"text": ["text"]}), set())
