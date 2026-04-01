"""Tests for lexdrift.nlp.sentiment — score_sentiment() with Loughran-McDonald dictionary."""

from pathlib import Path
from unittest.mock import patch

import pytest

from lexdrift.nlp.sentiment import CATEGORIES, LEXICON_PATH, score_sentiment

# Skip entire module if the Loughran-McDonald CSV is not present on disk.
pytestmark = pytest.mark.skipif(
    not LEXICON_PATH.exists(),
    reason=f"Loughran-McDonald dictionary not found at {LEXICON_PATH}",
)


def _reset_lexicon():
    """Force the sentiment module to re-load on next call."""
    import lexdrift.nlp.sentiment as mod
    mod._loaded = False
    mod._lexicon.clear()


@pytest.fixture(autouse=True)
def _fresh_lexicon():
    """Ensure each test starts with a clean lexicon state."""
    _reset_lexicon()
    yield
    _reset_lexicon()


class TestScoreSentiment:
    def test_returns_all_categories(self):
        result = score_sentiment("The company reported steady revenue growth.")
        assert set(result.keys()) == set(CATEGORIES)

    def test_negative_text_scores_higher_on_negative(self):
        negative_text = (
            "The company faces significant losses, impairment charges, "
            "and declining revenue. Default risk is elevated."
        )
        positive_text = (
            "The company achieved record profits, strong growth, "
            "and improved efficiency across all segments."
        )
        neg_scores = score_sentiment(negative_text)
        pos_scores = score_sentiment(positive_text)
        assert neg_scores["negative"] > pos_scores["negative"]

    def test_positive_text_scores_higher_on_positive(self):
        positive_text = (
            "The company achieved strong gains, improved profitability, "
            "and favorable market conditions."
        )
        negative_text = (
            "The company reported losses, declining margins, and adverse "
            "regulatory actions."
        )
        pos_scores = score_sentiment(positive_text)
        neg_scores = score_sentiment(negative_text)
        assert pos_scores["positive"] > neg_scores["positive"]

    def test_empty_string_returns_zeros(self):
        result = score_sentiment("")
        assert all(v == 0.0 for v in result.values())

    def test_scores_are_normalized_fractions(self):
        result = score_sentiment(
            "There is substantial uncertainty regarding litigation risks "
            "and constraining regulatory requirements."
        )
        for cat in CATEGORIES:
            assert 0.0 <= result[cat] <= 1.0
