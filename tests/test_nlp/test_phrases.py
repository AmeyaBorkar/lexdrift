"""Tests for lexdrift.nlp.phrases — TF-IDF keyphrases and watchlist phrases."""

from unittest.mock import patch

import pytest

from lexdrift.nlp.phrases import (
    check_priority_phrases,
    check_watchlist_phrases,
    compare_keyphrases,
    compare_phrases,
    extract_keyphrases_tfidf,
    _get_ngram_tf,
)


class TestExtractKeyphrasesTfidf:
    def test_returns_list_of_dicts(self):
        text = (
            "The company faces going concern issues. Going concern doubt "
            "was raised by auditors. Going concern language persists."
        )
        result = extract_keyphrases_tfidf(text, top_k=10)
        assert isinstance(result, list)
        if result:
            assert "phrase" in result[0]
            assert "score" in result[0]

    def test_empty_text_returns_empty(self):
        assert extract_keyphrases_tfidf("", top_k=5) == []

    def test_top_k_limits_results(self):
        text = (
            "alpha beta gamma delta epsilon zeta theta iota kappa lambda "
            "alpha beta gamma delta epsilon zeta theta iota kappa lambda "
        ) * 5
        result = extract_keyphrases_tfidf(text, top_k=3)
        assert len(result) <= 3


class TestCompareKeyphrases:
    def test_returns_expected_keys(self):
        prev = "The company has stable revenue and consistent growth."
        curr = "The company faces going concern issues raised by auditors."
        result = compare_keyphrases(prev, curr)
        assert "appeared" in result
        assert "disappeared" in result
        assert "intensified" in result
        assert "diminished" in result

    def test_identical_texts_minimal_changes(self):
        text = "The company reported quarterly earnings in line with expectations."
        result = compare_keyphrases(text, text)
        # Same text should have no appeared/disappeared
        assert result["appeared"] == []
        assert result["disappeared"] == []


class TestComparePhrasesBackwardCompat:
    def test_returns_priority_and_discovered_keys(self):
        prev = "The company has strong financials."
        curr = "The company faces some challenges."
        with patch("lexdrift.nlp.phrases.load_priority_phrases", return_value=set()):
            result = compare_phrases(prev, curr)
        assert "priority" in result
        assert "discovered" in result
        assert "appeared" in result["discovered"]
        assert "disappeared" in result["discovered"]


class TestCheckWatchlistPhrases:
    @patch("lexdrift.nlp.phrases.load_priority_phrases", return_value=set())
    def test_detects_appeared_extra_phrases(self, _mock_load):
        prev = "The company has strong financials."
        curr = "There is going concern doubt about the company."
        result = check_watchlist_phrases(
            prev, curr, extra_phrases={"going concern", "material weakness"}
        )
        assert "going concern" in result["appeared"]

    @patch("lexdrift.nlp.phrases.load_priority_phrases", return_value=set())
    def test_detects_disappeared_extra_phrases(self, _mock_load):
        prev = "The auditor noted a material weakness in controls."
        curr = "The company has improved its internal procedures."
        result = check_watchlist_phrases(
            prev, curr, extra_phrases={"material weakness"}
        )
        assert "material weakness" in result["disappeared"]

    @patch("lexdrift.nlp.phrases.load_priority_phrases", return_value=set())
    def test_detects_persisted_phrases(self, _mock_load):
        prev = "There is a going concern risk factor."
        curr = "The going concern risk persists into 2025."
        result = check_watchlist_phrases(
            prev, curr, extra_phrases={"going concern"}
        )
        assert "going concern" in result["persisted"]


class TestCheckPriorityPhrasesAlias:
    """Verify the old name still works (backward compat)."""

    @patch("lexdrift.nlp.phrases.load_priority_phrases", return_value=set())
    def test_alias_works(self, _mock_load):
        prev = "The company has strong financials."
        curr = "There is going concern doubt about the company."
        result = check_priority_phrases(
            prev, curr, extra_phrases={"going concern"}
        )
        assert "going concern" in result["appeared"]
