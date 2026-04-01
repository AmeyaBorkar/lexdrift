"""Tests for lexdrift.nlp.phrases — discover_ngram_changes() and check_priority_phrases()."""

from unittest.mock import patch

import pytest

from lexdrift.nlp.phrases import check_priority_phrases, discover_ngram_changes


class TestDiscoverNgramChanges:
    def test_detects_new_bigrams(self):
        prev = "The company has stable revenue and consistent growth over time."
        curr = (
            "The company has stable revenue and consistent growth over time. "
            "Going concern issues raised by auditors. Going concern doubt is real. "
            "Going concern language was added."
        )
        result = discover_ngram_changes(prev, curr, min_freq=1, top_k=20)
        appeared_phrases = {item["phrase"] for item in result["appeared"]}
        assert "going concern" in appeared_phrases

    def test_detects_disappeared_bigrams(self):
        prev = (
            "Material weakness in internal controls identified. "
            "Material weakness must be remediated. "
            "Material weakness disclosures are required."
        )
        curr = "The company has improved its internal controls substantially."
        result = discover_ngram_changes(prev, curr, min_freq=1, top_k=20)
        disappeared_phrases = {item["phrase"] for item in result["disappeared"]}
        assert "material weakness" in disappeared_phrases

    def test_returns_appeared_and_disappeared_keys(self):
        result = discover_ngram_changes("old text here", "new text here", min_freq=1, top_k=10)
        assert "appeared" in result
        assert "disappeared" in result

    def test_identical_texts_no_changes(self):
        text = "The company reported quarterly earnings in line with expectations."
        result = discover_ngram_changes(text, text, min_freq=1, top_k=10)
        assert result["appeared"] == []
        assert result["disappeared"] == []

    def test_top_k_limits_results(self):
        prev = "alpha beta " * 5
        curr = (
            "alpha beta " * 5
            + "gamma delta " * 5
            + "epsilon zeta " * 5
            + "theta iota " * 5
        )
        result = discover_ngram_changes(prev, curr, min_freq=1, top_k=2)
        assert len(result["appeared"]) <= 2


class TestCheckPriorityPhrases:
    @patch("lexdrift.nlp.phrases.load_priority_phrases", return_value=set())
    def test_detects_appeared_extra_phrases(self, _mock_load):
        prev = "The company has strong financials."
        curr = "There is going concern doubt about the company."
        result = check_priority_phrases(
            prev, curr, extra_phrases={"going concern", "material weakness"}
        )
        assert "going concern" in result["appeared"]

    @patch("lexdrift.nlp.phrases.load_priority_phrases", return_value=set())
    def test_detects_disappeared_extra_phrases(self, _mock_load):
        prev = "The auditor noted a material weakness in controls."
        curr = "The company has improved its internal procedures."
        result = check_priority_phrases(
            prev, curr, extra_phrases={"material weakness"}
        )
        assert "material weakness" in result["disappeared"]

    @patch("lexdrift.nlp.phrases.load_priority_phrases", return_value=set())
    def test_detects_persisted_phrases(self, _mock_load):
        prev = "There is a going concern risk factor."
        curr = "The going concern risk persists into 2025."
        result = check_priority_phrases(
            prev, curr, extra_phrases={"going concern"}
        )
        assert "going concern" in result["persisted"]
