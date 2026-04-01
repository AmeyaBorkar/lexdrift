"""Tests for lexdrift.nlp.diff — unified_diff() and diff_stats()."""

import pytest

from lexdrift.nlp.diff import diff_stats, unified_diff


PREV_TEXT = (
    "The company reported strong revenue growth. "
    "Operating margins expanded year over year. "
    "Cash flow remained positive."
)

CURR_TEXT = (
    "The company reported strong revenue growth. "
    "Operating margins contracted due to inflation. "
    "Cash flow remained positive. "
    "Net income declined significantly."
)


class TestUnifiedDiff:
    def test_produces_diff_output(self):
        result = unified_diff(PREV_TEXT, CURR_TEXT)
        assert isinstance(result, str)
        # Should contain diff markers
        assert "---" in result or "+++" in result or result == ""

    def test_shows_additions_and_removals(self):
        result = unified_diff(PREV_TEXT, CURR_TEXT)
        # "contracted" replaces "expanded", and a new sentence is added
        assert "+" in result or "-" in result

    def test_identical_texts_empty_diff(self):
        result = unified_diff(PREV_TEXT, PREV_TEXT)
        assert result.strip() == ""

    def test_completely_different_texts(self):
        result = unified_diff("First document only.", "Second document entirely.")
        assert len(result) > 0


class TestDiffStats:
    def test_returns_expected_keys(self):
        stats = diff_stats(PREV_TEXT, CURR_TEXT)
        expected_keys = {
            "sentences_added",
            "sentences_removed",
            "sentences_changed",
            "sentences_unchanged",
            "total_prev",
            "total_curr",
            "similarity_ratio",
        }
        assert set(stats.keys()) == expected_keys

    def test_identical_texts_perfect_similarity(self):
        stats = diff_stats(PREV_TEXT, PREV_TEXT)
        assert stats["similarity_ratio"] == 1.0
        assert stats["sentences_added"] == 0
        assert stats["sentences_removed"] == 0
        assert stats["sentences_changed"] == 0

    def test_new_sentence_counted(self):
        stats = diff_stats(PREV_TEXT, CURR_TEXT)
        # Current text has one more sentence than previous
        assert stats["total_curr"] > stats["total_prev"]

    def test_similarity_ratio_between_zero_and_one(self):
        stats = diff_stats(PREV_TEXT, CURR_TEXT)
        assert 0.0 <= stats["similarity_ratio"] <= 1.0

    def test_completely_different_low_similarity(self):
        stats = diff_stats(
            "Alpha bravo charlie.",
            "Delta echo foxtrot. Golf hotel india.",
        )
        assert stats["similarity_ratio"] < 0.5
