"""Tests for lexdrift.nlp.tokenizer — tokenize() and sentence_split()."""

import pytest

from lexdrift.nlp.tokenizer import sentence_split, tokenize


class TestTokenize:
    def test_basic_words(self):
        tokens = tokenize("The Company reported revenue of $10 million.")
        assert "the" in tokens
        assert "company" in tokens
        assert "reported" in tokens
        assert "revenue" in tokens
        # Numbers and dollar signs are not word tokens
        assert "$10" not in tokens
        assert "10" not in tokens

    def test_lowercases_output(self):
        tokens = tokenize("REVENUE Growth Outlook")
        assert tokens == ["revenue", "growth", "outlook"]

    def test_handles_contractions(self):
        tokens = tokenize("The company didn't disclose its won't")
        assert "didn't" in tokens
        assert "won't" in tokens

    def test_empty_string(self):
        assert tokenize("") == []

    def test_only_punctuation(self):
        assert tokenize("!!! --- $$$") == []


class TestSentenceSplit:
    def test_multiple_sentences(self):
        text = (
            "Revenue grew 5%. The company expanded operations. "
            "Risk factors include inflation."
        )
        sentences = sentence_split(text)
        assert len(sentences) == 3
        assert sentences[0].startswith("Revenue")
        assert sentences[2].startswith("Risk")

    def test_single_sentence(self):
        sentences = sentence_split("This is one sentence.")
        assert len(sentences) == 1

    def test_empty_string(self):
        assert sentence_split("") == []

    def test_preserves_abbreviations_mid_sentence(self):
        text = "Revenue was $10.5 million in Q1 2025."
        sentences = sentence_split(text)
        # Should stay as one sentence because the period in "10.5" is not
        # followed by a space + capital letter pattern.
        assert len(sentences) == 1
