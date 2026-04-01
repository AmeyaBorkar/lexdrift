"""Tests for lexdrift.nlp.entropy — compute_filing_entropy()."""

import pytest

from lexdrift.nlp.entropy import EntropyAnalysis, compute_filing_entropy


BOILERPLATE_TEXT = (
    "This report contains forward-looking statements within the meaning of "
    "the Private Securities Litigation Reform Act of 1995. These forward-looking "
    "statements involve known and unknown risks and uncertainties."
)

# Same boilerplate with minor word swaps -- low novelty
BOILERPLATE_VARIANT = (
    "This report contains forward-looking statements within the meaning of "
    "the Private Securities Litigation Reform Act of 1995. These forward-looking "
    "statements involve known and unknown risks and uncertainties and other factors."
)

# Completely different substantive content -- high novelty
NOVEL_TEXT = (
    "The cybersecurity breach compromised approximately 3.2 million customer records. "
    "The forensic investigation revealed unauthorized access to payment processing "
    "systems. Federal regulators initiated enforcement proceedings and the company "
    "faces potential penalties exceeding $200 million."
)


class TestComputeFilingEntropy:
    def test_returns_entropy_analysis(self):
        result = compute_filing_entropy(BOILERPLATE_TEXT, NOVEL_TEXT)
        assert isinstance(result, EntropyAnalysis)

    def test_novel_text_high_novelty_score(self):
        result = compute_filing_entropy(BOILERPLATE_TEXT, NOVEL_TEXT)
        assert result.novelty_score > 0.2

    def test_boilerplate_variant_low_novelty(self):
        result = compute_filing_entropy(BOILERPLATE_TEXT, BOILERPLATE_VARIANT)
        assert result.novelty_score < 0.3

    def test_identical_texts_zero_kl(self):
        result = compute_filing_entropy(BOILERPLATE_TEXT, BOILERPLATE_TEXT)
        assert result.kl_divergence == 0.0
        assert result.novelty_score < 0.05

    def test_entropy_values_non_negative(self):
        result = compute_filing_entropy(BOILERPLATE_TEXT, NOVEL_TEXT)
        assert result.entropy_prev >= 0.0
        assert result.entropy_curr >= 0.0
        assert result.kl_divergence >= 0.0

    def test_top_novel_tokens_populated(self):
        result = compute_filing_entropy(BOILERPLATE_TEXT, NOVEL_TEXT)
        # Novel text introduces words not in boilerplate
        assert len(result.top_novel_tokens) > 0
        assert result.unique_to_curr > 0
