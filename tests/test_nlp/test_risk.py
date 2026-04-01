"""Tests for lexdrift.nlp.risk — score_sentence_risk()."""

from unittest.mock import patch

import pytest

from lexdrift.nlp.risk import RiskScore, score_sentence_risk


def _fake_sentiment(text: str) -> dict[str, float]:
    """Stub that returns zero sentiment so we isolate keyword-based scoring."""
    return {
        "negative": 0.0,
        "positive": 0.0,
        "uncertainty": 0.0,
        "litigious": 0.0,
        "constraining": 0.0,
    }


@patch("lexdrift.nlp.risk.score_sentiment", side_effect=_fake_sentiment)
class TestScoreSentenceRisk:
    def test_going_concern_is_critical(self, _mock_sent):
        result = score_sentence_risk(
            "The auditor raised substantial doubt about the company's going concern status."
        )
        assert isinstance(result, RiskScore)
        assert result.level == "critical"
        assert result.score >= 0.9
        assert "going concern" in result.triggers

    def test_material_weakness_is_critical(self, _mock_sent):
        result = score_sentence_risk(
            "Management identified a material weakness in internal controls over financial reporting."
        )
        assert result.level == "critical"
        assert "material weakness" in result.triggers

    def test_boilerplate_asc_842(self, _mock_sent):
        result = score_sentence_risk(
            "In accordance with ASC 842, the Company adopted the new lease accounting standard."
        )
        assert result.level == "boilerplate"
        assert result.score < 0.2

    def test_medium_risk_restructuring(self, _mock_sent):
        result = score_sentence_risk(
            "The company announced a significant restructuring of its operations."
        )
        assert result.level == "medium"
        assert "restructuring" in result.triggers

    def test_low_risk_neutral_sentence(self, _mock_sent):
        result = score_sentence_risk(
            "The company updated its website and corporate branding."
        )
        assert result.level == "low"
        assert result.score <= 0.2

    def test_high_risk_goodwill_impairment(self, _mock_sent):
        result = score_sentence_risk(
            "The company recorded a goodwill impairment charge of $500 million."
        )
        assert result.level == "high"
        assert "goodwill impairment" in result.triggers
