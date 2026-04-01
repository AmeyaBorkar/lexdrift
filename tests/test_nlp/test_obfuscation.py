"""Tests for lexdrift.nlp.obfuscation — detect_obfuscation()."""

import pytest

from lexdrift.nlp.obfuscation import ObfuscationScore, detect_obfuscation


# A specific, concrete disclosure paragraph
CONCRETE_TEXT = (
    "On January 15, 2025, the Company terminated 1,200 employees across "
    "3 manufacturing facilities. The layoff resulted in $45.2 million in "
    "severance charges. Revenue declined 12% year over year to $890 million. "
    "The Company recorded a write-off of $23 million related to discontinued "
    "product lines."
)

# Same information wrapped in vague, euphemistic language
VAGUE_TEXT = (
    "The Company undertook an organizational realignment initiative to optimize "
    "its workforce and streamline operations. This restructuring initiative may "
    "result in certain non-cash charges. Revenue experienced some moderation "
    "relative to the prior period, reflecting broader macroeconomic headwinds. "
    "The Company also pursued a balance sheet optimization through certain "
    "asset rationalization measures that could potentially improve long-term "
    "efficiency. Management believes these actions will substantially enhance "
    "the Company's competitive positioning going forward and generally anticipates "
    "improved results in subsequent periods."
)


class TestDetectObfuscation:
    def test_returns_obfuscation_score(self):
        result = detect_obfuscation(CONCRETE_TEXT, VAGUE_TEXT)
        assert isinstance(result, ObfuscationScore)

    def test_concrete_to_vague_scores_higher(self):
        result = detect_obfuscation(CONCRETE_TEXT, VAGUE_TEXT)
        # The overall score should be above zero when text gets vaguer
        assert result.overall_obfuscation_score > 0.0

    def test_detects_euphemisms(self):
        result = detect_obfuscation(CONCRETE_TEXT, VAGUE_TEXT)
        # Should detect at least one euphemistic substitution
        # e.g., "layoff" -> "organizational realignment"
        assert len(result.detected_euphemisms) > 0
        euphemism_specifics = {e["specific_term"] for e in result.detected_euphemisms}
        assert "layoff" in euphemism_specifics or "write-off" in euphemism_specifics

    def test_identical_texts_low_score(self):
        result = detect_obfuscation(CONCRETE_TEXT, CONCRETE_TEXT)
        assert result.overall_obfuscation_score < 0.1
        assert result.detected_euphemisms == []

    def test_specificity_decreases(self):
        result = detect_obfuscation(CONCRETE_TEXT, VAGUE_TEXT)
        # Concrete text has more numbers/dates; vague text has more hedge words
        assert result.specificity_change < 0.0

    def test_component_scores_present(self):
        result = detect_obfuscation(CONCRETE_TEXT, VAGUE_TEXT)
        assert "density" in result.component_scores
        assert "specificity" in result.component_scores
        assert "readability" in result.component_scores
        assert "euphemism" in result.component_scores
