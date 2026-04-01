"""Tests for lexdrift.edgar.parser — clean_html() and extract_sections()."""

import pytest

from lexdrift.edgar.parser import clean_html, extract_sections


SIMPLE_HTML = """
<html>
<head><title>10-K Filing</title><style>body{color:red}</style></head>
<body>
  <p>The Company reported <b>strong revenue</b> growth.</p>
  <script>alert('x')</script>
  <p>Operating margins expanded   year   over   year.</p>
</body>
</html>
"""

FILING_TEXT_10K = """
PART I

Item 1A. Risk Factors

The Company faces significant risks including market volatility and
regulatory uncertainty. Revenue may decline in adverse conditions.
Competition from larger firms could reduce market share.

Item 7. Management Discussion and Analysis

Revenue grew 15% year over year driven by new product launches.
Operating margins expanded to 22% from 18%. The Company expects
continued growth in the next fiscal year.

Item 8. Financial Statements

Consolidated balance sheet and income statement are presented below.
Total assets were $5.2 billion as of December 31, 2024.
"""


class TestCleanHtml:
    def test_strips_tags(self):
        result = clean_html(SIMPLE_HTML)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "strong revenue" in result

    def test_removes_script_and_style(self):
        result = clean_html(SIMPLE_HTML)
        assert "alert" not in result
        assert "color:red" not in result

    def test_normalizes_whitespace(self):
        result = clean_html(SIMPLE_HTML)
        # Multiple spaces should be collapsed to single space
        assert "year   over" not in result
        assert "year over year" in result

    def test_empty_html(self):
        result = clean_html("<html><body></body></html>")
        assert result.strip() == ""


class TestExtractSections:
    def test_extracts_10k_risk_factors(self):
        sections = extract_sections(FILING_TEXT_10K, "10-K")
        assert "risk_factors" in sections
        assert "significant risks" in sections["risk_factors"]

    def test_extracts_10k_mdna(self):
        sections = extract_sections(FILING_TEXT_10K, "10-K")
        assert "mdna" in sections
        assert "Revenue grew" in sections["mdna"]

    def test_unknown_form_returns_full_text(self):
        sections = extract_sections(FILING_TEXT_10K, "8-K")
        assert "full_text" in sections
        assert sections["full_text"] == FILING_TEXT_10K

    def test_no_headings_returns_full_text(self):
        plain = "Just some plain text without any item headings."
        sections = extract_sections(plain, "10-K")
        assert "full_text" in sections
