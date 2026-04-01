import logging
import re

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# SEC 10-K item headings with robust regex patterns that handle:
# "Item 1A.", "ITEM 1A -", "Item 1A:", "Item 1A Risk Factors", etc.
SECTION_PATTERNS_10K = [
    ("business", r"item\s+1[\.\s:\-—]+(?!a|b)", "Item 1 — Business"),
    ("risk_factors", r"item\s+1a[\.\s:\-—]+", "Item 1A — Risk Factors"),
    ("unresolved_staff_comments", r"item\s+1b[\.\s:\-—]+", "Item 1B — Unresolved Staff Comments"),
    ("properties", r"item\s+2[\.\s:\-—]+", "Item 2 — Properties"),
    ("legal_proceedings", r"item\s+3[\.\s:\-—]+", "Item 3 — Legal Proceedings"),
    ("mdna", r"item\s+7[\.\s:\-—]+(?!a)", "Item 7 — MD&A"),
    ("quantitative_disclosures", r"item\s+7a[\.\s:\-—]+", "Item 7A — Quantitative Disclosures"),
    ("financial_statements", r"item\s+8[\.\s:\-—]+", "Item 8 — Financial Statements"),
]

SECTION_PATTERNS_10Q = [
    ("financial_statements", r"item\s+1[\.\s:\-—]+(?!a|b)", "Item 1 — Financial Statements"),
    ("risk_factors", r"item\s+1a[\.\s:\-—]+", "Item 1A — Risk Factors"),
    ("mdna", r"item\s+2[\.\s:\-—]+", "Item 2 — MD&A"),
    ("quantitative_disclosures", r"item\s+3[\.\s:\-—]+", "Item 3 — Quantitative Disclosures"),
    ("legal_proceedings", r"part\s+ii[\s\S]{0,30}?item\s+1[\.\s:\-—]+", "Part II Item 1 — Legal"),
]

SECTION_PATTERNS_8K = [
    ("material_agreement", r"item\s+1\.01[\s:\-—]+", "Item 1.01 — Entry into a Material Definitive Agreement"),
    ("material_agreement_termination", r"item\s+1\.02[\s:\-—]+", "Item 1.02 — Termination of a Material Definitive Agreement"),
    ("acquisition_disposition", r"item\s+2\.01[\s:\-—]+", "Item 2.01 — Completion of Acquisition or Disposition"),
    ("results_of_operations", r"item\s+2\.02[\s:\-—]+", "Item 2.02 — Results of Operations and Financial Condition"),
    ("exit_disposal_costs", r"item\s+2\.05[\s:\-—]+", "Item 2.05 — Costs Associated with Exit or Disposal Activities"),
    ("material_impairments", r"item\s+2\.06[\s:\-—]+", "Item 2.06 — Material Impairments"),
    ("delisting_notice", r"item\s+3\.01[\s:\-—]+", "Item 3.01 — Notice of Delisting"),
    ("accountant_change", r"item\s+4\.01[\s:\-—]+", "Item 4.01 — Changes in Registrant's Certifying Accountant"),
    ("financial_restatement", r"item\s+4\.02[\s:\-—]+", "Item 4.02 — Non-Reliance on Previously Issued Financial Statements"),
    ("director_officer_change", r"item\s+5\.02[\s:\-—]+", "Item 5.02 — Departure/Election of Directors or Officers"),
    ("regulation_fd", r"item\s+7\.01[\s:\-—]+", "Item 7.01 — Regulation FD Disclosure"),
    ("other_events", r"item\s+8\.01[\s:\-—]+", "Item 8.01 — Other Events"),
    ("financial_exhibits", r"item\s+9\.01[\s:\-—]+", "Item 9.01 — Financial Statements and Exhibits"),
]

# iXBRL section name mappings (used in modern filings since ~2019)
IXBRL_SECTION_MAP = {
    "BusinessDescriptionAndBasisOfPresentationTextBlock": "business",
    "RiskFactorsTextBlock": "risk_factors",
    "UnresolvedStaffCommentsTextBlock": "unresolved_staff_comments",
    "PropertiesTextBlock": "properties",
    "LegalProceedingsTextBlock": "legal_proceedings",
    "ManagementDiscussionAndAnalysisTextBlock": "mdna",
    "QuantitativeAndQualitativeDisclosuresAboutMarketRiskTextBlock": "quantitative_disclosures",
    "FinancialStatementsTextBlock": "financial_statements",
    # Common alternative tag names
    "DiscussionAndAnalysisOfFinancialConditionAndResultsOfOperations": "mdna",
    "RiskFactors": "risk_factors",
    "LegalProceedings": "legal_proceedings",
}

# Patterns that indicate a Table of Contents zone
_TOC_INDICATORS = re.compile(
    r"table\s+of\s+contents|index\s+to|"
    r"(?:page|pg\.?)\s*\d+|"
    r"\.\s*\.\s*\.\s*\d+",  # dot leaders like "Item 1A...........12"
    re.IGNORECASE,
)


def clean_html(html_text: str) -> str:
    """Strip HTML tags and normalize whitespace from SEC filing HTML."""
    soup = BeautifulSoup(html_text, "lxml")

    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def _try_ixbrl_extraction(html_text: str) -> dict[str, str] | None:
    """Try to extract sections from iXBRL tagged filings (post-2019).

    Modern SEC filings use inline XBRL with <ix:nonNumeric> tags that
    explicitly label sections, making extraction reliable.
    Returns None if the filing doesn't use iXBRL or sections aren't found.
    """
    soup = BeautifulSoup(html_text, "lxml")

    # Look for ix:nonNumeric tags (iXBRL text blocks) — include case variations
    # and ix:nonfraction for numeric data that may contain section references
    ix_tags = soup.find_all(["ix:nonnumeric", "ix:nonNumeric", "ix:nonfraction"])
    if not ix_tags:
        # Try with namespace prefix variations — search specifically for tags
        # with name attributes containing known iXBRL prefixes, not all named elements
        _IXBRL_PREFIXES = ("us-gaap:", "dei:", "srt:", "country:")
        ix_tags = soup.find_all(attrs={"name": True})
        ix_tags = [
            tag for tag in ix_tags
            if any(str(tag.get("name", "")).startswith(prefix) for prefix in _IXBRL_PREFIXES)
            and any(key in str(tag.get("name", "")) for key in IXBRL_SECTION_MAP)
        ]

    if not ix_tags:
        return None

    sections: dict[str, str] = {}
    for tag in ix_tags:
        tag_name = tag.get("name", "")
        # Extract the local name (after the namespace prefix)
        local_name = tag_name.split(":")[-1] if ":" in tag_name else tag_name

        for ixbrl_key, section_type in IXBRL_SECTION_MAP.items():
            if ixbrl_key.lower() in local_name.lower():
                text = tag.get_text(separator="\n").strip()
                text = re.sub(r"[ \t]+", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text)
                if text and len(text) > len(sections.get(section_type, "")):
                    sections[section_type] = text
                break

    if sections:
        logger.info(f"Extracted {len(sections)} sections via iXBRL tags")
        return sections

    return None


def _detect_toc_zone(text: str) -> tuple[int, int]:
    """Detect the approximate start and end of the Table of Contents.

    Returns (toc_start, toc_end) character positions. If no TOC is detected,
    returns (0, 0) so no filtering occurs.
    """
    # Look for "TABLE OF CONTENTS" heading
    toc_match = re.search(r"table\s+of\s+contents", text.lower())
    if not toc_match:
        return (0, 0)

    toc_start = toc_match.start()

    # The TOC typically ends when we hit the first real section heading
    # that's followed by substantial content (not just a page number).
    # Scan forward from TOC start for a region without dot leaders / page numbers.
    lines = text[toc_start:].split("\n")
    char_count = toc_start
    consecutive_content_lines = 0

    for line in lines:
        char_count += len(line) + 1
        stripped = line.strip()
        if not stripped:
            continue

        has_toc_marker = bool(_TOC_INDICATORS.search(stripped))
        is_short = len(stripped) < 100

        if has_toc_marker or (is_short and consecutive_content_lines == 0):
            consecutive_content_lines = 0
        else:
            consecutive_content_lines += 1
            # 5 consecutive content lines = we've left the TOC
            if consecutive_content_lines >= 5:
                toc_end = char_count - len(line)
                return (toc_start, toc_end)

    return (toc_start, toc_start + min(len(text) - toc_start, 5000))


def extract_sections(text: str, form_type: str) -> dict[str, str]:
    """Extract named sections from cleaned filing text using regex.

    Uses TOC detection to avoid matching section headings that are just
    table-of-contents entries rather than actual section starts.
    """
    if form_type.startswith("10-K"):
        patterns = SECTION_PATTERNS_10K
    elif form_type.startswith("10-Q"):
        patterns = SECTION_PATTERNS_10Q
    elif form_type.startswith("8-K"):
        patterns = SECTION_PATTERNS_8K
    else:
        return {"full_text": text}

    text_lower = text.lower()
    toc_start, toc_end = _detect_toc_zone(text)

    # Find all section boundaries, excluding TOC zone
    boundaries: list[tuple[int, str]] = []
    for section_type, pattern, _label in patterns:
        for match in re.finditer(pattern, text_lower):
            # Skip matches inside the Table of Contents
            if toc_start < match.start() < toc_end:
                continue
            boundaries.append((match.start(), section_type))

    if not boundaries:
        logger.warning(f"No section headings found in {form_type} filing")
        return {"full_text": text}

    boundaries.sort(key=lambda x: x[0])

    # Extract text between consecutive boundaries
    sections: dict[str, str] = {}
    for i, (pos, section_type) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        section_text = text[pos:end].strip()

        # Keep the longer match for each section type
        # (handles cases where a heading appears in both TOC and body)
        if section_type in sections and len(section_text) < len(sections[section_type]):
            continue

        sections[section_type] = section_text

    if not sections:
        logger.warning("No valid sections extracted, returning full text")
        return {"full_text": text}

    return sections


def parse_filing(html_text: str, form_type: str) -> dict[str, str]:
    """Full pipeline: iXBRL extraction → fallback to regex-based extraction.

    For modern filings (post-2019), tries structured iXBRL tags first.
    Falls back to HTML cleaning + regex section detection for older filings.
    """
    # Try iXBRL first (most reliable for modern filings)
    sections = _try_ixbrl_extraction(html_text)
    if sections:
        return sections

    # Fallback: clean HTML and use regex
    cleaned = clean_html(html_text)
    return extract_sections(cleaned, form_type)
