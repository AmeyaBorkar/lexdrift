import logging
import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

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


def _strip_toc_block(text: str) -> str:
    """Remove leading Table of Contents blocks from cleaned text.

    If the cleaned text starts with lines that look like a TOC
    (multiple 'Item X...page' patterns), strip them out.
    """
    lines = text.split("\n")
    toc_pattern = re.compile(
        r"^\s*item\s+\d+[a-z]?\b.*\d+\s*$",
        re.IGNORECASE,
    )
    toc_heading = re.compile(r"table\s+of\s+contents", re.IGNORECASE)

    # Find the end of any leading TOC block
    toc_line_count = 0
    end_idx = 0
    in_toc = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if toc_heading.search(stripped):
            in_toc = True
            end_idx = i + 1
            continue
        if in_toc:
            if toc_pattern.match(stripped) or len(stripped) < 80:
                toc_line_count += 1
                end_idx = i + 1
            else:
                # We've left the TOC — stop if we saw enough TOC lines
                if toc_line_count >= 3:
                    break
                else:
                    # False positive — not really a TOC
                    in_toc = False
                    toc_line_count = 0
                    end_idx = 0
        elif i > 30:
            # Don't scan forever; TOC is near the top
            break

    if in_toc and toc_line_count >= 3:
        text = "\n".join(lines[end_idx:])

    return text


def _strip_xbrl_preamble(text: str) -> str:
    """Strip XBRL namespace URIs and metadata from the start of cleaned text.

    Some filings (GE, HON, MCD, etc.) are iXBRL documents where the first
    thousands of lines are XBRL namespace URIs, taxonomy references, and
    metadata. The actual prose starts much later. Detect and skip this preamble
    by finding the first line that looks like real prose (8+ words, not a URL,
    not a namespace declaration).
    """
    lines = text.split("\n")
    preamble_end = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Skip lines that look like XBRL/namespace junk
        if (
            stripped.startswith("http")
            or stripped.startswith("{")
            or stripped.startswith("xmlns")
            or re.match(r"^[\d\-]+$", stripped)  # just numbers/dates
            or re.match(r"^[a-z\-]+:[\w/\.\-]+$", stripped, re.IGNORECASE)  # namespace:uri
            or len(stripped) < 3
        ):
            preamble_end = i + 1
            continue
        # Real prose: 6+ words, contains spaces
        words = stripped.split()
        if len(words) >= 6:
            break
        # Short non-URL lines — could be start of content or still preamble
        # Keep scanning up to 5000 lines
        if i > 5000:
            break

    if preamble_end > 50:
        logger.info(f"Stripped {preamble_end} lines of XBRL preamble")
        return "\n".join(lines[preamble_end:])
    return text


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

    text = text.strip()

    # Strip XBRL preamble (namespace URIs, taxonomy references)
    text = _strip_xbrl_preamble(text)

    # Strip any leading Table of Contents blocks
    text = _strip_toc_block(text)

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


_CROSS_REFERENCE_PATTERNS = re.compile(
    r"see\s+item|refer\s+to\s+item|incorporated\s+by\s+reference|"
    r"filed\s+as\s+exhibit|included\s+in\s+part|"
    r"information\s+required\s+by\s+this\s+item\s+is\s+included\s+in|"
    r"information\s+relating\s+to\s+this\s+item\s+is\s+set\s+forth\s+in|"
    r"reference\s+is\s+made\s+to\s+the\s+information|"
    r"the\s+information\s+required\s+by\s+this\s+item\s+is\s+incorporated|"
    r"this\s+item\s+is\s+not\s+applicable|"
    r"response\s+to\s+this\s+item\s+is\s+included\s+in|"
    r"for\s+a\s+discussion.*?see\s+(?:part|note|item)",
    re.IGNORECASE,
)

# Patterns to find where a cross-reference points to
_XREF_TARGET_PATTERN = re.compile(
    r"(?:see|refer\s+to)\s+(?:note|footnote)\s+(\d+)",
    re.IGNORECASE,
)


def _is_cross_reference(text: str) -> bool:
    """Return True if a section is a cross-reference stub instead of real content.

    Cross-references are short sections (under 200 words) that say things like
    'See Item 2 below' or 'Information required by this item is incorporated
    by reference' instead of containing actual disclosure content.

    Also catches:
    - "Not applicable" or "None" as the entire section body
    - Sections under 20 words without a complete sentence
    """
    words = text.split()
    if len(words) >= 200:
        return False

    # Strip leading heading (e.g., "Item 1B. Unresolved Staff Comments")
    body = re.sub(
        r"^(?:item\s+\d+[a-z]?[\.\s:\-—]*\S*\s*)+",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    ).strip()
    body_lower = body.lower().strip().rstrip(".")

    # "None" or "Not applicable" as the entire section body
    if body_lower in ("none", "not applicable", "n/a", "na"):
        return True

    # Regex-based cross-reference detection
    if _CROSS_REFERENCE_PATTERNS.search(text):
        return True

    # Sections under 20 words that don't contain a complete sentence
    # (no period after 10+ words of actual content)
    body_words = body.split()
    if len(body_words) < 20:
        # Check if there's a period after at least 10 words of content
        sentences = re.split(r"[.!?]", body)
        has_complete_sentence = any(
            len(s.strip().split()) >= 10 for s in sentences if s.strip()
        )
        if not has_complete_sentence:
            return True

    return False


def _try_resolve_cross_reference(text: str, sections: dict[str, str]) -> str | None:
    """Try to find the actual content a cross-reference points to.

    For example, if the stub says 'See Note 15 to Consolidated Financial
    Statements', search for 'Note 15' in the financial_statements section.

    Returns the referenced content if found, otherwise None.
    """
    match = _XREF_TARGET_PATTERN.search(text)
    if not match:
        return None

    note_num = match.group(1)
    target_pattern = re.compile(
        rf"note\s+{note_num}\b[^\n]*\n",
        re.IGNORECASE,
    )

    # Search in financial_statements first, then all other sections
    search_order = []
    if "financial_statements" in sections:
        search_order.append(sections["financial_statements"])
    for sec_type, sec_text in sections.items():
        if sec_type != "financial_statements":
            search_order.append(sec_text)

    for sec_text in search_order:
        note_match = target_pattern.search(sec_text)
        if note_match:
            # Extract content from the note heading until the next note or
            # 5000 chars, whichever comes first
            start = note_match.start()
            next_note = re.search(
                rf"note\s+(?!{note_num}\b)\d+\b",
                sec_text[start + len(note_match.group()):],
                re.IGNORECASE,
            )
            if next_note:
                end = start + len(note_match.group()) + next_note.start()
            else:
                end = min(start + 5000, len(sec_text))
            content = sec_text[start:end].strip()
            if len(content.split()) >= 50:
                return content

    return None


# More aggressive patterns for the second pass — handle ALL-CAPS headings,
# headings without punctuation, and headings that include the section title.
_SECOND_PASS_PATTERNS_10K = [
    ("business", r"(?:^|\n)\s*(?:ITEM\s+1|item\s+1)[\.\s:\-—]*(?:business)?(?:\s|$)(?!a|b)", "Item 1"),
    ("risk_factors", r"(?:^|\n)\s*(?:ITEM\s+1A|item\s+1a)[\.\s:\-—]*(?:risk\s+factors)?", "Item 1A"),
    ("unresolved_staff_comments", r"(?:^|\n)\s*(?:ITEM\s+1B|item\s+1b)[\.\s:\-—]*(?:unresolved)?", "Item 1B"),
    ("properties", r"(?:^|\n)\s*(?:ITEM\s+2|item\s+2)[\.\s:\-—]*(?:properties)?(?:\s|$)", "Item 2"),
    ("legal_proceedings", r"(?:^|\n)\s*(?:ITEM\s+3|item\s+3)[\.\s:\-—]*(?:legal)?(?:\s|$)", "Item 3"),
    ("mdna", r"(?:^|\n)\s*(?:ITEM\s+7|item\s+7)[\.\s:\-—]*(?:management)?(?:\s|$)(?!a)", "Item 7"),
    ("quantitative_disclosures", r"(?:^|\n)\s*(?:ITEM\s+7A|item\s+7a)[\.\s:\-—]*(?:quantitative)?", "Item 7A"),
    ("financial_statements", r"(?:^|\n)\s*(?:ITEM\s+8|item\s+8)[\.\s:\-—]*(?:financial)?(?:\s|$)", "Item 8"),
]


def extract_sections(text: str, form_type: str) -> dict[str, str]:
    """Extract named sections from cleaned filing text using regex.

    Uses TOC detection to avoid matching section headings that are just
    table-of-contents entries rather than actual section starts.

    Cross-reference stubs (e.g. 'See Item 2 below') are automatically
    skipped. If any extracted section has fewer than 100 words, a second
    pass with more aggressive heading patterns is attempted.
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
    cross_ref_stubs: dict[str, str] = {}  # section_type -> stub text for resolution
    for i, (pos, section_type) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        section_text = text[pos:end].strip()

        # Detect cross-reference stubs — save for later resolution
        if _is_cross_reference(section_text):
            logger.debug(
                "Cross-reference stub for %s (%d words)",
                section_type, len(section_text.split()),
            )
            cross_ref_stubs[section_type] = section_text
            continue

        # Keep the longer match for each section type
        # (handles cases where a heading appears in both TOC and body)
        if section_type in sections and len(section_text) < len(sections[section_type]):
            continue

        sections[section_type] = section_text

    # ------------------------------------------------------------------
    # Second pass: if any section has < 100 words, try aggressive patterns
    # ------------------------------------------------------------------
    has_short_section = any(
        len(s.split()) < 100 for s in sections.values()
    )

    if has_short_section and form_type.startswith("10-K"):
        logger.info("Short sections detected; attempting second pass with aggressive patterns")
        aggressive_boundaries: list[tuple[int, str]] = []
        for section_type, pattern, _label in _SECOND_PASS_PATTERNS_10K:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if toc_start < match.start() < toc_end:
                    continue
                aggressive_boundaries.append((match.start(), section_type))

        if aggressive_boundaries:
            aggressive_boundaries.sort(key=lambda x: x[0])
            for i, (pos, section_type) in enumerate(aggressive_boundaries):
                end = (
                    aggressive_boundaries[i + 1][0]
                    if i + 1 < len(aggressive_boundaries)
                    else len(text)
                )
                section_text = text[pos:end].strip()

                if _is_cross_reference(section_text):
                    continue

                # Only replace if the aggressive match is longer (more content)
                existing = sections.get(section_type, "")
                if len(section_text.split()) > len(existing.split()):
                    sections[section_type] = section_text

    # ------------------------------------------------------------------
    # Third pass: match by section TITLE (not "Item X" prefix).
    # Some filings (GE, HON, MCD) use headings like "RISK FACTORS."
    # or "LEGAL PROCEEDINGS." without the "Item 1A" prefix.
    # ------------------------------------------------------------------
    still_short = any(len(s.split()) < 100 for s in sections.values())
    if still_short and form_type.startswith("10-K"):
        logger.info("Still short sections; trying title-based matching (third pass)")
        _TITLE_PATTERNS_10K = [
            ("risk_factors", r"(?:^|\n)\s*RISK\s+FACTORS[\.\s:\-—]*(?:\n|$)"),
            ("legal_proceedings", r"(?:^|\n)\s*LEGAL\s+PROCEEDINGS[\.\s:\-—]*(?:\n|$)"),
            ("mdna", r"(?:^|\n)\s*MANAGEMENT.S?\s+DISCUSSION\s+AND\s+ANALYSIS"),
            ("business", r"(?:^|\n)\s*(?:DESCRIPTION\s+OF\s+)?BUSINESS[\.\s:\-—]*(?:\n|$)"),
            ("properties", r"(?:^|\n)\s*PROPERTIES[\.\s:\-—]*(?:\n|$)"),
            ("quantitative_disclosures", r"(?:^|\n)\s*QUANTITATIVE\s+AND\s+QUALITATIVE"),
            ("financial_statements", r"(?:^|\n)\s*FINANCIAL\s+STATEMENTS\s+AND\s+SUPPLEMENTARY"),
            ("unresolved_staff_comments", r"(?:^|\n)\s*UNRESOLVED\s+STAFF\s+COMMENTS"),
        ]
        title_boundaries: list[tuple[int, str]] = []
        for section_type, pattern in _TITLE_PATTERNS_10K:
            for match in re.finditer(pattern, text):
                if toc_start < match.start() < toc_end:
                    continue
                title_boundaries.append((match.start(), section_type))

        if title_boundaries:
            title_boundaries.sort(key=lambda x: x[0])
            for i, (pos, section_type) in enumerate(title_boundaries):
                end = (
                    title_boundaries[i + 1][0]
                    if i + 1 < len(title_boundaries)
                    else len(text)
                )
                section_text = text[pos:end].strip()
                if _is_cross_reference(section_text):
                    continue
                existing = sections.get(section_type, "")
                if len(section_text.split()) > len(existing.split()):
                    sections[section_type] = section_text

    if not sections:
        logger.warning("No valid sections extracted, returning full text")
        return {"full_text": text}

    # ------------------------------------------------------------------
    # Cross-reference resolution: for stubs we skipped, try to find the
    # actual content they point to in other extracted sections.
    # ------------------------------------------------------------------
    for sec_type, stub_text in cross_ref_stubs.items():
        if sec_type in sections:
            continue  # already resolved through a different match
        resolved = _try_resolve_cross_reference(stub_text, sections)
        if resolved:
            logger.info(
                "Resolved cross-reference for %s (%d words)",
                sec_type, len(resolved.split()),
            )
            sections[sec_type] = resolved

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
