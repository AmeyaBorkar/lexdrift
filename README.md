# LexDrift — SEC Semantic Drift Analyzer

## What Is This?

A financial intelligence tool that tracks **how public companies change their language** across SEC filings (10-K, 10-Q, 8-K) over time. Instead of asking "what did the company say?", this tool answers **"what did the company say *differently* compared to last quarter, and why does that matter?"**

When a company quietly adds "supply chain uncertainty" to its risk factors, shifts from "confident" to "cautiously optimistic," removes a previously stated growth target, or rewrites its revenue recognition disclosures — these linguistic shifts are **leading indicators** of financial trouble, strategic pivots, or regulatory concerns. Analysts who catch these changes early have a material information advantage.

---

## The Problem

Thousands of public companies file 10-K (annual) and 10-Q (quarterly) reports with the SEC every quarter. These filings contain rich, unstructured text — risk factors, management discussion & analysis (MD&A), forward-looking statements, legal disclosures — that evolves over time.

**The current workflow is broken:**

- Analysts manually read hundreds of pages per company per quarter
- They rely on memory or personal notes to recall what changed from the prior filing
- There is no systematic way to track linguistic evolution across a coverage universe of 50–500 companies
- Subtle but critical changes (a new risk factor buried on page 87, a removed growth target, a shift in tone) get missed regularly
- By the time the language change manifests in the financial numbers, the stock has already moved

**Who suffers from this problem:**

- **Equity analysts** covering 20+ companies who can't read every page of every filing
- **Credit analysts** monitoring debt covenants and risk disclosures across large portfolios
- **Short sellers** looking for early signs of deterioration in management confidence
- **Compliance/audit teams** tracking disclosure consistency
- **Retail investors** who never read filings at all and rely on summaries that miss nuance

---

## Why This Is Novel

Existing NLP tools in finance do **sentiment analysis** — they score a single document as positive or negative. Bloomberg Terminal shows filing text but doesn't analyze it. Refinitiv provides news sentiment. None of them do what this tool does:

**Semantic drift analysis** is fundamentally different from sentiment analysis:

| Sentiment Analysis | Semantic Drift Analysis |
|---|---|
| Scores one document in isolation | Compares the same company's documents over time |
| "This filing is negative" | "This filing is *more negative than the last three*" |
| Binary/scalar output | Rich diff output showing exactly what changed |
| Generic model, same for all companies | Company-specific baseline that learns each firm's normal language |
| Widely available | Does not exist as a product |

**Academic validation:** Research by Loughran & McDonald (2011), Cohen, Malloy & Nguyen (2020), and others has demonstrated that textual changes in SEC filings predict future stock returns, earnings restatements, and SEC enforcement actions. This research has **never been productized** in an accessible way.

---

## What the End Product Looks Like

### Company Dashboard
A per-company view showing:
- **Filing Timeline:** Every 10-K and 10-Q plotted chronologically with an overall "drift score" indicating how much the language changed from the prior period
- **Section-by-Section Diff:** For each major filing section (Risk Factors, MD&A, Business Description, Legal Proceedings, etc.), a visual diff highlighting additions, deletions, and modifications — similar to a GitHub diff but for financial prose
- **Risk Factor Evolution:** A dedicated tracker for risk factors, showing when new risks were added, old risks removed, and existing risks reworded. Risk factors are the most predictive section for future trouble
- **Tone Shift Timeline:** Charts showing how sentiment, uncertainty, litigiousness, and confidence in language evolve across quarters using the Loughran-McDonald financial lexicon
- **Key Phrase Tracker:** Tracks the appearance and disappearance of critical phrases ("going concern," "material weakness," "restatement," "goodwill impairment," "supply chain," etc.)

### Watchlist & Alerts
- Users add companies to a watchlist
- The system automatically ingests new filings when they appear on EDGAR
- Alerts are triggered when:
  - A filing's drift score exceeds a configurable threshold
  - A new risk factor appears that wasn't in the prior filing
  - Specific monitored phrases appear or disappear
  - The overall tone of MD&A shifts beyond historical norms for that company

### Comparison View
- Compare any two filings from the same company side-by-side
- Compare the same section across multiple quarters to see evolution
- Compare how two different companies in the same sector describe the same risk

### Portfolio / Screener View
- Upload a list of tickers or select an index (S&P 500, Russell 2000)
- See a ranked table of all companies sorted by recent drift score
- Filter by section, by type of change (additions vs. deletions), by sentiment direction
- Identify "outlier" companies whose language is changing much more than peers

---

## Data Sources & Resources

### Primary Data — SEC EDGAR (Free, No API Key)

| Resource | URL | What It Provides |
|---|---|---|
| EDGAR Full-Text Search API | `https://efts.sec.gov/LATEST/search-index?q=...` | Search across all filings by keyword, date, form type. JSON response. |
| Company Filing Metadata | `https://data.sec.gov/submissions/CIK{cik}.json` | All filings for a given company — accession numbers, dates, form types. JSON. |
| Individual Filing Access | `https://www.sec.gov/Archives/edgar/data/{CIK}/{accession}/` | Full text of any filing in HTML/XML/SGML. |
| Bulk Filing Index | `https://www.sec.gov/Archives/edgar/full-index/` | Quarterly indexes of all filings. Pipe-delimited text files. |
| Company Tickers & CIK Map | `https://www.sec.gov/files/company_tickers.json` | Maps ticker symbols to CIK numbers. JSON. |
| XBRL Company Facts | `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` | Structured financial data for any filer. JSON. |

**Rate limit:** 10 requests/second. Must include a `User-Agent` header with name and email.

**Coverage:** ~800,000+ filers. Full-text filings available from ~1993 onward. XBRL structured data from ~2009 onward.

### Financial Sentiment Lexicon

| Resource | URL | What It Provides |
|---|---|---|
| Loughran-McDonald Master Dictionary | `https://sraf.nd.edu/loughranmcdonald-master-dictionary/` | ~85,000 words classified as positive, negative, uncertain, litigious, constraining, superfluous. CSV. The gold standard for financial text analysis. |
| Loughran-McDonald Sentiment Word Lists | `https://sraf.nd.edu/loughranmcdonald-master-dictionary/` | Pre-filtered word lists for each sentiment category. |

**License:** Free for academic and research use.

### Pre-Parsed Datasets (Supplementary)

| Resource | Platform | What It Provides |
|---|---|---|
| SEC 10-K Filings Dataset | Kaggle (search "SEC filings 10-K") | Pre-parsed MD&A sections and risk factors. CSV. |
| SEC Filings NLP Dataset | HuggingFace (search "sec-filings") | Tokenized filing text ready for NLP pipelines. Parquet. |
| EDGAR Online Parsed Filings | Various academic sources | Section-segmented filing text. Availability varies. |

### Earnings Releases (Supplementary)

| Resource | URL | What It Provides |
|---|---|---|
| 8-K Item 2.02 Filings | EDGAR (form type `8-K`, search "Item 2.02") | Earnings release text filed with the SEC. Free. Not full call transcripts, but contains management's prepared commentary. |

### Academic Papers (Background Reading)

| Paper | Authors | Key Finding |
|---|---|---|
| "When Is a Liability Not a Liability?" | Loughran & McDonald (2011) | Financial-specific word lists outperform generic sentiment dictionaries for SEC filing analysis. Introduced the Loughran-McDonald lexicon. |
| "Lazy Prices" | Cohen, Malloy & Nguyen (2020) | Quarter-over-quarter changes in 10-K/10-Q text predict future stock returns. Firms that change their filings more earn lower future returns. The foundational paper for this tool's thesis. |
| "Measuring Readability in Financial Disclosures" | Loughran & McDonald (2014) | Bog Index for measuring filing complexity. Complex filings correlate with worse outcomes. |
| "Textual Analysis in Accounting and Finance" | Loughran & McDonald (2016) | Comprehensive survey of NLP methods applied to financial disclosures. |
| "The Annual Report Algorithm" | Hoberg & Lewis (2017) | Using 10-K text to identify peer firms and predict competitive dynamics. |
| "MD&A Disclosure and the Firm's Ability to Continue as a Going Concern" | Mayew, Sethuraman & Venkatachalam (2015) | MD&A linguistic features predict going-concern risk before auditors flag it. |

---

## Competitive Landscape

| Tool | What It Does | What It Lacks |
|---|---|---|
| Bloomberg Terminal | Displays raw filing text; basic keyword search | No temporal analysis, no drift detection, no linguistic comparison |
| Refinitiv/LSEG | News sentiment scores | Single-document sentiment, not cross-filing drift |
| AlphaSense | Enterprise search across filings and transcripts | Powerful search, but no systematic drift scoring or evolution tracking |
| Sentieo (now AlphaSense) | Document search with annotations | Search-oriented, not drift-oriented |
| Calcbench | Structured financial data extraction from XBRL | Numbers only, no text analysis |
| This Tool | Cross-filing semantic drift detection and alerting | — |

The key differentiator: every existing tool treats each filing as an **independent document**. This tool treats each filing as a **data point in a time series** of corporate communication.

---

## Target Users

1. **Buy-side equity analysts** — Detect early warning signals across their coverage universe
2. **Credit analysts at banks and funds** — Monitor risk disclosure evolution in credit portfolios
3. **Short sellers / activist investors** — Identify companies whose language is deteriorating before fundamentals
4. **Compliance and audit teams** — Track disclosure consistency and flag material changes
5. **Academic researchers** — Study corporate communication patterns at scale
6. **Retail investors (power users)** — Access institutional-grade filing analysis without reading 200-page documents

---

## Success Metrics

This tool is successful when:

- A user can onboard a new company and see its complete filing evolution in under 30 seconds
- Drift alerts fire 1–2 quarters before material events (earnings misses, restatements, lawsuits)
- The system processes new EDGAR filings within 1 hour of publication
- Coverage spans the entire EDGAR universe (all public filers), not just large-caps
- False positive rate on alerts is low enough that users don't disable them
