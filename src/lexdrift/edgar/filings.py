import logging
from datetime import date

from lexdrift.edgar.client import edgar_client
from lexdrift.edgar.tickers import pad_cik

logger = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_path}"

FORM_TYPES = {"10-K", "10-Q", "8-K", "10-K/A", "10-Q/A"}


async def get_filing_metadata(cik: str) -> dict:
    """Fetch all filing metadata for a company from SEC EDGAR."""
    url = SUBMISSIONS_URL.format(cik=pad_cik(cik))
    return await edgar_client.get_json(url)


def parse_filing_list(
    metadata: dict,
    form_types: set[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict]:
    """Parse the filing list from SEC submission metadata JSON.

    Returns a list of dicts with: accession_number, form_type, filing_date,
    report_date, primary_document.
    """
    if form_types is None:
        form_types = FORM_TYPES

    recent = metadata.get("filings", {}).get("recent", {})
    if not recent:
        return []

    accessions = recent.get("accessionNumber", [])
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    primary_docs = recent.get("primaryDocument", [])

    filings = []
    for i in range(len(accessions)):
        form = forms[i] if i < len(forms) else ""
        if form not in form_types:
            continue

        filing_date_str = filing_dates[i] if i < len(filing_dates) else ""
        if filing_date_str:
            fdate = date.fromisoformat(filing_date_str)
            if start_date and fdate < start_date:
                continue
            if end_date and fdate > end_date:
                continue
        else:
            fdate = None

        report_date_str = report_dates[i] if i < len(report_dates) else ""
        rdate = date.fromisoformat(report_date_str) if report_date_str else None

        filings.append({
            "accession_number": accessions[i],
            "form_type": form,
            "filing_date": fdate,
            "report_date": rdate,
            "primary_document": primary_docs[i] if i < len(primary_docs) else "",
        })

    return filings


def build_document_url(cik: str, accession_number: str, primary_document: str) -> str:
    """Build the full URL for a filing document on EDGAR."""
    accession_path = accession_number.replace("-", "")
    return ARCHIVES_URL.format(cik=cik, accession_path=f"{accession_path}/{primary_document}")


async def download_filing(cik: str, accession_number: str, primary_document: str) -> str:
    """Download the full text of a filing document."""
    url = build_document_url(cik, accession_number, primary_document)
    logger.info(f"Downloading filing: {url}")
    return await edgar_client.get_text(url)


async def get_company_filings(
    cik: str,
    form_types: set[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict]:
    """Fetch and parse filing list for a company."""
    metadata = await get_filing_metadata(cik)
    return parse_filing_list(metadata, form_types, start_date, end_date)
