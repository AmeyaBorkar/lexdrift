import re
from pathlib import Path as FilePath

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lexdrift.db.models import Company, Filing, Section
from lexdrift.db.session import get_db
from lexdrift.edgar.filings import build_document_url, download_filing, get_filing_metadata, parse_filing_list
from lexdrift.edgar.parser import parse_filing
from lexdrift.edgar.tickers import lookup_ticker

VALID_FORM_TYPES = {"10-K", "10-Q", "8-K", "10-K/A", "10-Q/A"}
_TICKER_RE = re.compile(r"^[A-Za-z0-9]{1,10}$")

router = APIRouter(prefix="/filings", tags=["filings"])


@router.post("/ingest/{ticker}")
async def ingest_filings(
    ticker: str,
    form_type: str = Query("10-K", description="SEC form type"),
    limit: int = Query(5, ge=1, le=20, description="Max filings to ingest (1-20)"),
    db: AsyncSession = Depends(get_db),
):
    """Ingest filings for a company: download, parse, and store sections."""
    if not _TICKER_RE.match(ticker):
        raise HTTPException(status_code=400, detail="Ticker must be alphanumeric and at most 10 characters")
    if form_type not in VALID_FORM_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid form_type '{form_type}'. Must be one of: {', '.join(sorted(VALID_FORM_TYPES))}",
        )

    info = await lookup_ticker(ticker)
    if not info:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    # Ensure company exists in DB
    stmt = select(Company).where(Company.cik == info["cik"])
    result = await db.execute(stmt)
    company = result.scalar_one_or_none()
    if not company:
        company = Company(cik=info["cik"], ticker=info["ticker"], name=info["name"])
        db.add(company)
        await db.flush()

    # Fetch filing list from EDGAR
    metadata = await get_filing_metadata(info["cik"])
    filing_list = parse_filing_list(metadata, form_types={form_type})[:limit]

    ingested = []
    for f_meta in filing_list:
        # Skip if already in DB
        stmt = select(Filing).where(Filing.accession_number == f_meta["accession_number"])
        result = await db.execute(stmt)
        if result.scalar_one_or_none():
            continue

        # Download and parse
        doc_url = build_document_url(info["cik"], f_meta["accession_number"], f_meta["primary_document"])
        try:
            html = await download_filing(info["cik"], f_meta["accession_number"], f_meta["primary_document"])
        except Exception as e:
            ingested.append({"accession": f_meta["accession_number"], "status": "error", "error": str(e)})
            continue

        # Save raw HTML to disk instead of storing in DB
        filing_dir = FilePath("data/filings") / info["cik"]
        filing_dir.mkdir(parents=True, exist_ok=True)
        html_path = filing_dir / f"{f_meta['accession_number']}.html"
        html_path.write_text(html, encoding="utf-8")

        # Create filing record (raw_text=None to keep DB small)
        filing = Filing(
            company_id=company.id,
            accession_number=f_meta["accession_number"],
            form_type=f_meta["form_type"],
            filing_date=f_meta["filing_date"],
            report_date=f_meta["report_date"],
            document_url=str(html_path),
            raw_text=None,
            status="parsed",
        )
        db.add(filing)
        await db.flush()

        # Parse sections
        sections = parse_filing(html, f_meta["form_type"])
        for section_type, section_text in sections.items():
            section = Section(
                filing_id=filing.id,
                section_type=section_type,
                section_text=section_text,
                word_count=len(section_text.split()),
            )
            db.add(section)

        ingested.append({
            "accession": f_meta["accession_number"],
            "form_type": f_meta["form_type"],
            "filing_date": f_meta["filing_date"].isoformat() if f_meta["filing_date"] else None,
            "sections": list(sections.keys()),
            "status": "parsed",
        })

    await db.commit()
    return {"ticker": ticker, "ingested": ingested, "count": len(ingested)}


@router.get("/{filing_id}")
async def get_filing(filing_id: int, db: AsyncSession = Depends(get_db)):
    """Get filing metadata and section list."""
    stmt = select(Filing).where(Filing.id == filing_id)
    result = await db.execute(stmt)
    filing = result.scalar_one_or_none()
    if not filing:
        raise HTTPException(status_code=404, detail="Filing not found")

    # Get sections
    stmt = select(Section.section_type, Section.word_count).where(Section.filing_id == filing_id)
    result = await db.execute(stmt)
    sections = [{"type": row[0], "word_count": row[1]} for row in result.all()]

    return {
        "id": filing.id,
        "accession_number": filing.accession_number,
        "form_type": filing.form_type,
        "filing_date": filing.filing_date.isoformat() if filing.filing_date else None,
        "report_date": filing.report_date.isoformat() if filing.report_date else None,
        "status": filing.status,
        "sections": sections,
    }


@router.get("/{filing_id}/sections/{section_type}")
async def get_section(filing_id: int, section_type: str, db: AsyncSession = Depends(get_db)):
    """Get full section text and word count."""
    stmt = select(Section).where(
        Section.filing_id == filing_id, Section.section_type == section_type
    )
    result = await db.execute(stmt)
    section = result.scalar_one_or_none()
    if not section:
        raise HTTPException(status_code=404, detail="Section not found")

    return {
        "filing_id": filing_id,
        "section_type": section.section_type,
        "word_count": section.word_count,
        "text": section.section_text,
    }
