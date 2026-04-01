from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lexdrift.db.models import Company, Filing
from lexdrift.db.session import get_db
from lexdrift.edgar.filings import build_document_url, get_company_filings
from lexdrift.edgar.tickers import lookup_ticker, search_companies

router = APIRouter(prefix="/companies", tags=["companies"])


@router.get("")
async def search(
    q: str = Query(..., min_length=1, description="Search by ticker or company name"),
    limit: int = Query(20, ge=1, le=100),
):
    results = await search_companies(q, limit=limit)
    return {"data": results, "total": len(results)}


@router.get("/{ticker}")
async def get_company(ticker: str, db: AsyncSession = Depends(get_db)):
    info = await lookup_ticker(ticker)
    if not info:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    # Check if company exists in our DB
    stmt = select(Company).where(Company.cik == info["cik"])
    result = await db.execute(stmt)
    company = result.scalar_one_or_none()

    return {
        "cik": info["cik"],
        "ticker": info["ticker"],
        "name": info["name"],
        "in_database": company is not None,
        "id": company.id if company else None,
    }


@router.get("/{ticker}/filings")
async def list_filings(
    ticker: str,
    form_type: str | None = Query(None, description="Filter by form type (10-K, 10-Q, 8-K)"),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    info = await lookup_ticker(ticker)
    if not info:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    form_types = {form_type} if form_type else None
    filings = await get_company_filings(
        info["cik"], form_types=form_types, start_date=start_date, end_date=end_date
    )

    # Add document URLs
    for f in filings:
        if f.get("primary_document"):
            f["document_url"] = build_document_url(
                info["cik"], f["accession_number"], f["primary_document"]
            )

    total = len(filings)
    filings = filings[offset : offset + limit]

    # Check which filings are already in our DB
    accession_numbers = [f["accession_number"] for f in filings]
    if accession_numbers:
        stmt = select(Filing.accession_number, Filing.status).where(
            Filing.accession_number.in_(accession_numbers)
        )
        result = await db.execute(stmt)
        db_filings = {row[0]: row[1] for row in result.all()}
        for f in filings:
            f["db_status"] = db_filings.get(f["accession_number"])

    # Serialize dates to strings
    for f in filings:
        if f.get("filing_date"):
            f["filing_date"] = f["filing_date"].isoformat()
        if f.get("report_date"):
            f["report_date"] = f["report_date"].isoformat()

    return {"data": filings, "total": total}
