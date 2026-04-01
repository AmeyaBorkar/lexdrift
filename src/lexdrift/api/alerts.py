from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from lexdrift.db.models import Alert, Company, Filing, Watchlist, WatchlistCompany
from lexdrift.db.session import get_db
from lexdrift.edgar.tickers import lookup_ticker

router = APIRouter(tags=["alerts & watchlists"])


# --- Watchlists ---

class WatchlistCreate(BaseModel):
    name: str


class WatchlistAddCompany(BaseModel):
    ticker: str


@router.post("/watchlists")
async def create_watchlist(body: WatchlistCreate, db: AsyncSession = Depends(get_db)):
    watchlist = Watchlist(name=body.name)
    db.add(watchlist)
    await db.commit()
    await db.refresh(watchlist)
    return {"id": watchlist.id, "name": watchlist.name}


@router.get("/watchlists")
async def list_watchlists(db: AsyncSession = Depends(get_db)):
    stmt = select(Watchlist)
    result = await db.execute(stmt)
    watchlists = result.scalars().all()
    return {
        "data": [{"id": w.id, "name": w.name, "created_at": w.created_at.isoformat()} for w in watchlists],
        "total": len(watchlists),
    }


@router.post("/watchlists/{watchlist_id}/companies")
async def add_company_to_watchlist(
    watchlist_id: int,
    body: WatchlistAddCompany,
    db: AsyncSession = Depends(get_db),
):
    # Verify watchlist exists
    stmt = select(Watchlist).where(Watchlist.id == watchlist_id)
    result = await db.execute(stmt)
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Watchlist not found")

    # Look up ticker
    info = await lookup_ticker(body.ticker)
    if not info:
        raise HTTPException(status_code=404, detail=f"Ticker '{body.ticker}' not found")

    # Ensure company in DB
    stmt = select(Company).where(Company.cik == info["cik"])
    result = await db.execute(stmt)
    company = result.scalar_one_or_none()
    if not company:
        company = Company(cik=info["cik"], ticker=info["ticker"], name=info["name"])
        db.add(company)
        await db.flush()

    # Add to watchlist
    wc = WatchlistCompany(watchlist_id=watchlist_id, company_id=company.id)
    db.add(wc)
    await db.commit()
    return {"watchlist_id": watchlist_id, "ticker": info["ticker"], "company_id": company.id}


@router.delete("/watchlists/{watchlist_id}/companies/{ticker}")
async def remove_company_from_watchlist(
    watchlist_id: int,
    ticker: str,
    db: AsyncSession = Depends(get_db),
):
    info = await lookup_ticker(ticker)
    if not info:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    stmt = select(Company).where(Company.cik == info["cik"])
    result = await db.execute(stmt)
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail="Company not in database")

    stmt = select(WatchlistCompany).where(
        WatchlistCompany.watchlist_id == watchlist_id,
        WatchlistCompany.company_id == company.id,
    )
    result = await db.execute(stmt)
    wc = result.scalar_one_or_none()
    if not wc:
        raise HTTPException(status_code=404, detail="Company not in watchlist")

    await db.delete(wc)
    await db.commit()
    return {"removed": True}


# --- Alerts ---

@router.get("/alerts")
async def list_alerts(
    watchlist_id: int | None = Query(None),
    unread: bool | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    query = (
        select(
            Alert.id,
            Alert.alert_type,
            Alert.severity,
            Alert.message,
            Alert.metadata_,
            Alert.read,
            Alert.created_at,
            Company.ticker,
            Company.name,
            Filing.filing_date,
        )
        .join(Company, Alert.company_id == Company.id)
        .join(Filing, Alert.filing_id == Filing.id)
    )

    if watchlist_id is not None:
        query = query.join(
            WatchlistCompany,
            (WatchlistCompany.company_id == Alert.company_id)
            & (WatchlistCompany.watchlist_id == watchlist_id),
        )

    if unread is not None:
        query = query.where(Alert.read == (not unread))

    query = query.order_by(Alert.created_at.desc()).offset(offset).limit(limit)

    result = await db.execute(query)
    rows = result.all()

    data = []
    for row in rows:
        data.append({
            "id": row.id,
            "alert_type": row.alert_type,
            "severity": row.severity,
            "message": row.message,
            "metadata": row[4],  # metadata_
            "read": row.read,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "ticker": row.ticker,
            "company_name": row.name,
            "filing_date": row.filing_date.isoformat() if row.filing_date else None,
        })

    return {"data": data, "total": len(data)}


@router.put("/alerts/{alert_id}/read")
async def mark_alert_read(alert_id: int, db: AsyncSession = Depends(get_db)):
    stmt = update(Alert).where(Alert.id == alert_id).values(read=True)
    result = await db.execute(stmt)
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Alert not found")
    await db.commit()
    return {"id": alert_id, "read": True}
