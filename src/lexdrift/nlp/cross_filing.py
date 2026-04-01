"""Cross-filing intelligence — connects dots across companies' SEC filings.

Detects sector-wide trends, risk language propagation between companies,
and divergent filers whose disclosure drift deviates from their peer group.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from lexdrift.db.models import Company, DriftScore, Filing, KeyPhrase

logger = logging.getLogger(__name__)


@dataclass
class CrossFilingSignal:
    """A signal detected across multiple companies' filings."""
    signal_type: str  # "risk_propagation", "sector_trend", "divergence"
    title: str
    description: str
    companies_involved: list[str]  # tickers
    first_appeared: str  # date
    propagation_lag: int  # quarters between appearances
    significance: float  # 0.0-1.0


@dataclass
class MarketIntelligence:
    """Synthesised market-level cross-filing report."""
    sector_trends: list[CrossFilingSignal]
    risk_propagations: list[CrossFilingSignal]
    divergent_companies: list[CrossFilingSignal]
    overall_market_drift_level: float
    date: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quarters_between(d1: date, d2: date) -> int:
    """Approximate number of quarters between two dates."""
    delta_days = abs((d2 - d1).days)
    return max(1, round(delta_days / 91))


def _quarter_key(d: date) -> str:
    """Return a 'YYYY-QN' string for grouping by quarter."""
    q = (d.month - 1) // 3 + 1
    return f"{d.year}-Q{q}"


# ---------------------------------------------------------------------------
# detect_sector_trends
# ---------------------------------------------------------------------------

async def detect_sector_trends(db_session: AsyncSession) -> list[CrossFilingSignal]:
    """Find sector-wide trends: 3+ companies with rising drift in the same quarter."""

    # Fetch recent drift scores with company info
    stmt = (
        select(
            DriftScore.company_id,
            DriftScore.cosine_distance,
            DriftScore.section_type,
            Filing.filing_date,
            Company.ticker,
            Company.sic_code,
            Company.name,
        )
        .join(Filing, DriftScore.filing_id == Filing.id)
        .join(Company, DriftScore.company_id == Company.id)
        .where(DriftScore.cosine_distance.isnot(None))
        .order_by(Filing.filing_date)
    )
    result = await db_session.execute(stmt)
    rows = result.all()

    if not rows:
        return []

    # Group by (sic_code or "all") and quarter, tracking average drift per company
    # Structure: group_key -> quarter -> {company_id: avg_drift}
    group_quarter_drift: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    company_info: dict[int, dict] = {}

    for row in rows:
        group_key = row.sic_code if row.sic_code else "unknown"
        qk = _quarter_key(row.filing_date)
        group_quarter_drift[group_key][qk][row.company_id].append(row.cosine_distance)
        company_info[row.company_id] = {
            "ticker": row.ticker or f"CID-{row.company_id}",
            "name": row.name,
        }

    signals: list[CrossFilingSignal] = []

    for group_key, quarters in group_quarter_drift.items():
        sorted_quarters = sorted(quarters.keys())
        if len(sorted_quarters) < 2:
            continue

        for i in range(1, len(sorted_quarters)):
            prev_q = sorted_quarters[i - 1]
            curr_q = sorted_quarters[i]

            # Compute average drift per company in each quarter
            prev_avgs: dict[int, float] = {}
            for cid, vals in quarters[prev_q].items():
                prev_avgs[cid] = sum(vals) / len(vals)

            curr_avgs: dict[int, float] = {}
            for cid, vals in quarters[curr_q].items():
                curr_avgs[cid] = sum(vals) / len(vals)

            # Find companies present in both quarters with increased drift
            increasing = []
            for cid in set(prev_avgs) & set(curr_avgs):
                if curr_avgs[cid] > prev_avgs[cid]:
                    increasing.append(cid)

            if len(increasing) >= 3:
                tickers = [company_info[cid]["ticker"] for cid in increasing]
                avg_increase = sum(
                    curr_avgs[cid] - prev_avgs[cid] for cid in increasing
                ) / len(increasing)
                significance = min(1.0, avg_increase * 10)

                signals.append(CrossFilingSignal(
                    signal_type="sector_trend",
                    title=f"Sector-wide drift increase in {curr_q} (SIC {group_key})",
                    description=(
                        f"{len(increasing)} companies in SIC group {group_key} showed "
                        f"increased disclosure drift in {curr_q} compared to {prev_q}. "
                        f"Average drift increase: {avg_increase:.4f}."
                    ),
                    companies_involved=tickers,
                    first_appeared=curr_q,
                    propagation_lag=0,
                    significance=round(significance, 3),
                ))

    # Sort by significance descending
    signals.sort(key=lambda s: s.significance, reverse=True)
    return signals


# ---------------------------------------------------------------------------
# detect_risk_propagation
# ---------------------------------------------------------------------------

async def detect_risk_propagation(
    db_session: AsyncSession,
) -> list[CrossFilingSignal]:
    """Find phrases that propagated from one company to others with a lag."""

    # Fetch key phrases with filing dates and company tickers
    stmt = (
        select(
            KeyPhrase.phrase,
            KeyPhrase.status,
            Filing.filing_date,
            Filing.company_id,
            Company.ticker,
            Company.name,
        )
        .join(Filing, KeyPhrase.filing_id == Filing.id)
        .join(Company, Filing.company_id == Company.id)
        .where(KeyPhrase.status == "added")
        .order_by(Filing.filing_date)
    )
    result = await db_session.execute(stmt)
    rows = result.all()

    if not rows:
        return []

    # Group by phrase -> list of (filing_date, company_id, ticker)
    phrase_appearances: dict[str, list[tuple[date, int, str]]] = defaultdict(list)
    for row in rows:
        phrase_appearances[row.phrase].append(
            (row.filing_date, row.company_id, row.ticker or f"CID-{row.company_id}")
        )

    signals: list[CrossFilingSignal] = []

    for phrase, appearances in phrase_appearances.items():
        # Only consider phrases that appeared in 2+ different companies
        unique_companies = {a[1] for a in appearances}
        if len(unique_companies) < 2:
            continue

        # Sort by date; the earliest appearance is the "first mover"
        appearances.sort(key=lambda a: a[0])
        first_date, first_cid, first_ticker = appearances[0]

        # Find subsequent appearances in OTHER companies
        followers = []
        seen_companies = {first_cid}
        for app_date, cid, ticker in appearances[1:]:
            if cid not in seen_companies:
                seen_companies.add(cid)
                lag = _quarters_between(first_date, app_date)
                if 1 <= lag <= 4:  # 1-4 quarter lag is meaningful propagation
                    followers.append((ticker, lag, app_date))

        if not followers:
            continue

        all_tickers = [first_ticker] + [f[0] for f in followers]
        avg_lag = sum(f[1] for f in followers) / len(followers)
        # More companies + shorter lag = higher significance
        significance = min(1.0, len(followers) * 0.2 + (1.0 / avg_lag) * 0.3)

        signals.append(CrossFilingSignal(
            signal_type="risk_propagation",
            title=f'"{phrase}" propagated from {first_ticker}',
            description=(
                f'The phrase "{phrase}" first appeared in {first_ticker}\'s filing '
                f"on {first_date.isoformat()}, then appeared in "
                f"{', '.join(f[0] for f in followers)} with an average lag of "
                f"{avg_lag:.1f} quarters."
            ),
            companies_involved=all_tickers,
            first_appeared=first_date.isoformat(),
            propagation_lag=round(avg_lag),
            significance=round(significance, 3),
        ))

    signals.sort(key=lambda s: s.significance, reverse=True)
    return signals


# ---------------------------------------------------------------------------
# detect_divergence
# ---------------------------------------------------------------------------

async def detect_divergence(
    db_session: AsyncSession,
    company_id: int,
) -> list[CrossFilingSignal]:
    """Detect if a company's drift diverges significantly from all others."""

    # Get the latest drift score per company
    # Sub-query: max filing_date per company
    latest_filing_sub = (
        select(
            DriftScore.company_id,
            func.max(Filing.filing_date).label("max_date"),
        )
        .join(Filing, DriftScore.filing_id == Filing.id)
        .where(DriftScore.cosine_distance.isnot(None))
        .group_by(DriftScore.company_id)
        .subquery()
    )

    stmt = (
        select(
            DriftScore.company_id,
            DriftScore.cosine_distance,
            Company.ticker,
            Company.name,
        )
        .join(Filing, DriftScore.filing_id == Filing.id)
        .join(Company, DriftScore.company_id == Company.id)
        .join(
            latest_filing_sub,
            (DriftScore.company_id == latest_filing_sub.c.company_id)
            & (Filing.filing_date == latest_filing_sub.c.max_date),
        )
        .where(DriftScore.cosine_distance.isnot(None))
    )
    result = await db_session.execute(stmt)
    rows = result.all()

    if len(rows) < 3:
        return []

    # Compute per-company average of latest drift scores (may have multiple sections)
    company_drifts: dict[int, list[float]] = defaultdict(list)
    company_meta: dict[int, dict] = {}
    for row in rows:
        company_drifts[row.company_id].append(row.cosine_distance)
        company_meta[row.company_id] = {
            "ticker": row.ticker or f"CID-{row.company_id}",
            "name": row.name,
        }

    company_avg: dict[int, float] = {
        cid: sum(vals) / len(vals) for cid, vals in company_drifts.items()
    }

    if company_id not in company_avg:
        return []

    # Compute group stats (excluding target company)
    other_avgs = [v for cid, v in company_avg.items() if cid != company_id]
    if not other_avgs:
        return []

    group_mean = sum(other_avgs) / len(other_avgs)
    variance = sum((v - group_mean) ** 2 for v in other_avgs) / len(other_avgs)
    group_std = variance ** 0.5

    if group_std == 0:
        return []

    target_avg = company_avg[company_id]
    z_score = (target_avg - group_mean) / group_std
    target_ticker = company_meta[company_id]["ticker"]

    signals: list[CrossFilingSignal] = []

    if abs(z_score) > 2.0:
        if z_score > 0:
            direction = "UP"
            interpretation = (
                f"{target_ticker}'s disclosure drift ({target_avg:.4f}) is "
                f"{z_score:.1f} standard deviations above the peer group average "
                f"({group_mean:.4f}). This level of revision activity may signal "
                f"material changes in the company's risk profile."
            )
        else:
            direction = "DOWN"
            interpretation = (
                f"{target_ticker}'s disclosure drift ({target_avg:.4f}) is "
                f"{abs(z_score):.1f} standard deviations below the peer group average "
                f"({group_mean:.4f}). Unusually low revision activity may indicate "
                f"the company is not adequately updating its disclosures."
            )

        significance = min(1.0, abs(z_score) / 5.0)

        signals.append(CrossFilingSignal(
            signal_type="divergence",
            title=f"{target_ticker} diverging {direction} from peers",
            description=interpretation,
            companies_involved=[target_ticker],
            first_appeared=date.today().isoformat(),
            propagation_lag=0,
            significance=round(significance, 3),
        ))

    return signals


# ---------------------------------------------------------------------------
# generate_market_intelligence
# ---------------------------------------------------------------------------

async def generate_market_intelligence(
    db_session: AsyncSession,
) -> MarketIntelligence:
    """Synthesise all cross-filing signals into a market-level report."""

    sector_trends = await detect_sector_trends(db_session)
    risk_propagations = await detect_risk_propagation(db_session)

    # Detect divergence for every company in the database
    stmt = select(Company.id)
    result = await db_session.execute(stmt)
    company_ids = [row[0] for row in result.all()]

    divergent: list[CrossFilingSignal] = []
    for cid in company_ids:
        divergent.extend(await detect_divergence(db_session, cid))

    # Overall market drift level: average of all recent drift scores
    drift_stmt = (
        select(func.avg(DriftScore.cosine_distance))
        .where(DriftScore.cosine_distance.isnot(None))
    )
    result = await db_session.execute(drift_stmt)
    avg_drift = result.scalar() or 0.0

    return MarketIntelligence(
        sector_trends=sector_trends,
        risk_propagations=risk_propagations,
        divergent_companies=divergent,
        overall_market_drift_level=round(float(avg_drift), 6),
        date=date.today().isoformat(),
    )
