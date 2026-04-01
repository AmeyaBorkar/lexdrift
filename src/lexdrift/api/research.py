"""Research API -- novel NLP analysis endpoints.

Provides obfuscation detection, information-theoretic entropy analysis,
semantic kinematics (velocity/acceleration/momentum), and a combined
overview endpoint for at-a-glance company assessment.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

import numpy as np

from lexdrift.db.models import Company, DriftScore, Filing, Section
from lexdrift.db.session import get_db
from lexdrift.edgar.tickers import lookup_ticker
from lexdrift.nlp.anomaly import detect_anomaly, detect_trends
from lexdrift.nlp.contagion import build_risk_graph, compute_systemic_risk
from lexdrift.nlp.intelligence import generate_intelligence, _get_sync_session, CompanyIntelligence
from lexdrift.nlp.cross_filing import (
    detect_divergence,
    detect_risk_propagation,
    generate_market_intelligence,
)
from lexdrift.nlp.embeddings import bytes_to_embedding
from lexdrift.nlp.entropy import compute_filing_entropy
from lexdrift.nlp.latent_space import build_latent_space
from lexdrift.nlp.narrative import generate_market_narrative
from lexdrift.nlp.obfuscation import detect_obfuscation
from lexdrift.nlp.velocity import compute_semantic_kinematics

router = APIRouter(prefix="/research", tags=["research"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_filing(filing_id: int, db: AsyncSession) -> Filing:
    """Fetch a filing by ID or raise 404."""
    stmt = select(Filing).where(Filing.id == filing_id)
    result = await db.execute(stmt)
    filing = result.scalar_one_or_none()
    if not filing:
        raise HTTPException(status_code=404, detail="Filing not found")
    return filing


async def _get_previous_filing(filing: Filing, db: AsyncSession) -> Filing:
    """Find the previous filing of the same company and form type, or raise 404."""
    stmt = (
        select(Filing)
        .where(
            Filing.company_id == filing.company_id,
            Filing.form_type == filing.form_type,
            Filing.filing_date < filing.filing_date,
        )
        .order_by(desc(Filing.filing_date))
        .limit(1)
    )
    result = await db.execute(stmt)
    prev_filing = result.scalar_one_or_none()
    if not prev_filing:
        raise HTTPException(
            status_code=404,
            detail="No previous filing found for comparison",
        )
    return prev_filing


async def _get_section_pairs(
    filing_id: int,
    prev_filing_id: int,
    db: AsyncSession,
) -> list[tuple[str, Section, Section]]:
    """Return matched (section_type, prev_section, curr_section) triples.

    Only includes sections that exist in both filings and have non-empty text.
    """
    stmt = select(Section).where(Section.filing_id == filing_id)
    result = await db.execute(stmt)
    curr_sections = {s.section_type: s for s in result.scalars().all()}

    stmt = select(Section).where(Section.filing_id == prev_filing_id)
    result = await db.execute(stmt)
    prev_sections = {s.section_type: s for s in result.scalars().all()}

    pairs = []
    for section_type, curr in curr_sections.items():
        prev = prev_sections.get(section_type)
        if prev and prev.section_text and curr.section_text:
            pairs.append((section_type, prev, curr))
    return pairs


async def _resolve_company(ticker: str, db: AsyncSession) -> Company:
    """Look up a ticker via EDGAR and then find the Company row, or raise 404."""
    info = await lookup_ticker(ticker)
    if not info:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    stmt = select(Company).where(Company.cik == info["cik"])
    result = await db.execute(stmt)
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(
            status_code=404,
            detail="Company not in database. Ingest filings first.",
        )
    return company


async def _build_drift_history_by_section(
    company: Company,
    db: AsyncSession,
) -> dict[str, list[dict]]:
    """Query all DriftScore records for a company grouped by section_type.

    Each entry in the returned lists has ``filing_date`` and ``cosine_distance``
    keys, suitable for passing to ``compute_semantic_kinematics``.
    """
    stmt = (
        select(
            DriftScore.section_type,
            DriftScore.cosine_distance,
            Filing.filing_date,
        )
        .join(Filing, DriftScore.filing_id == Filing.id)
        .where(DriftScore.company_id == company.id)
        .order_by(Filing.filing_date)
    )
    result = await db.execute(stmt)
    rows = result.all()

    by_section: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.cosine_distance is not None and row.filing_date is not None:
            by_section[row.section_type].append({
                "filing_date": row.filing_date.isoformat(),
                "cosine_distance": row.cosine_distance,
            })
    return dict(by_section)


# ---------------------------------------------------------------------------
# POST /research/obfuscation/{filing_id}
# ---------------------------------------------------------------------------

@router.post("/obfuscation/{filing_id}")
async def analyze_obfuscation(
    filing_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Detect adversarial obfuscation between a filing and its predecessor.

    For each matching section, computes information density change, specificity
    change, readability shift, and euphemism detection -- fused into a single
    obfuscation score.
    """
    filing = await _get_filing(filing_id, db)
    prev_filing = await _get_previous_filing(filing, db)
    pairs = await _get_section_pairs(filing.id, prev_filing.id, db)

    if not pairs:
        raise HTTPException(
            status_code=404,
            detail="No matching sections with text found between the two filings",
        )

    results = []
    for section_type, prev_section, curr_section in pairs:
        score = detect_obfuscation(prev_section.section_text, curr_section.section_text)
        results.append({
            "section_type": section_type,
            **dataclasses.asdict(score),
        })

    return {
        "filing_id": filing.id,
        "prev_filing_id": prev_filing.id,
        "form_type": filing.form_type,
        "sections_analyzed": len(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# POST /research/entropy/{filing_id}
# ---------------------------------------------------------------------------

@router.post("/entropy/{filing_id}")
async def analyze_entropy(
    filing_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Run information-theoretic entropy analysis between a filing and its predecessor.

    For each matching section, computes Shannon entropy, cross-entropy,
    conditional entropy, KL divergence, and a composite novelty score.
    """
    filing = await _get_filing(filing_id, db)
    prev_filing = await _get_previous_filing(filing, db)
    pairs = await _get_section_pairs(filing.id, prev_filing.id, db)

    if not pairs:
        raise HTTPException(
            status_code=404,
            detail="No matching sections with text found between the two filings",
        )

    results = []
    for section_type, prev_section, curr_section in pairs:
        analysis = compute_filing_entropy(prev_section.section_text, curr_section.section_text)
        results.append({
            "section_type": section_type,
            **dataclasses.asdict(analysis),
        })

    return {
        "filing_id": filing.id,
        "prev_filing_id": prev_filing.id,
        "form_type": filing.form_type,
        "sections_analyzed": len(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# GET /research/kinematics/{ticker}
# ---------------------------------------------------------------------------

@router.get("/kinematics/{ticker}")
async def get_kinematics(
    ticker: str,
    db: AsyncSession = Depends(get_db),
):
    """Compute semantic velocity, acceleration, momentum, and phase for a company.

    Groups drift history by section type and runs kinematic analysis on each,
    treating the drift time series as a position signal and computing its
    first three derivatives.
    """
    company = await _resolve_company(ticker, db)
    by_section = await _build_drift_history_by_section(company, db)

    if not by_section:
        raise HTTPException(
            status_code=404,
            detail="No drift scores found for this company. Run analysis first.",
        )

    results = {}
    for section_type, history in by_section.items():
        if len(history) < 2:
            results[section_type] = {
                "error": "Insufficient data points (need at least 2)",
                "periods_available": len(history),
            }
            continue

        kinematics = compute_semantic_kinematics(history)
        results[section_type] = {
            "latest_velocity": kinematics.latest_velocity,
            "latest_acceleration": kinematics.latest_acceleration,
            "latest_momentum": kinematics.latest_momentum,
            "phase": kinematics.phase,
            "velocity_mean": kinematics.velocity_mean,
            "velocity_std": kinematics.velocity_std,
            "max_velocity": kinematics.max_velocity,
            "periods_analyzed": kinematics.periods_analyzed,
        }

    return {
        "ticker": ticker,
        "company": company.name,
        "sections": results,
    }


# ---------------------------------------------------------------------------
# GET /research/overview/{ticker}
# ---------------------------------------------------------------------------

@router.get("/overview/{ticker}")
async def get_overview(
    ticker: str,
    db: AsyncSession = Depends(get_db),
):
    """Combined research overview for a company -- one call for the full picture.

    Returns the latest drift scores per section, anomaly status, kinematics
    phase, and any trend signals detected across the filing history.
    """
    company = await _resolve_company(ticker, db)

    # ------------------------------------------------------------------
    # 1. Latest drift scores per section
    # ------------------------------------------------------------------
    latest_stmt = (
        select(
            DriftScore.section_type,
            DriftScore.cosine_distance,
            DriftScore.jaccard_distance,
            DriftScore.sentiment_delta,
            DriftScore.added_words,
            DriftScore.removed_words,
            DriftScore.computed_at,
            Filing.filing_date,
            Filing.form_type,
        )
        .join(Filing, DriftScore.filing_id == Filing.id)
        .where(DriftScore.company_id == company.id)
        .order_by(desc(DriftScore.computed_at))
    )
    result = await db.execute(latest_stmt)
    all_rows = result.all()

    if not all_rows:
        raise HTTPException(
            status_code=404,
            detail="No drift scores found for this company. Run analysis first.",
        )

    # Pick the most recent score per section type
    latest_by_section: dict[str, dict] = {}
    for row in all_rows:
        if row.section_type not in latest_by_section:
            latest_by_section[row.section_type] = {
                "cosine_distance": round(row.cosine_distance, 4) if row.cosine_distance else None,
                "jaccard_distance": round(row.jaccard_distance, 4) if row.jaccard_distance else None,
                "sentiment_delta": row.sentiment_delta,
                "added_words": row.added_words,
                "removed_words": row.removed_words,
                "filing_date": row.filing_date.isoformat() if row.filing_date else None,
                "form_type": row.form_type,
            }

    # ------------------------------------------------------------------
    # 2. Anomaly detection per section
    # ------------------------------------------------------------------
    # Build full history grouped by section
    history_by_section: dict[str, list[float]] = defaultdict(list)
    drift_history_all: list[float] = []
    sentiment_history_all: list[dict] = []

    for row in all_rows:
        if row.cosine_distance is not None:
            history_by_section[row.section_type].append(row.cosine_distance)
            drift_history_all.append(row.cosine_distance)
        if row.sentiment_delta:
            sentiment_history_all.append(row.sentiment_delta)

    anomaly_by_section: dict[str, dict] = {}
    for section_type, latest in latest_by_section.items():
        current_drift = latest["cosine_distance"]
        if current_drift is None:
            continue
        company_history = history_by_section.get(section_type, [])
        # Exclude the current value from history for a fair comparison
        comparison_history = company_history[1:] if len(company_history) > 1 else []
        anomaly = detect_anomaly(
            current_drift=current_drift,
            company_history=comparison_history,
        )
        anomaly_by_section[section_type] = {
            "company_z_score": anomaly.company_z_score,
            "is_anomalous": anomaly.is_anomalous,
            "anomaly_level": anomaly.anomaly_level,
            "company_mean": anomaly.company_mean,
            "company_stddev": anomaly.company_stddev,
        }

    # ------------------------------------------------------------------
    # 3. Kinematics phase per section
    # ------------------------------------------------------------------
    by_section_history = await _build_drift_history_by_section(company, db)

    kinematics_by_section: dict[str, dict] = {}
    for section_type, history in by_section_history.items():
        if len(history) < 2:
            kinematics_by_section[section_type] = {"phase": "insufficient_data"}
            continue

        kinematics = compute_semantic_kinematics(history)
        kinematics_by_section[section_type] = {
            "phase": kinematics.phase,
            "latest_velocity": kinematics.latest_velocity,
            "latest_acceleration": kinematics.latest_acceleration,
            "latest_momentum": kinematics.latest_momentum,
        }

    # ------------------------------------------------------------------
    # 4. Trend signals (aggregate across all sections)
    # ------------------------------------------------------------------
    trends = detect_trends(
        drift_history=drift_history_all,
        sentiment_history=sentiment_history_all if sentiment_history_all else None,
    )

    return {
        "ticker": ticker,
        "company": company.name,
        "latest_drift": latest_by_section,
        "anomaly": anomaly_by_section,
        "kinematics": kinematics_by_section,
        "trends": trends,
    }


# ---------------------------------------------------------------------------
# GET /research/contagion/{ticker}
# ---------------------------------------------------------------------------

@router.get("/contagion/{ticker}")
async def get_contagion(
    ticker: str,
    db: AsyncSession = Depends(get_db),
):
    """Build a risk contagion graph and compute systemic risk metrics.

    Retrieves the latest risk_factors section text for ALL companies in the
    database, builds an inter-company similarity graph, and returns graph
    statistics, systemic risk metrics, and this company's centrality and
    connections.

    NOTE: This is an expensive operation that encodes all companies' risk
    factor sections. In production this should be cached or pre-computed
    on a schedule.
    """
    company = await _resolve_company(ticker, db)

    # Fetch the latest risk_factors section text for every company.
    # We join Section -> Filing and pick the most recent filing per company
    # that has a risk_factors section with text.
    stmt = (
        select(
            Section.section_text,
            Filing.company_id,
            Filing.filing_date,
        )
        .join(Filing, Section.filing_id == Filing.id)
        .where(
            Section.section_type == "risk_factors",
            Section.section_text.isnot(None),
        )
        .order_by(desc(Filing.filing_date))
    )
    result = await db.execute(stmt)
    rows = result.all()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No risk factor sections found in database. Ingest filings first.",
        )

    # Keep only the latest section per company
    company_sections: dict[int, str] = {}
    for row in rows:
        if row.company_id not in company_sections:
            company_sections[row.company_id] = row.section_text

    if company.id not in company_sections:
        raise HTTPException(
            status_code=404,
            detail="No risk factor section found for this company.",
        )

    # Build the risk contagion graph
    graph = build_risk_graph(company_sections)
    systemic = compute_systemic_risk(graph)

    # Extract this company's centrality and connections
    company_centrality = systemic.betweenness_centrality.get(company.id, 0.0)
    company_clustering = systemic.clustering_coefficients.get(company.id, 0.0)
    company_connections = []
    if company.id in graph:
        for neighbor in graph.neighbors(company.id):
            edge_data = graph.get_edge_data(company.id, neighbor)
            company_connections.append({
                "company_id": neighbor,
                "similarity": round(edge_data["weight"], 4) if edge_data else 0.0,
            })
        company_connections.sort(key=lambda x: x["similarity"], reverse=True)

    # Serialize community sets to lists for JSON
    communities_serialized = [sorted(c) for c in systemic.modularity_communities]

    return {
        "ticker": ticker,
        "company": company.name,
        "graph_stats": {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "density": systemic.density,
            "connected_components": systemic.connected_components,
            "largest_component_fraction": systemic.largest_component_fraction,
            "average_path_length": systemic.average_path_length,
        },
        "systemic_risk": {
            "risk_hubs": systemic.risk_hubs,
            "spectral_gap": systemic.spectral_gap,
            "communities": communities_serialized,
        },
        "company_metrics": {
            "betweenness_centrality": company_centrality,
            "clustering_coefficient": company_clustering,
            "is_risk_hub": company.id in systemic.risk_hubs,
            "connections": company_connections,
        },
    }


# ---------------------------------------------------------------------------
# GET /research/latent-space
# ---------------------------------------------------------------------------

@router.get("/latent-space")
async def get_latent_space(
    db: AsyncSession = Depends(get_db),
):
    """Project all section embeddings into a shared low-dimensional latent space.

    Retrieves all section embeddings from the database (sections table),
    builds a latent space projection (PCA or UMAP), and returns the 2D/3D
    projected coordinates per company+filing for visualization.

    NOTE: This is an expensive operation that processes all embeddings in the
    database. In production this should be cached or pre-computed on a
    schedule.
    """
    # Fetch all sections that have embeddings, with company_id and filing_date
    stmt = (
        select(
            Section.embedding,
            Section.section_type,
            Filing.company_id,
            Filing.filing_date,
        )
        .join(Filing, Section.filing_id == Filing.id)
        .where(Section.embedding.isnot(None))
        .order_by(Filing.filing_date)
    )
    result = await db.execute(stmt)
    rows = result.all()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No section embeddings found in database. Run analysis first.",
        )

    # Build the input list for build_latent_space
    section_embeddings = []
    for row in rows:
        try:
            embedding = bytes_to_embedding(row.embedding)
            section_embeddings.append({
                "company_id": row.company_id,
                "filing_date": row.filing_date.isoformat() if row.filing_date else "",
                "section_type": row.section_type,
                "embedding": embedding,
            })
        except Exception:
            # Skip malformed embeddings
            continue

    if not section_embeddings:
        raise HTTPException(
            status_code=404,
            detail="No valid embeddings could be decoded. Re-run analysis.",
        )

    latent = build_latent_space(section_embeddings)

    # Format the output as a list of coordinate records for visualization
    coordinates = []
    for i in range(len(latent.company_ids)):
        point = latent.points[i]
        coord = {
            "company_id": latent.company_ids[i],
            "filing_date": latent.filing_dates[i],
            "section_type": latent.section_types[i],
        }
        # Add coordinate dimensions (x, y, and optionally z)
        for dim in range(latent.n_components):
            coord[f"dim_{dim}"] = round(float(point[dim]), 6)
        coordinates.append(coord)

    return {
        "projection_method": latent.projection_method,
        "n_components": latent.n_components,
        "total_points": len(coordinates),
        "coordinates": coordinates,
    }


# ---------------------------------------------------------------------------
# GET /research/market-intelligence
# ---------------------------------------------------------------------------

@router.get("/market-intelligence")
async def get_market_intelligence(
    db: AsyncSession = Depends(get_db),
):
    """Generate a cross-filing market intelligence report.

    Synthesises sector trends, risk language propagation, and divergent
    filers into a single market-level view with an optional narrative.
    """
    import dataclasses as _dc

    intel = await generate_market_intelligence(db)
    result = _dc.asdict(intel)

    # LLM-reasoned narrative (falls back to template)
    from lexdrift.nlp.reasoning import reason_about_market
    result["narrative"] = reason_about_market(result)

    return result


# ---------------------------------------------------------------------------
# GET /research/cross-filing/{ticker}
# ---------------------------------------------------------------------------

@router.get("/cross-filing/{ticker}")
async def get_cross_filing(
    ticker: str,
    db: AsyncSession = Depends(get_db),
):
    """Return cross-filing signals relevant to a specific company.

    Includes divergence analysis (is this company drifting more or less
    than peers?) and any risk language propagation signals involving this
    company.
    """
    import dataclasses as _dc

    company = await _resolve_company(ticker, db)

    divergence_signals = await detect_divergence(db, company.id)

    # Get all propagation signals and filter to those involving this ticker
    all_propagations = await detect_risk_propagation(db)
    company_ticker = company.ticker or ticker.upper()
    relevant_propagations = [
        sig for sig in all_propagations
        if company_ticker in sig.companies_involved
    ]

    return {
        "ticker": ticker,
        "company": company.name,
        "divergence_signals": [_dc.asdict(s) for s in divergence_signals],
        "propagation_signals": [_dc.asdict(s) for s in relevant_propagations],
        "total_signals": len(divergence_signals) + len(relevant_propagations),
    }


# ---------------------------------------------------------------------------
# GET /research/intelligence/{ticker}
# ---------------------------------------------------------------------------

@router.get("/intelligence/{ticker}")
async def get_intelligence(
    ticker: str,
    db: AsyncSession = Depends(get_db),
):
    """Generate comprehensive intelligence assessment for a company.

    Synthesises all available NLP signals -- drift scores, sentiment deltas,
    phrase changes, anomaly z-scores, kinematics, and historical pattern
    matches -- into a single structured assessment with risk scoring,
    findings, predictions, comparables, and recommended actions.
    """
    # The intelligence engine uses synchronous SQLAlchemy (same as training
    # scripts) so it can be called from both API and CLI contexts.
    sync_session = _get_sync_session()
    try:
        intel = generate_intelligence(sync_session, ticker)
    finally:
        sync_session.close()

    result = dataclasses.asdict(intel)

    # Generate LLM-reasoned narrative (falls back to template if no API key)
    from lexdrift.nlp.reasoning import reason_about_company
    result["narrative"] = reason_about_company(result)

    return result
