import logging

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from lexdrift.db.models import Alert, Company, DriftScore, Filing, Section, KeyPhrase, SentenceChange
from lexdrift.db.session import get_db
from lexdrift.edgar.tickers import lookup_ticker
from lexdrift.nlp.anomaly import detect_anomaly, detect_trends
from lexdrift.nlp.diff import unified_diff, diff_stats
from lexdrift.nlp.drift import compute_drift
from lexdrift.nlp.entropy import compute_filing_entropy
from lexdrift.nlp.obfuscation import detect_obfuscation
from lexdrift.nlp.phrases import compare_keyphrases, check_watchlist_phrases

logger = logging.getLogger(__name__)

router = APIRouter(tags=["drift"])


@router.post("/filings/{filing_id}/analyze")
async def analyze_filing(
    filing_id: int = Path(..., gt=0, description="Filing ID (must be positive)"),
    force: bool = Query(False, description="Force re-analysis by deleting existing results"),
    db: AsyncSession = Depends(get_db),
):
    """Run NLP drift analysis on a filing, comparing against the previous filing."""
    # Get current filing and its company
    stmt = select(Filing).where(Filing.id == filing_id)
    result = await db.execute(stmt)
    filing = result.scalar_one_or_none()
    if not filing:
        raise HTTPException(status_code=404, detail="Filing not found")

    if filing.status == "analyzed":
        if not force:
            raise HTTPException(status_code=409, detail="Filing has already been analyzed")

        # Force re-analysis: delete existing results and reset status
        logger.info("Force re-analysis requested for filing %s; clearing existing results", filing_id)

        # Delete SentenceChange records (must go first due to FK on drift_scores)
        existing_drift_ids_stmt = select(DriftScore.id).where(DriftScore.filing_id == filing_id)
        existing_drift_ids = (await db.execute(existing_drift_ids_stmt)).scalars().all()
        if existing_drift_ids:
            await db.execute(
                SentenceChange.__table__.delete().where(
                    SentenceChange.drift_score_id.in_(existing_drift_ids)
                )
            )

        # Delete DriftScore records for this filing
        await db.execute(
            DriftScore.__table__.delete().where(DriftScore.filing_id == filing_id)
        )

        # Delete KeyPhrase records for this filing
        await db.execute(
            KeyPhrase.__table__.delete().where(KeyPhrase.filing_id == filing_id)
        )

        # Reset filing status to parsed
        filing.status = "parsed"
        await db.flush()

    # Find previous filing of same form type for this company
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
        return {"message": "No previous filing found for comparison", "filing_id": filing_id}

    # Get sections for both filings
    stmt = select(Section).where(Section.filing_id == filing_id)
    result = await db.execute(stmt)
    curr_sections = {s.section_type: s for s in result.scalars().all()}

    stmt = select(Section).where(Section.filing_id == prev_filing.id)
    result = await db.execute(stmt)
    prev_sections = {s.section_type: s for s in result.scalars().all()}

    # Wrap entire analysis in try/except so partial data is rolled back on failure
    try:
        # Compute drift for each matching section
        results = []
        for section_type, curr_section in curr_sections.items():
            prev_section = prev_sections.get(section_type)
            if not prev_section or not prev_section.section_text or not curr_section.section_text:
                continue

            drift = compute_drift(
                prev_section.section_text,
                curr_section.section_text,
                prev_embedding=prev_section.embedding,
                curr_embedding=curr_section.embedding,
            )

            # Update section embeddings if they were computed
            if not curr_section.embedding:
                curr_section.embedding = drift["curr_embedding_bytes"]
            if not prev_section.embedding:
                prev_section.embedding = drift["prev_embedding_bytes"]

            # Store drift score
            drift_score = DriftScore(
                company_id=filing.company_id,
                filing_id=filing.id,
                prev_filing_id=prev_filing.id,
                section_type=section_type,
                cosine_distance=drift["cosine_distance"],
                jaccard_distance=drift["jaccard_distance"],
                added_words=drift["added_words"],
                removed_words=drift["removed_words"],
                sentiment_delta=drift["sentiment_delta"],
            )
            db.add(drift_score)
            await db.flush()  # get drift_score.id for sentence changes FK

            # Store sentence-level changes
            sent = drift["sentence_changes"]
            for entry in sent["added"]:
                db.add(SentenceChange(
                    drift_score_id=drift_score.id, change_type="added",
                    sentence_text=entry["text"], sentence_index=entry["index"],
                ))
            for entry in sent["removed"]:
                db.add(SentenceChange(
                    drift_score_id=drift_score.id, change_type="removed",
                    sentence_text=entry["text"], sentence_index=entry["index"],
                ))
            for entry in sent["changed"]:
                db.add(SentenceChange(
                    drift_score_id=drift_score.id, change_type="changed",
                    sentence_text=entry["curr_text"],
                    matched_text=entry["prev_text"],
                    similarity_score=entry["similarity"],
                    sentence_index=entry["curr_index"],
                ))

            # Track phrases: watchlist check + TF-IDF/KeyBERT keyphrase discovery
            watchlist = check_watchlist_phrases(prev_section.section_text, curr_section.section_text)
            keyphrase_changes = compare_keyphrases(prev_section.section_text, curr_section.section_text)

            # Build combined phrase_comparison for response (backward compat)
            phrase_comparison = {
                "priority": watchlist,
                "discovered": {
                    "appeared": keyphrase_changes["appeared"],
                    "disappeared": keyphrase_changes["disappeared"],
                },
                "keyphrase_changes": keyphrase_changes,
            }

            # Store watchlist phrase changes in DB
            for phrase in watchlist["appeared"]:
                db.add(KeyPhrase(
                    filing_id=filing.id, section_type=section_type,
                    phrase=phrase, first_seen_filing_id=filing.id, status="appeared",
                ))
            for phrase in watchlist["disappeared"]:
                db.add(KeyPhrase(
                    filing_id=filing.id, section_type=section_type,
                    phrase=phrase, status="disappeared",
                ))
            for phrase in watchlist["persisted"]:
                db.add(KeyPhrase(
                    filing_id=filing.id, section_type=section_type,
                    phrase=phrase, status="persisted",
                ))

            # Store top TF-IDF discovered keyphrase changes
            for entry in keyphrase_changes["appeared"][:10]:
                db.add(KeyPhrase(
                    filing_id=filing.id, section_type=section_type,
                    phrase=entry["phrase"], first_seen_filing_id=filing.id, status="appeared",
                ))
            for entry in keyphrase_changes["disappeared"][:10]:
                db.add(KeyPhrase(
                    filing_id=filing.id, section_type=section_type,
                    phrase=entry["phrase"], status="disappeared",
                ))
            for entry in keyphrase_changes["intensified"][:10]:
                db.add(KeyPhrase(
                    filing_id=filing.id, section_type=section_type,
                    phrase=entry["phrase"], status="intensified",
                ))
            for entry in keyphrase_changes["diminished"][:10]:
                db.add(KeyPhrase(
                    filing_id=filing.id, section_type=section_type,
                    phrase=entry["phrase"], status="diminished",
                ))

            # Obfuscation detection
            obfuscation = detect_obfuscation(prev_section.section_text, curr_section.section_text)
            obfuscation_result = {
                "overall_obfuscation_score": obfuscation.overall_obfuscation_score,
                "density_change": obfuscation.density_change,
                "specificity_change": obfuscation.specificity_change,
                "readability_change": obfuscation.readability_change,
                "detected_euphemisms": obfuscation.detected_euphemisms,
            }

            # Entropy / information-theoretic analysis
            entropy = compute_filing_entropy(prev_section.section_text, curr_section.section_text)
            entropy_result = {
                "kl_divergence": entropy.kl_divergence,
                "novelty_score": entropy.novelty_score,
                "entropy_rate_change": entropy.entropy_rate_change,
                "vocab_overlap": entropy.vocab_overlap,
            }

            results.append({
                "section_type": section_type,
                "cosine_distance": round(drift["cosine_distance"], 4),
                "jaccard_distance": round(drift["jaccard_distance"], 4),
                "added_words": drift["added_words"],
                "removed_words": drift["removed_words"],
                "sentiment_delta": {k: round(v, 4) for k, v in drift["sentiment_delta"].items()},
                "phrases": phrase_comparison,
                "sentence_changes": sent["stats"],
                "top_changes": {
                    "added": [e["text"][:200] for e in sent["added"][:5]],
                    "removed": [e["text"][:200] for e in sent["removed"][:5]],
                    "changed": [
                        {"from": e["prev_text"][:200], "to": e["curr_text"][:200],
                         "similarity": e["similarity"]}
                        for e in sent["changed"][:5]
                    ],
                },
                "obfuscation": obfuscation_result,
                "entropy": entropy_result,
                "_scored_changes": sent,  # internal: used for alert generation
            })

        # ------------------------------------------------------------------
        # Anomaly detection: compare this filing's drift against history
        # ------------------------------------------------------------------
        anomaly_results = {}
        trend_result = {}

        if results:
            # Fetch all historical drift scores for this company
            history_stmt = (
                select(DriftScore.section_type, DriftScore.cosine_distance, DriftScore.sentiment_delta)
                .join(Filing, DriftScore.filing_id == Filing.id)
                .where(
                    DriftScore.company_id == filing.company_id,
                    DriftScore.filing_id != filing.id,  # exclude current
                )
                .order_by(Filing.filing_date)
            )
            history_result = await db.execute(history_stmt)
            history_rows = history_result.all()

            # Group history by section type
            history_by_section: dict[str, list[float]] = {}
            sentiment_history_all: list[dict[str, float]] = []
            drift_history_all: list[float] = []

            for row in history_rows:
                history_by_section.setdefault(row.section_type, []).append(row.cosine_distance)
                if row.cosine_distance is not None:
                    drift_history_all.append(row.cosine_distance)
                if row.sentiment_delta:
                    sentiment_history_all.append(row.sentiment_delta)

            # Run anomaly detection per section
            for section_result in results:
                section_type = section_result["section_type"]
                current_drift = section_result["cosine_distance"]
                company_history = history_by_section.get(section_type, [])

                anomaly = detect_anomaly(
                    current_drift=current_drift,
                    company_history=company_history,
                    sector_history=None,  # sector data not yet available
                )
                anomaly_results[section_type] = {
                    "company_z_score": anomaly.company_z_score,
                    "sector_z_score": anomaly.sector_z_score,
                    "is_anomalous": anomaly.is_anomalous,
                    "anomaly_level": anomaly.anomaly_level,
                    "company_mean": anomaly.company_mean,
                    "company_stddev": anomaly.company_stddev,
                }

            # Run trend detection across all sections (aggregate view)
            trend_result = detect_trends(
                drift_history=drift_history_all,
                sentiment_history=sentiment_history_all if sentiment_history_all else None,
            )

        # ------------------------------------------------------------------
        # Alert generation
        # ------------------------------------------------------------------
        alerts_created = []

        for section_result in results:
            section_type = section_result["section_type"]

            # 1. Drift anomaly alerts
            anomaly_info = anomaly_results.get(section_type)
            if anomaly_info and anomaly_info["is_anomalous"]:
                severity_map = {"extreme": "critical", "high": "high", "elevated": "medium"}
                severity = severity_map.get(anomaly_info["anomaly_level"], "medium")
                alert = Alert(
                    company_id=filing.company_id,
                    filing_id=filing.id,
                    alert_type="drift_anomaly",
                    severity=severity,
                    message=(
                        f"Anomalous drift detected in {section_type}: "
                        f"z-score {anomaly_info['company_z_score'] or 'N/A'} "
                        f"({anomaly_info['anomaly_level']} level)"
                    ),
                    metadata_={
                        "section_type": section_type,
                        "z_score": anomaly_info["company_z_score"],
                        "anomaly_level": anomaly_info["anomaly_level"],
                        "company_mean": anomaly_info["company_mean"],
                        "company_stddev": anomaly_info["company_stddev"],
                        "cosine_distance": section_result["cosine_distance"],
                    },
                )
                db.add(alert)
                alerts_created.append({"alert_type": "drift_anomaly", "section_type": section_type, "severity": severity})

            # 2. Critical risk language alerts (from sentence-level risk scoring)
            sent = section_result.get("_scored_changes")
            if sent:
                risk_summary = sent.get("risk_summary", {})
                if risk_summary.get("critical_changes", 0) > 0:
                    alert = Alert(
                        company_id=filing.company_id,
                        filing_id=filing.id,
                        alert_type="critical_risk_language",
                        severity="critical",
                        message=(
                            f"Critical risk language detected in {section_type}: "
                            f"{risk_summary['critical_changes']} critical change(s), "
                            f"max risk score {risk_summary.get('max_risk_score', 0):.4f}"
                        ),
                        metadata_={
                            "section_type": section_type,
                            "critical_changes": risk_summary["critical_changes"],
                            "high_risk_changes": risk_summary.get("high_risk_changes", 0),
                            "max_risk_score": risk_summary.get("max_risk_score", 0),
                            "max_risk_level": risk_summary.get("max_risk_level", "unknown"),
                        },
                    )
                    db.add(alert)
                    alerts_created.append({"alert_type": "critical_risk_language", "section_type": section_type, "severity": "critical"})

            # 3. Phrase change alerts (priority phrases appeared or disappeared)
            phrases = section_result.get("phrases", {})
            priority = phrases.get("priority", {})
            appeared = priority.get("appeared", [])
            disappeared = priority.get("disappeared", [])
            if appeared or disappeared:
                alert = Alert(
                    company_id=filing.company_id,
                    filing_id=filing.id,
                    alert_type="phrase_change",
                    severity="high",
                    message=(
                        f"Priority phrase changes in {section_type}: "
                        f"{len(appeared)} appeared, {len(disappeared)} disappeared"
                    ),
                    metadata_={
                        "section_type": section_type,
                        "appeared": appeared,
                        "disappeared": disappeared,
                    },
                )
                db.add(alert)
                alerts_created.append({"alert_type": "phrase_change", "section_type": section_type, "severity": "high"})

            # 4. Obfuscation alerts
            obfuscation_data = section_result.get("obfuscation", {})
            if obfuscation_data.get("overall_obfuscation_score", 0) > 0.5:
                alert = Alert(
                    company_id=filing.company_id,
                    filing_id=filing.id,
                    alert_type="obfuscation_detected",
                    severity="high",
                    message=(
                        f"High obfuscation score in {section_type}: "
                        f"{obfuscation_data['overall_obfuscation_score']:.4f}"
                    ),
                    metadata_={
                        "section_type": section_type,
                        "overall_obfuscation_score": obfuscation_data["overall_obfuscation_score"],
                        "density_change": obfuscation_data.get("density_change"),
                        "specificity_change": obfuscation_data.get("specificity_change"),
                        "readability_change": obfuscation_data.get("readability_change"),
                        "detected_euphemisms": obfuscation_data.get("detected_euphemisms", []),
                    },
                )
                db.add(alert)
                alerts_created.append({"alert_type": "obfuscation_detected", "section_type": section_type, "severity": "high"})

        # Mark filing as analyzed right before the single commit
        filing.status = "analyzed"
        await db.commit()

    except Exception:
        await db.rollback()
        logger.exception("Analysis failed for filing %s; transaction rolled back", filing_id)
        raise HTTPException(status_code=500, detail="Analysis failed; all changes rolled back")

    # Strip internal keys before returning
    clean_results = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    ]

    return {
        "filing_id": filing.id,
        "prev_filing_id": prev_filing.id,
        "form_type": filing.form_type,
        "sections_analyzed": len(clean_results),
        "results": clean_results,
        "anomaly": anomaly_results,
        "trends": trend_result,
        "alerts_created": alerts_created,
    }


@router.get("/drift/{drift_score_id}/sentences")
async def get_sentence_changes(
    drift_score_id: int,
    change_type: str | None = Query(None, description="Filter: added, removed, changed"),
    db: AsyncSession = Depends(get_db),
):
    """Get sentence-level changes for a specific drift score."""
    query = select(SentenceChange).where(SentenceChange.drift_score_id == drift_score_id)
    if change_type:
        query = query.where(SentenceChange.change_type == change_type)
    query = query.order_by(SentenceChange.sentence_index)

    result = await db.execute(query)
    changes = result.scalars().all()
    if not changes:
        raise HTTPException(status_code=404, detail="No sentence changes found")

    return {
        "drift_score_id": drift_score_id,
        "data": [
            {
                "change_type": c.change_type,
                "text": c.sentence_text,
                "matched_text": c.matched_text,
                "similarity": round(c.similarity_score, 4) if c.similarity_score else None,
                "index": c.sentence_index,
            }
            for c in changes
        ],
        "total": len(changes),
    }


@router.get("/companies/{ticker}/drift")
async def get_drift_timeline(
    ticker: str,
    section_type: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Get drift score time series for a company."""
    info = await lookup_ticker(ticker)
    if not info:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    stmt = select(Company).where(Company.cik == info["cik"])
    result = await db.execute(stmt)
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail="Company not in database. Ingest filings first.")

    # Build drift timeline query
    query = (
        select(
            DriftScore.section_type,
            DriftScore.cosine_distance,
            DriftScore.jaccard_distance,
            DriftScore.sentiment_delta,
            DriftScore.added_words,
            DriftScore.removed_words,
            Filing.filing_date,
            Filing.form_type,
            Filing.accession_number,
        )
        .join(Filing, DriftScore.filing_id == Filing.id)
        .where(DriftScore.company_id == company.id)
        .order_by(Filing.filing_date)
    )

    if section_type:
        query = query.where(DriftScore.section_type == section_type)

    result = await db.execute(query)
    rows = result.all()

    timeline = []
    for row in rows:
        timeline.append({
            "section_type": row.section_type,
            "cosine_distance": round(row.cosine_distance, 4) if row.cosine_distance else None,
            "jaccard_distance": round(row.jaccard_distance, 4) if row.jaccard_distance else None,
            "sentiment_delta": row.sentiment_delta,
            "added_words": row.added_words,
            "removed_words": row.removed_words,
            "filing_date": row.filing_date.isoformat() if row.filing_date else None,
            "form_type": row.form_type,
            "accession_number": row.accession_number,
        })

    return {"ticker": ticker, "data": timeline, "total": len(timeline)}


@router.get("/filings/{filing_id}/diff")
async def get_filing_diff(
    filing_id: int,
    vs: int = Query(..., description="ID of the previous filing to diff against"),
    section_type: str = Query("risk_factors"),
    db: AsyncSession = Depends(get_db),
):
    """Get unified diff between two filings for a given section."""
    # Fetch both sections
    stmt = select(Section).where(Section.filing_id == filing_id, Section.section_type == section_type)
    result = await db.execute(stmt)
    curr_section = result.scalar_one_or_none()

    stmt = select(Section).where(Section.filing_id == vs, Section.section_type == section_type)
    result = await db.execute(stmt)
    prev_section = result.scalar_one_or_none()

    if not curr_section or not prev_section:
        raise HTTPException(status_code=404, detail="Section not found in one or both filings")

    diff_text = unified_diff(prev_section.section_text, curr_section.section_text)
    stats = diff_stats(prev_section.section_text, curr_section.section_text)

    return {
        "filing_id": filing_id,
        "vs_filing_id": vs,
        "section_type": section_type,
        "diff": diff_text,
        "stats": stats,
    }


@router.get("/drift/screener")
async def drift_screener(
    sort_by: str = Query("cosine_distance", description="Sort by: cosine_distance, jaccard_distance"),
    section_type: str = Query("risk_factors"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Screener: rank all companies by their most recent drift score."""
    # Subquery for most recent drift score per company
    latest_drift = (
        select(
            DriftScore.company_id,
            func.max(DriftScore.computed_at).label("latest"),
        )
        .where(DriftScore.section_type == section_type)
        .group_by(DriftScore.company_id)
        .subquery()
    )

    sort_col = getattr(DriftScore, sort_by, DriftScore.cosine_distance)

    query = (
        select(
            Company.ticker,
            Company.name,
            DriftScore.cosine_distance,
            DriftScore.jaccard_distance,
            DriftScore.added_words,
            DriftScore.removed_words,
            DriftScore.sentiment_delta,
            Filing.filing_date,
        )
        .join(DriftScore, DriftScore.company_id == Company.id)
        .join(Filing, DriftScore.filing_id == Filing.id)
        .join(
            latest_drift,
            (DriftScore.company_id == latest_drift.c.company_id)
            & (DriftScore.computed_at == latest_drift.c.latest),
        )
        .where(DriftScore.section_type == section_type)
        .order_by(desc(sort_col))
        .offset(offset)
        .limit(limit)
    )

    result = await db.execute(query)
    rows = result.all()

    data = []
    for row in rows:
        data.append({
            "ticker": row.ticker,
            "name": row.name,
            "cosine_distance": round(row.cosine_distance, 4) if row.cosine_distance else None,
            "jaccard_distance": round(row.jaccard_distance, 4) if row.jaccard_distance else None,
            "added_words": row.added_words,
            "removed_words": row.removed_words,
            "sentiment_delta": row.sentiment_delta,
            "filing_date": row.filing_date.isoformat() if row.filing_date else None,
        })

    return {"data": data, "total": len(data), "section_type": section_type}


@router.get("/companies/{ticker}/phrases")
async def get_phrase_timeline(
    ticker: str,
    db: AsyncSession = Depends(get_db),
):
    """Get key phrase tracking timeline for a company."""
    info = await lookup_ticker(ticker)
    if not info:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    stmt = select(Company).where(Company.cik == info["cik"])
    result = await db.execute(stmt)
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail="Company not in database")

    query = (
        select(
            KeyPhrase.phrase,
            KeyPhrase.section_type,
            KeyPhrase.status,
            Filing.filing_date,
            Filing.form_type,
        )
        .join(Filing, KeyPhrase.filing_id == Filing.id)
        .where(Filing.company_id == company.id)
        .order_by(Filing.filing_date)
    )

    result = await db.execute(query)
    rows = result.all()

    timeline = []
    for row in rows:
        timeline.append({
            "phrase": row.phrase,
            "section_type": row.section_type,
            "status": row.status,
            "filing_date": row.filing_date.isoformat() if row.filing_date else None,
            "form_type": row.form_type,
        })

    return {"ticker": ticker, "data": timeline, "total": len(timeline)}
