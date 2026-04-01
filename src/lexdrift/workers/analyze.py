"""Celery task: run NLP drift analysis on an ingested filing.

Compares each section of the target filing against the corresponding section
of the most recent *previous* filing of the same form type for the same
company.  Stores DriftScore, KeyPhrase, and SentenceChange records.

Uses SYNCHRONOUS SQLAlchemy because Celery workers run in regular threads.
"""

import logging

from sqlalchemy import create_engine, desc, select
from sqlalchemy.orm import Session, sessionmaker

from lexdrift.config import settings
from lexdrift.db.models import (
    Base,
    DriftScore,
    Filing,
    KeyPhrase,
    Section,
    SentenceChange,
)
from lexdrift.nlp.drift import compute_drift
from lexdrift.nlp.phrases import compare_phrases
from lexdrift.workers.celery_app import app

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synchronous DB setup (same pattern as ingest.py)
# ---------------------------------------------------------------------------

def _make_sync_url(async_url: str) -> str:
    url = async_url.replace("+aiosqlite", "")
    url = url.replace("+asyncpg", "+psycopg2")
    return url


_sync_engine = create_engine(_make_sync_url(settings.database_url), echo=False, future=True)
SyncSessionFactory = sessionmaker(bind=_sync_engine, class_=Session, expire_on_commit=False)

Base.metadata.create_all(_sync_engine)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@app.task(name="lexdrift.workers.analyze.analyze_filing", bind=True, max_retries=3)
def analyze_filing(self, filing_id: int):
    """Compute drift for every section of *filing_id* vs. its predecessor.

    Steps
    -----
    1. Load the target filing and its sections.
    2. Find the previous filing of the same form_type for the same company.
    3. For each section that exists in both filings, call ``compute_drift``.
    4. Store DriftScore, KeyPhrase, and SentenceChange records.
    5. Mark the filing status as ``"analyzed"``.

    Returns a summary dict.
    """
    try:
        return _do_analyze(filing_id)
    except Exception as exc:
        logger.exception("Analysis failed for filing %s", filing_id)
        raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))


def _do_analyze(filing_id: int) -> dict:
    with SyncSessionFactory() as session:
        # 1. Load the target filing
        filing = session.execute(
            select(Filing).where(Filing.id == filing_id)
        ).scalar_one_or_none()

        if filing is None:
            raise ValueError(f"Filing {filing_id} not found")

        # 2. Find the previous filing of the same form type for this company
        prev_filing = session.execute(
            select(Filing)
            .where(
                Filing.company_id == filing.company_id,
                Filing.form_type == filing.form_type,
                Filing.filing_date < filing.filing_date,
            )
            .order_by(desc(Filing.filing_date))
            .limit(1)
        ).scalar_one_or_none()

        if prev_filing is None:
            logger.info(
                "No previous %s filing for company_id=%s; nothing to compare",
                filing.form_type, filing.company_id,
            )
            filing.status = "analyzed"
            session.commit()
            return {
                "filing_id": filing_id,
                "message": "No previous filing found for comparison",
                "sections_analyzed": 0,
            }

        # Load sections for both filings
        curr_sections = {
            s.section_type: s
            for s in session.execute(
                select(Section).where(Section.filing_id == filing.id)
            ).scalars().all()
        }

        prev_sections = {
            s.section_type: s
            for s in session.execute(
                select(Section).where(Section.filing_id == prev_filing.id)
            ).scalars().all()
        }

        # 3. Compute drift for each matching section
        sections_analyzed = 0
        for section_type, curr_section in curr_sections.items():
            prev_section = prev_sections.get(section_type)
            if not prev_section:
                continue
            if not prev_section.section_text or not curr_section.section_text:
                continue

            drift = compute_drift(
                prev_section.section_text,
                curr_section.section_text,
                prev_embedding=prev_section.embedding,
                curr_embedding=curr_section.embedding,
            )

            # Persist computed embeddings back to sections
            if not curr_section.embedding:
                curr_section.embedding = drift["curr_embedding_bytes"]
            if not prev_section.embedding:
                prev_section.embedding = drift["prev_embedding_bytes"]

            # 4a. Store DriftScore
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
            session.add(drift_score)
            session.flush()  # materialise drift_score.id for FK references

            # 4b. Store SentenceChange records
            sent = drift["sentence_changes"]

            for entry in sent["added"]:
                session.add(SentenceChange(
                    drift_score_id=drift_score.id,
                    change_type="added",
                    sentence_text=entry["text"],
                    sentence_index=entry["index"],
                ))

            for entry in sent["removed"]:
                session.add(SentenceChange(
                    drift_score_id=drift_score.id,
                    change_type="removed",
                    sentence_text=entry["text"],
                    sentence_index=entry["index"],
                ))

            for entry in sent["changed"]:
                session.add(SentenceChange(
                    drift_score_id=drift_score.id,
                    change_type="changed",
                    sentence_text=entry["curr_text"],
                    matched_text=entry["prev_text"],
                    similarity_score=entry["similarity"],
                    sentence_index=entry["curr_index"],
                ))

            # 4c. Store KeyPhrase records (priority + auto-discovered)
            phrase_comparison = compare_phrases(
                prev_section.section_text, curr_section.section_text
            )

            for phrase in phrase_comparison["priority"]["appeared"]:
                session.add(KeyPhrase(
                    filing_id=filing.id,
                    section_type=section_type,
                    phrase=phrase,
                    first_seen_filing_id=filing.id,
                    status="appeared",
                ))

            for phrase in phrase_comparison["priority"]["disappeared"]:
                session.add(KeyPhrase(
                    filing_id=filing.id,
                    section_type=section_type,
                    phrase=phrase,
                    status="disappeared",
                ))

            for phrase in phrase_comparison["priority"]["persisted"]:
                session.add(KeyPhrase(
                    filing_id=filing.id,
                    section_type=section_type,
                    phrase=phrase,
                    status="persisted",
                ))

            for entry in phrase_comparison["discovered"]["appeared"][:10]:
                session.add(KeyPhrase(
                    filing_id=filing.id,
                    section_type=section_type,
                    phrase=entry["phrase"],
                    first_seen_filing_id=filing.id,
                    status="appeared",
                ))

            for entry in phrase_comparison["discovered"]["disappeared"][:10]:
                session.add(KeyPhrase(
                    filing_id=filing.id,
                    section_type=section_type,
                    phrase=entry["phrase"],
                    status="disappeared",
                ))

            sections_analyzed += 1
            logger.info(
                "Drift computed for filing %s, section '%s': cosine=%.4f, jaccard=%.4f",
                filing_id, section_type,
                drift["cosine_distance"], drift["jaccard_distance"],
            )

        # 5. Mark filing as analyzed
        filing.status = "analyzed"
        session.commit()

    summary = {
        "filing_id": filing_id,
        "prev_filing_id": prev_filing.id,
        "form_type": filing.form_type,
        "sections_analyzed": sections_analyzed,
    }
    logger.info("Analysis complete: %s", summary)
    return summary
