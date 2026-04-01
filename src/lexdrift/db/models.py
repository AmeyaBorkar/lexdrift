from datetime import datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cik: Mapped[str] = mapped_column(String(10), unique=True, nullable=False)
    ticker: Mapped[str | None] = mapped_column(String(10))
    name: Mapped[str | None] = mapped_column(Text)
    sic_code: Mapped[str | None] = mapped_column(String(4))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    filings: Mapped[list["Filing"]] = relationship(back_populates="company")
    drift_scores: Mapped[list["DriftScore"]] = relationship(back_populates="company")
    alerts: Mapped[list["Alert"]] = relationship(back_populates="company")


class Filing(Base):
    __tablename__ = "filings"
    __table_args__ = (Index("ix_filings_company_date", "company_id", "filing_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), nullable=False)
    accession_number: Mapped[str] = mapped_column(String(25), unique=True, nullable=False)
    form_type: Mapped[str] = mapped_column(String(10), nullable=False)
    filing_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    report_date: Mapped[datetime | None] = mapped_column(Date)
    document_url: Mapped[str | None] = mapped_column(Text)
    raw_text: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    company: Mapped["Company"] = relationship(back_populates="filings")
    sections: Mapped[list["Section"]] = relationship(back_populates="filing")
    drift_scores: Mapped[list["DriftScore"]] = relationship(
        back_populates="filing", foreign_keys="DriftScore.filing_id"
    )
    alerts: Mapped[list["Alert"]] = relationship(back_populates="filing")
    key_phrases: Mapped[list["KeyPhrase"]] = relationship(
        back_populates="filing", foreign_keys="KeyPhrase.filing_id"
    )


class Section(Base):
    __tablename__ = "sections"
    __table_args__ = (
        UniqueConstraint("filing_id", "section_type", name="uq_section_filing_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filing_id: Mapped[int] = mapped_column(ForeignKey("filings.id"), nullable=False)
    section_type: Mapped[str] = mapped_column(String(50), nullable=False)
    section_text: Mapped[str | None] = mapped_column(Text)
    word_count: Mapped[int | None] = mapped_column(Integer)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)

    filing: Mapped["Filing"] = relationship(back_populates="sections")


class DriftScore(Base):
    __tablename__ = "drift_scores"
    __table_args__ = (
        UniqueConstraint("filing_id", "section_type", name="uq_drift_filing_section"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), nullable=False)
    filing_id: Mapped[int] = mapped_column(ForeignKey("filings.id"), nullable=False)
    prev_filing_id: Mapped[int] = mapped_column(ForeignKey("filings.id"), nullable=False)
    section_type: Mapped[str] = mapped_column(String(50), nullable=False)
    cosine_distance: Mapped[float | None] = mapped_column(Float)
    jaccard_distance: Mapped[float | None] = mapped_column(Float)
    added_words: Mapped[int | None] = mapped_column(Integer)
    removed_words: Mapped[int | None] = mapped_column(Integer)
    sentiment_delta: Mapped[dict | None] = mapped_column(JSON)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    company: Mapped["Company"] = relationship(back_populates="drift_scores")
    filing: Mapped["Filing"] = relationship(
        back_populates="drift_scores", foreign_keys=[filing_id]
    )
    prev_filing: Mapped["Filing"] = relationship(foreign_keys=[prev_filing_id])


class Watchlist(Base):
    __tablename__ = "watchlists"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    companies: Mapped[list["Company"]] = relationship(
        secondary="watchlist_companies", viewonly=True
    )


class WatchlistCompany(Base):
    __tablename__ = "watchlist_companies"

    watchlist_id: Mapped[int] = mapped_column(
        ForeignKey("watchlists.id"), primary_key=True
    )
    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id"), primary_key=True
    )


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (Index("ix_alerts_company_created", "company_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), nullable=False)
    filing_id: Mapped[int] = mapped_column(ForeignKey("filings.id"), nullable=False)
    alert_type: Mapped[str] = mapped_column(String(30), nullable=False)
    severity: Mapped[str] = mapped_column(String(10), default="medium")
    message: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON)
    read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    company: Mapped["Company"] = relationship(back_populates="alerts")
    filing: Mapped["Filing"] = relationship(back_populates="alerts")


class KeyPhrase(Base):
    __tablename__ = "key_phrases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filing_id: Mapped[int] = mapped_column(ForeignKey("filings.id"), nullable=False)
    section_type: Mapped[str] = mapped_column(String(50), nullable=False)
    phrase: Mapped[str] = mapped_column(String(200), nullable=False)
    first_seen_filing_id: Mapped[int | None] = mapped_column(ForeignKey("filings.id"))
    status: Mapped[str] = mapped_column(String(15), nullable=False)

    filing: Mapped["Filing"] = relationship(
        back_populates="key_phrases", foreign_keys=[filing_id]
    )
    first_seen_filing: Mapped["Filing | None"] = relationship(
        foreign_keys=[first_seen_filing_id]
    )


class SentenceChange(Base):
    """Stores individual sentence-level changes detected between two filings.

    Each row represents a sentence that was added, removed, or semantically
    changed between consecutive filings of the same type.
    """
    __tablename__ = "sentence_changes"
    __table_args__ = (
        Index("ix_sentence_changes_drift", "drift_score_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    drift_score_id: Mapped[int] = mapped_column(ForeignKey("drift_scores.id"), nullable=False)
    change_type: Mapped[str] = mapped_column(String(15), nullable=False)  # added, removed, changed
    sentence_text: Mapped[str] = mapped_column(Text, nullable=False)
    matched_text: Mapped[str | None] = mapped_column(Text)  # the paired sentence (for 'changed' type)
    similarity_score: Mapped[float | None] = mapped_column(Float)  # cosine sim with best match
    sentence_index: Mapped[int | None] = mapped_column(Integer)  # position in the section

    drift_score: Mapped["DriftScore"] = relationship()
