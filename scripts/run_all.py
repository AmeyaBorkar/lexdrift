"""
LexDrift Complete Pipeline -- Run everything in one go.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --skip-ingest
    python scripts/run_all.py --skip-train
    python scripts/run_all.py --tickers AAPL,MSFT,GOOGL --limit 2

This script:
1. Ingests filings for 30 S&P 500 companies (3 per company = 90 filings)
2. Analyzes all filings (drift + sentence + risk + entropy + obfuscation)
3. Generates elite training data (text-level, no circular dependency)
4. Fine-tunes the embedding model
5. Trains the risk classifier
6. Trains the boilerplate classifier
7. Re-analyzes all filings with the improved models
8. Prints final statistics

Total runtime: ~30-60 minutes (mostly SEC rate limits).
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure src/ is importable when running as a standalone script
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

# ---------------------------------------------------------------------------
# S&P 500 tickers (same list as scripts/backfill.py)
# ---------------------------------------------------------------------------
SP500_TICKERS: list[str] = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    # Finance
    "JPM", "BAC", "GS", "WFC", "BRK-B",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV",
    # Consumer
    "WMT", "PG", "KO", "MCD", "NKE",
    # Industrial
    "CAT", "BA", "HON", "GE",
    # Energy
    "XOM", "CVX",
    # Telecom
    "VZ", "T",
    # Real Estate
    "AMT",
    # Utilities
    "NEE",
]

SEC_REQUEST_DELAY = 0.15  # seconds between SEC requests

# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

_pipeline_start: float = 0.0
_interrupted: bool = False


def _timestamp() -> str:
    """Return current time as HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")


def _elapsed() -> str:
    """Return elapsed time since pipeline start as MM:SS."""
    elapsed = time.time() - _pipeline_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes:02d}:{seconds:02d}"


def _step(num: int, total: int, description: str) -> None:
    """Print a step header with timestamp."""
    print(f"\n[{_timestamp()}] Step {num}/{total}: {description}")
    print("=" * 70)


def _info(msg: str) -> None:
    """Print an info message with timestamp."""
    print(f"  [{_timestamp()}] {msg}")


def _warn(msg: str) -> None:
    """Print a warning message."""
    print(f"  [{_timestamp()}] WARNING: {msg}")


def _handle_interrupt(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _interrupted
    if _interrupted:
        # Second interrupt -- force exit
        print("\n\nForce exit.")
        sys.exit(1)
    _interrupted = True
    print("\n\nInterrupt received. Finishing current step, then exiting...")
    print("Press Ctrl+C again to force exit immediately.")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_sync_session_factory():
    """Create a synchronous SQLAlchemy session factory."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session, sessionmaker

    from lexdrift.config import settings
    from lexdrift.db.models import Base

    db_url = settings.database_url
    sync_url = db_url.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg2")

    engine = create_engine(sync_url, echo=False, future=True)
    Base.metadata.create_all(engine)

    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def _run_async(coro):
    """Run an async coroutine from synchronous code."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Step 1: Ingest filings
# ---------------------------------------------------------------------------

def step_ingest(tickers: list[str], limit: int) -> dict:
    """Ingest SEC filings for the given tickers. Returns summary stats."""
    from sqlalchemy import select

    from lexdrift.db.models import Company, Filing, Section
    from lexdrift.edgar.filings import (
        build_document_url,
        download_filing,
        get_filing_metadata,
        parse_filing_list,
    )
    from lexdrift.edgar.parser import parse_filing
    from lexdrift.edgar.tickers import lookup_ticker

    SessionFactory = _get_sync_session_factory()

    stats = {
        "companies_processed": 0,
        "companies_failed": 0,
        "filings_ingested": 0,
        "filings_skipped": 0,
        "filing_ids": [],
    }

    for i, ticker in enumerate(tickers, 1):
        if _interrupted:
            _warn(f"Interrupted after {i - 1}/{len(tickers)} tickers")
            break

        _info(f"[{i}/{len(tickers)}] {ticker}")

        try:
            # Resolve ticker -> CIK
            info = _run_async(lookup_ticker(ticker))
            if not info:
                _warn(f"Ticker '{ticker}' not found in SEC data -- skipping")
                stats["companies_failed"] += 1
                continue

            cik = info["cik"]
            company_name = info["name"]

            # Fetch filing metadata
            time.sleep(SEC_REQUEST_DELAY)
            metadata = _run_async(get_filing_metadata(cik))
            filings_list = parse_filing_list(metadata, form_types={"10-K"})

            if not filings_list:
                _warn(f"No 10-K filings found for {ticker}")
                stats["companies_failed"] += 1
                continue

            filings_list = filings_list[:limit]

            with SessionFactory() as session:
                # Upsert company
                company = session.execute(
                    select(Company).where(Company.cik == cik)
                ).scalar_one_or_none()

                if company is None:
                    company = Company(cik=cik, ticker=ticker.upper(), name=company_name)
                    session.add(company)
                    session.flush()

                for filing_meta in filings_list:
                    if _interrupted:
                        break

                    accession = filing_meta["accession_number"]

                    # Skip if already ingested
                    exists = session.execute(
                        select(Filing.id).where(Filing.accession_number == accession)
                    ).scalar_one_or_none()
                    if exists is not None:
                        stats["filings_skipped"] += 1
                        stats["filing_ids"].append(exists)
                        continue

                    time.sleep(SEC_REQUEST_DELAY)

                    try:
                        html_text = _run_async(
                            download_filing(cik, accession, filing_meta["primary_document"])
                        )
                    except Exception as e:
                        _warn(f"Failed to download {accession}: {e}")
                        continue

                    doc_url = build_document_url(cik, accession, filing_meta["primary_document"])
                    sections = parse_filing(html_text, "10-K")

                    filing = Filing(
                        company_id=company.id,
                        accession_number=accession,
                        form_type=filing_meta["form_type"],
                        filing_date=filing_meta["filing_date"],
                        report_date=filing_meta.get("report_date"),
                        document_url=doc_url,
                        raw_text=html_text[:500_000],
                        status="ingested",
                    )
                    session.add(filing)
                    session.flush()

                    for section_type, section_text in sections.items():
                        word_count = len(section_text.split())
                        section = Section(
                            filing_id=filing.id,
                            section_type=section_type,
                            section_text=section_text,
                            word_count=word_count,
                        )
                        session.add(section)

                    stats["filings_ingested"] += 1
                    stats["filing_ids"].append(filing.id)

                session.commit()

            stats["companies_processed"] += 1

        except Exception as e:
            _warn(f"Failed to process {ticker}: {e}")
            stats["companies_failed"] += 1

        # Polite pause between companies
        if i < len(tickers) and not _interrupted:
            time.sleep(1.0)

    return stats


# ---------------------------------------------------------------------------
# Step 2: Analyze filings
# ---------------------------------------------------------------------------

def step_analyze(filing_ids: list[int] | None = None) -> dict:
    """Run drift analysis on filings. If no IDs given, analyze all un-analyzed."""
    from sqlalchemy import select

    from lexdrift.db.models import Filing

    SessionFactory = _get_sync_session_factory()

    if filing_ids is None:
        # Find all un-analyzed filings
        with SessionFactory() as session:
            rows = session.execute(
                select(Filing.id).where(Filing.status != "analyzed")
            ).scalars().all()
            filing_ids = list(rows)

    if not filing_ids:
        _info("No filings to analyze")
        return {"analyzed": 0, "failed": 0}

    _info(f"Analyzing {len(filing_ids)} filings...")

    # Import the analysis function (avoids Celery dependency at module level)
    from lexdrift.workers.analyze import _do_analyze

    stats = {"analyzed": 0, "failed": 0}

    for i, fid in enumerate(filing_ids, 1):
        if _interrupted:
            _warn(f"Interrupted after {i - 1}/{len(filing_ids)} filings")
            break

        try:
            result = _do_analyze(fid)
            stats["analyzed"] += 1
            sections = result.get("sections_analyzed", 0)
            if i % 10 == 0 or i == len(filing_ids):
                _info(f"  [{i}/{len(filing_ids)}] Filing {fid}: {sections} sections")
        except Exception as e:
            _warn(f"Analysis failed for filing {fid}: {e}")
            stats["failed"] += 1

    return stats


# ---------------------------------------------------------------------------
# Step 3: Generate elite training data
# ---------------------------------------------------------------------------

def step_generate_training_data() -> tuple[list, dict]:
    """Generate elite training pairs and return (pairs, report)."""
    from lexdrift.training.data_quality import data_quality_report, generate_elite_pairs

    SessionFactory = _get_sync_session_factory()

    with SessionFactory() as session:
        pairs = generate_elite_pairs(session, max_pairs=50000)

    report = data_quality_report(pairs)
    return pairs, report


# ---------------------------------------------------------------------------
# Step 4: Fine-tune embedding model
# ---------------------------------------------------------------------------

def step_finetune(training_pairs: list) -> str:
    """Fine-tune sentence-transformer on the training pairs. Returns model path."""
    from lexdrift.training.finetune import finetune_embeddings

    if not training_pairs:
        _warn("No training pairs; skipping fine-tuning")
        return ""

    saved_path = finetune_embeddings(
        training_pairs,
        model_name="all-MiniLM-L6-v2",
        output_path="models/lexdrift-finetuned",
        epochs=3,
        batch_size=32,
    )
    return saved_path


# ---------------------------------------------------------------------------
# Step 5: Train risk classifier
# ---------------------------------------------------------------------------

def step_train_risk_classifier() -> str:
    """Train the risk severity classifier. Returns model path."""
    from lexdrift.training.risk_classifier import (
        generate_risk_labels,
        train_risk_classifier,
    )

    SessionFactory = _get_sync_session_factory()

    with SessionFactory() as session:
        training_data = generate_risk_labels(session)

    if not training_data:
        _warn("No risk training data; skipping risk classifier training")
        return ""

    _info(f"Training risk classifier on {len(training_data)} samples")
    saved_path = train_risk_classifier(training_data)
    return saved_path


# ---------------------------------------------------------------------------
# Step 6: Train boilerplate classifier
# ---------------------------------------------------------------------------

def step_train_boilerplate_classifier() -> str:
    """Train the boilerplate classifier. Returns model path."""
    from lexdrift.training.boilerplate_classifier import (
        generate_boilerplate_labels,
        train_boilerplate_classifier,
    )

    SessionFactory = _get_sync_session_factory()

    with SessionFactory() as session:
        training_data = generate_boilerplate_labels(session)

    if not training_data:
        _warn("No boilerplate training data; skipping boilerplate classifier training")
        return ""

    _info(f"Training boilerplate classifier on {len(training_data)} samples")
    saved_path = train_boilerplate_classifier(training_data)
    return saved_path


# ---------------------------------------------------------------------------
# Step 7: Re-analyze with improved models
# ---------------------------------------------------------------------------

def step_reanalyze() -> dict:
    """Re-analyze all filings using the improved models."""
    from sqlalchemy import select

    from lexdrift.db.models import Filing

    SessionFactory = _get_sync_session_factory()

    with SessionFactory() as session:
        filing_ids = list(
            session.execute(select(Filing.id)).scalars().all()
        )

    if not filing_ids:
        _info("No filings to re-analyze")
        return {"analyzed": 0, "failed": 0}

    _info(f"Re-analyzing {len(filing_ids)} filings with improved models...")
    return step_analyze(filing_ids)


# ---------------------------------------------------------------------------
# Step 8: Print final statistics
# ---------------------------------------------------------------------------

def step_print_stats(summary: dict) -> None:
    """Print a final summary of the pipeline run."""
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    elapsed = time.time() - _pipeline_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n  Total elapsed time: {minutes}m {seconds}s")
    print()

    # Ingestion stats
    ingest = summary.get("ingest", {})
    if ingest:
        print("  Ingestion:")
        print(f"    Companies processed:  {ingest.get('companies_processed', 0)}")
        print(f"    Companies failed:     {ingest.get('companies_failed', 0)}")
        print(f"    Filings ingested:     {ingest.get('filings_ingested', 0)}")
        print(f"    Filings skipped:      {ingest.get('filings_skipped', 0)}")
        print()

    # Analysis stats
    analyze = summary.get("analyze", {})
    if analyze:
        print("  Analysis (initial):")
        print(f"    Filings analyzed:     {analyze.get('analyzed', 0)}")
        print(f"    Filings failed:       {analyze.get('failed', 0)}")
        print()

    # Training data stats
    training = summary.get("training_report", {})
    if training:
        print("  Training Data:")
        print(f"    Total pairs:          {training.get('total_pairs', 0)}")
        tier_counts = training.get("tier_counts", {})
        for tier, count in sorted(tier_counts.items()):
            print(f"      Tier {tier}:             {count}")
        label_dist = training.get("label_distribution", {})
        print(f"    Label mean:           {label_dist.get('mean', 0):.4f}")
        print(f"    Label std:            {label_dist.get('std', 0):.4f}")
        vocab = training.get("vocabulary_coverage", {})
        print(f"    Unique tokens:        {vocab.get('total_unique_tokens', 0)}")
        print()

    # Model paths
    models = summary.get("models", {})
    if models:
        print("  Saved Models:")
        for name, path in models.items():
            if path:
                print(f"    {name}: {path}")
        print()

    # Re-analysis stats
    reanalyze = summary.get("reanalyze", {})
    if reanalyze:
        print("  Re-analysis (with improved models):")
        print(f"    Filings re-analyzed:  {reanalyze.get('analyzed', 0)}")
        print(f"    Filings failed:       {reanalyze.get('failed', 0)}")
        print()

    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LexDrift Complete Pipeline -- ingest, analyze, train, re-analyze.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_all.py                     # Full run with 30 S&P 500 tickers\n"
            "  python scripts/run_all.py --skip-ingest       # Skip ingestion (data already exists)\n"
            "  python scripts/run_all.py --skip-train        # Skip training steps\n"
            "  python scripts/run_all.py --tickers AAPL,MSFT # Custom tickers\n"
        ),
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help=(
            "Comma-separated list of ticker symbols. "
            "Defaults to 30 major S&P 500 tickers."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum 10-K filings per company to ingest (default: 3)",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip step 1 (filing ingestion) if data already exists",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip steps 3-6 (training data generation and model training)",
    )
    parser.add_argument(
        "--skip-reanalyze",
        action="store_true",
        help="Skip step 7 (re-analysis with improved models)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the complete LexDrift pipeline."""
    global _pipeline_start

    args = _parse_args(argv)

    # Resolve tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = SP500_TICKERS

    # Register interrupt handler
    signal.signal(signal.SIGINT, _handle_interrupt)

    _pipeline_start = time.time()
    total_steps = 8
    summary: dict = {"models": {}}

    print("=" * 70)
    print("LexDrift Complete Pipeline")
    print(f"  Tickers: {len(tickers)} companies")
    print(f"  Filings per company: {args.limit}")
    print(f"  Skip ingest: {args.skip_ingest}")
    print(f"  Skip train:  {args.skip_train}")
    print(f"  Started at:  {_timestamp()}")
    print("=" * 70)

    # === Step 1: Ingest ===
    if args.skip_ingest:
        _step(1, total_steps, "Ingest SEC filings [SKIPPED]")
        _info("Skipping ingestion (--skip-ingest flag)")
        summary["ingest"] = {}
    else:
        _step(1, total_steps, "Ingest SEC filings")
        t0 = time.time()
        summary["ingest"] = step_ingest(tickers, args.limit)
        _info(f"Ingestion complete in {time.time() - t0:.0f}s")

    if _interrupted:
        step_print_stats(summary)
        return

    # === Step 2: Analyze ===
    _step(2, total_steps, "Analyze all filings")
    t0 = time.time()
    ingest_ids = summary.get("ingest", {}).get("filing_ids")
    summary["analyze"] = step_analyze(ingest_ids if not args.skip_ingest else None)
    _info(f"Analysis complete in {time.time() - t0:.0f}s")

    if _interrupted:
        step_print_stats(summary)
        return

    # === Step 3: Generate elite training data ===
    if args.skip_train:
        _step(3, total_steps, "Generate elite training data [SKIPPED]")
        _info("Skipping training data generation (--skip-train flag)")
        training_pairs = []
        summary["training_report"] = {}
    else:
        _step(3, total_steps, "Generate elite training data")
        t0 = time.time()
        training_pairs, report = step_generate_training_data()
        summary["training_report"] = report
        _info(f"Generated {len(training_pairs)} pairs in {time.time() - t0:.0f}s")
        _info(f"Tier breakdown: {report.get('tier_counts', {})}")

    if _interrupted:
        step_print_stats(summary)
        return

    # === Step 4: Fine-tune embedding model ===
    if args.skip_train:
        _step(4, total_steps, "Fine-tune embedding model [SKIPPED]")
        _info("Skipping fine-tuning (--skip-train flag)")
    else:
        _step(4, total_steps, "Fine-tune embedding model")
        t0 = time.time()
        try:
            model_path = step_finetune(training_pairs)
            summary["models"]["embedding"] = model_path
            _info(f"Fine-tuning complete in {time.time() - t0:.0f}s -> {model_path}")
        except Exception as e:
            _warn(f"Fine-tuning failed: {e}")
            summary["models"]["embedding"] = ""

    if _interrupted:
        step_print_stats(summary)
        return

    # === Step 5: Train risk classifier ===
    if args.skip_train:
        _step(5, total_steps, "Train risk classifier [SKIPPED]")
        _info("Skipping risk classifier (--skip-train flag)")
    else:
        _step(5, total_steps, "Train risk classifier")
        t0 = time.time()
        try:
            risk_path = step_train_risk_classifier()
            summary["models"]["risk_classifier"] = risk_path
            _info(f"Risk classifier trained in {time.time() - t0:.0f}s -> {risk_path}")
        except Exception as e:
            _warn(f"Risk classifier training failed: {e}")
            summary["models"]["risk_classifier"] = ""

    if _interrupted:
        step_print_stats(summary)
        return

    # === Step 6: Train boilerplate classifier ===
    if args.skip_train:
        _step(6, total_steps, "Train boilerplate classifier [SKIPPED]")
        _info("Skipping boilerplate classifier (--skip-train flag)")
    else:
        _step(6, total_steps, "Train boilerplate classifier")
        t0 = time.time()
        try:
            bp_path = step_train_boilerplate_classifier()
            summary["models"]["boilerplate_classifier"] = bp_path
            _info(f"Boilerplate classifier trained in {time.time() - t0:.0f}s -> {bp_path}")
        except Exception as e:
            _warn(f"Boilerplate classifier training failed: {e}")
            summary["models"]["boilerplate_classifier"] = ""

    if _interrupted:
        step_print_stats(summary)
        return

    # === Step 7: Re-analyze with improved models ===
    if args.skip_train or args.skip_reanalyze:
        _step(7, total_steps, "Re-analyze with improved models [SKIPPED]")
        _info("Skipping re-analysis")
        summary["reanalyze"] = {}
    else:
        _step(7, total_steps, "Re-analyze with improved models")
        t0 = time.time()
        summary["reanalyze"] = step_reanalyze()
        _info(f"Re-analysis complete in {time.time() - t0:.0f}s")

    if _interrupted:
        step_print_stats(summary)
        return

    # === Step 8: Final statistics ===
    _step(8, total_steps, "Final statistics")
    step_print_stats(summary)


if __name__ == "__main__":
    main()
