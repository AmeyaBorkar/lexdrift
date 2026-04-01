"""End-to-end integration tests exercising the full pipeline through the API."""

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

FAKE_TICKER_INFO = {"cik": "12345", "ticker": "TEST", "name": "Test Corp"}

RISK_FACTORS_PARAGRAPH = (
    "The company faces significant uncertainty in its operations. "
    "Market conditions remain challenging and there is no assurance of future profitability. "
    "Material weakness in internal controls has been identified. "
    "Going concern risks exist due to declining revenue and mounting debt obligations. "
    "Regulatory changes could adversely affect operations. "
    "Cybersecurity threats continue to escalate. "
    "Competition in the industry is intensifying. "
    "Key personnel retention remains a concern. "
    "Macroeconomic volatility may reduce consumer demand. "
    "Supply chain disruptions could impact production timelines."
)

MDNA_PARAGRAPH = (
    "Revenue declined 15% year over year. "
    "The company implemented restructuring initiatives affecting approximately 500 employees. "
    "Operating margins contracted due to supply chain disruptions and increased input costs. "
    "Gross profit decreased from $450M to $380M driven by higher raw material costs. "
    "SGA expenses were reduced by 8% through workforce optimization measures. "
    "Cash flow from operations fell to $120M compared to $200M in the prior year. "
    "Capital expenditures totaled $75M primarily for technology infrastructure upgrades. "
    "The company repaid $50M in long term debt during the fiscal year. "
    "Working capital improved marginally due to better inventory management. "
    "Management expects gradual recovery driven by new product launches in Q3."
)

MOCK_10K_HTML = (
    "<html><body>"
    "<p>Item 1A. Risk Factors</p>"
    + "".join(f"<p>{RISK_FACTORS_PARAGRAPH}</p>" for _ in range(10))
    + "<p>Item 7. Management's Discussion and Analysis</p>"
    + "".join(f"<p>{MDNA_PARAGRAPH}</p>" for _ in range(10))
    + "</body></html>"
)

FAKE_METADATA = {
    "filings": {
        "recent": {
            "accessionNumber": ["0001234567-24-000001", "0001234567-24-000002"],
            "form": ["10-K", "10-K"],
            "filingDate": ["2024-03-15", "2023-03-14"],
            "reportDate": ["2024-01-31", "2023-01-31"],
            "primaryDocument": ["filing1.htm", "filing2.htm"],
        }
    }
}


# ---------------------------------------------------------------------------
# Test 1: Full ingest and analyze pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_ingest_and_analyze_pipeline(client: AsyncClient, db_session):
    """Ingest filings for a ticker, then verify filing and section retrieval."""
    with (
        patch(
            "lexdrift.api.filings.lookup_ticker",
            new_callable=AsyncMock,
            return_value=FAKE_TICKER_INFO,
        ),
        patch(
            "lexdrift.api.filings.get_filing_metadata",
            new_callable=AsyncMock,
            return_value=FAKE_METADATA,
        ),
        patch(
            "lexdrift.api.filings.download_filing",
            new_callable=AsyncMock,
            return_value=MOCK_10K_HTML,
        ),
    ):
        # POST /filings/ingest/TEST?form_type=10-K&limit=2
        resp = await client.post("/filings/ingest/TEST?form_type=10-K&limit=2")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ticker"] == "TEST"
        assert body["count"] == 2

        for entry in body["ingested"]:
            assert entry["status"] == "parsed"
            assert "risk_factors" in entry["sections"]
            assert "mdna" in entry["sections"]

    # GET /filings/1 -- verify filing exists (no mocks needed, data is in DB)
    resp = await client.get("/filings/1")
    assert resp.status_code == 200
    filing = resp.json()
    assert filing["form_type"] == "10-K"
    section_types = [s["type"] for s in filing["sections"]]
    assert "risk_factors" in section_types
    assert "mdna" in section_types

    # GET /filings/1/sections/risk_factors -- verify section text
    resp = await client.get("/filings/1/sections/risk_factors")
    assert resp.status_code == 200
    section = resp.json()
    assert section["section_type"] == "risk_factors"
    assert len(section["text"]) > 500
    assert section["word_count"] > 0


# ---------------------------------------------------------------------------
# Test 2: Watchlist lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_watchlist_lifecycle(client: AsyncClient, db_session):
    """Create a watchlist, add a company, then remove it."""
    # POST /watchlists
    resp = await client.post("/watchlists", json={"name": "My List"})
    assert resp.status_code == 200
    wl = resp.json()
    assert wl["name"] == "My List"
    watchlist_id = wl["id"]

    # GET /watchlists -- verify it's there
    resp = await client.get("/watchlists")
    assert resp.status_code == 200
    data = resp.json()
    names = [w["name"] for w in data["data"]]
    assert "My List" in names

    # Add a company (mock ticker lookup)
    with patch(
        "lexdrift.api.alerts.lookup_ticker",
        new_callable=AsyncMock,
        return_value=FAKE_TICKER_INFO,
    ):
        resp = await client.post(
            f"/watchlists/{watchlist_id}/companies",
            json={"ticker": "TEST"},
        )
        assert resp.status_code == 200
        assert resp.json()["ticker"] == "TEST"

    # Remove the company (mock ticker lookup for the delete as well)
    with patch(
        "lexdrift.api.alerts.lookup_ticker",
        new_callable=AsyncMock,
        return_value=FAKE_TICKER_INFO,
    ):
        resp = await client.delete(f"/watchlists/{watchlist_id}/companies/TEST")
        assert resp.status_code == 200
        assert resp.json()["removed"] is True


# ---------------------------------------------------------------------------
# Test 3: Alert listing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_alert_listing(client: AsyncClient, db_session):
    """Verify alerts endpoints return empty results when no alerts exist."""
    # GET /alerts -- empty initially
    resp = await client.get("/alerts")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == []
    assert body["total"] == 0

    # GET /alerts?unread=true -- also empty
    resp = await client.get("/alerts?unread=true")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == []
    assert body["total"] == 0
