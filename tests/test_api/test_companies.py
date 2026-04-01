"""Tests for GET /companies?q=... endpoint."""

from unittest.mock import AsyncMock, patch

import pytest

# The router imports search_companies into its own namespace, so we patch there.
_SEARCH_TARGET = "lexdrift.api.companies.search_companies"


@pytest.mark.asyncio
@patch(
    _SEARCH_TARGET,
    new_callable=AsyncMock,
    return_value=[
        {"cik": "123456", "ticker": "TEST", "name": "Test Corp"},
        {"cik": "789012", "ticker": "TSTR", "name": "Tester Inc"},
    ],
)
async def test_search_companies_returns_results(mock_search, client):
    response = await client.get("/companies", params={"q": "test"})
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "total" in data
    assert data["total"] == 2
    assert data["data"][0]["ticker"] == "TEST"


@pytest.mark.asyncio
@patch(
    _SEARCH_TARGET,
    new_callable=AsyncMock,
    return_value=[],
)
async def test_search_companies_empty_results(mock_search, client):
    response = await client.get("/companies", params={"q": "xyznonexistent"})
    assert response.status_code == 200
    data = response.json()
    assert data["data"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_search_companies_missing_query(client):
    response = await client.get("/companies")
    # FastAPI returns 422 for missing required query param
    assert response.status_code == 422


@pytest.mark.asyncio
@patch(
    _SEARCH_TARGET,
    new_callable=AsyncMock,
    return_value=[{"cik": "111", "ticker": "A", "name": "Alpha"}],
)
async def test_search_companies_calls_search_with_query(mock_search, client):
    await client.get("/companies", params={"q": "alpha", "limit": "5"})
    mock_search.assert_awaited_once_with("alpha", limit=5)
