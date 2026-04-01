import json
import logging
from pathlib import Path

from lexdrift.edgar.client import edgar_client

logger = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
CACHE_FILE = Path("data/company_tickers.json")

# In-memory cache: ticker -> {cik, name}, cik -> {ticker, name}
_by_ticker: dict[str, dict] = {}
_by_cik: dict[str, dict] = {}


def _index(raw: dict) -> None:
    """Build lookup indexes from the SEC company_tickers.json structure."""
    _by_ticker.clear()
    _by_cik.clear()
    for entry in raw.values():
        cik = str(entry["cik_str"])
        ticker = entry["ticker"].upper()
        name = entry["title"]
        record = {"cik": cik, "ticker": ticker, "name": name}
        _by_ticker[ticker] = record
        _by_cik[cik] = record


async def load_tickers(force_refresh: bool = False) -> None:
    """Load the ticker→CIK mapping. Uses local cache if available."""
    if _by_ticker and not force_refresh:
        return

    if CACHE_FILE.exists() and not force_refresh:
        logger.info("Loading tickers from cache")
        raw = json.loads(CACHE_FILE.read_text())
        _index(raw)
        return

    logger.info("Fetching tickers from SEC EDGAR")
    raw = await edgar_client.get_json(TICKERS_URL)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(raw))
    _index(raw)
    logger.info(f"Loaded {len(_by_ticker)} tickers")


async def lookup_ticker(ticker: str) -> dict | None:
    """Look up a company by ticker symbol. Returns {cik, ticker, name} or None."""
    await load_tickers()
    return _by_ticker.get(ticker.upper())


async def lookup_cik(cik: str) -> dict | None:
    """Look up a company by CIK number. Returns {cik, ticker, name} or None."""
    await load_tickers()
    return _by_cik.get(cik)


async def search_companies(query: str, limit: int = 20) -> list[dict]:
    """Search companies by name or ticker substring."""
    await load_tickers()
    query_upper = query.upper()
    results = []
    for record in _by_ticker.values():
        if query_upper in record["ticker"] or query_upper in record["name"].upper():
            results.append(record)
            if len(results) >= limit:
                break
    return results


def pad_cik(cik: str) -> str:
    """Pad CIK to 10 digits for SEC API URLs."""
    return cik.zfill(10)
