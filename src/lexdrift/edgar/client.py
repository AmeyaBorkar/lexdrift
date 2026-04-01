import asyncio
import logging

import httpx

from lexdrift.config import settings

logger = logging.getLogger(__name__)

_semaphore = asyncio.Semaphore(settings.sec_rate_limit)


class EdgarClient:
    """Rate-limited async HTTP client for SEC EDGAR APIs."""

    BASE_URL = "https://data.sec.gov"
    WWW_URL = "https://www.sec.gov"
    EFTS_URL = "https://efts.sec.gov/LATEST"

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": settings.sec_user_agent},
                timeout=httpx.Timeout(30.0),
                follow_redirects=True,
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def get(self, url: str, retries: int = 3) -> httpx.Response:
        """Rate-limited GET request with retries."""
        for attempt in range(retries):
            async with _semaphore:
                try:
                    response = await self.client.get(url)
                    if response.status_code == 429:
                        wait = 2 ** attempt
                        logger.warning(f"Rate limited by SEC, waiting {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    response.raise_for_status()
                    return response
                except httpx.HTTPStatusError:
                    if attempt == retries - 1:
                        raise
                    await asyncio.sleep(1)
                except httpx.RequestError as e:
                    if attempt == retries - 1:
                        raise
                    logger.warning(f"Request error (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(1)
        raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")

    async def get_json(self, url: str) -> dict:
        response = await self.get(url)
        return response.json()

    async def get_text(self, url: str) -> str:
        response = await self.get(url)
        return response.text


edgar_client = EdgarClient()
