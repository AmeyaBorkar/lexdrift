import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lexdrift.api.alerts import router as alerts_router
from lexdrift.api.companies import router as companies_router
from lexdrift.api.drift import router as drift_router
from lexdrift.api.filings import router as filings_router
from lexdrift.api.research import router as research_router
from lexdrift.db.engine import engine
from lexdrift.db.models import Base
from lexdrift.edgar.client import edgar_client
from lexdrift.nlp.phrases import bootstrap_corpus_from_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup (dev convenience; use Alembic in production)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Bootstrap TF-IDF corpus from existing sections to avoid cold-start
    try:
        count = bootstrap_corpus_from_db()
        logger.info("TF-IDF corpus bootstrapped from %d sections", count)
    except Exception:
        logger.warning("Failed to bootstrap TF-IDF corpus", exc_info=True)
    yield
    await edgar_client.close()
    await engine.dispose()


app = FastAPI(
    title="LexDrift",
    description="SEC Semantic Drift Analyzer",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(companies_router)
app.include_router(filings_router)
app.include_router(drift_router)
app.include_router(alerts_router)
app.include_router(research_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "lexdrift"}
