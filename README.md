# LexDrift — SEC Semantic Drift Analyzer

A financial intelligence system that tracks **how public companies change their language** across SEC filings (10-K, 10-Q, 8-K) over time. Instead of asking "what did the company say?", LexDrift answers **"what did the company say *differently*, and why does that matter?"**

When a company quietly adds "supply chain uncertainty" to its risk factors, shifts from "confident" to "cautiously optimistic," removes a growth target, or rewrites revenue recognition disclosures — these linguistic shifts are **leading indicators** of financial trouble, strategic pivots, or regulatory concerns.

---

## Quick Start

```bash
# 1. Install backend
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

# 2. Download Loughran-McDonald dictionary
# Get CSV from https://sraf.nd.edu/loughranmcdonald-master-dictionary/
# Save to data/Loughran-McDonald_MasterDictionary_1993-2024.csv

# 3. Start backend
uvicorn lexdrift.main:app --reload
# API at http://localhost:8000 — Swagger docs at http://localhost:8000/docs

# 4. Start frontend
cd frontend && npm install && npm run dev
# UI at http://localhost:3000

# 5. Ingest + analyze a company
curl -X POST "http://localhost:8000/filings/ingest/TSLA?form_type=10-K&limit=3"
curl -X POST "http://localhost:8000/filings/1/analyze"
```

Or run everything at once (ingest 30 S&P 500 companies, analyze, train models):

```bash
python scripts/run_all.py
```

---

## Architecture

```
SEC EDGAR ──→ Ingest ──→ Parse ──→ Analyze ──→ Score ──→ Alert
                │           │          │          │         │
            Download    iXBRL +    3-Layer     Risk     Anomaly
              HTML      Regex     Drift NLP   Scoring   Detection
                       Fallback              per sentence
```

### The 3-Layer Drift Engine

Every filing is compared against the previous filing of the same type:

| Layer | Method | What It Catches |
|---|---|---|
| **Section** | Cosine distance on chunked sentence-transformer embeddings | Overall semantic shift in a section |
| **Sentence** | NxN similarity matrix with greedy alignment | Specific sentences added, removed, or reworded |
| **Word** | Auto n-gram discovery (bigram/trigram frequency diff) | New/disappeared phrases without any hardcoded list |

Sentence alignment also detects **likely replacements** — when a company removes "workforce reduction of 2,000 employees" and adds "organizational realignment initiatives," the system links them even though no words overlap.

### 5 Novel Research Contributions

| Module | What It Does |
|---|---|
| **Adversarial Obfuscation Detection** | Detects when companies deliberately hide bad news: information density drop, specificity decrease, readability worsening, euphemism substitution |
| **Filing Entropy Analysis** | Shannon entropy, KL divergence, cross-entropy to distinguish genuinely novel disclosures from recycled boilerplate |
| **Semantic Kinematics** | Velocity, acceleration, jerk, momentum of drift over time. Phase classification: stable → drifting → accelerating → regime_change |
| **Cross-Company Risk Contagion** | Graph-based detection of risk language propagation through supply chains and sectors |
| **Latent Risk Trajectories** | UMAP/PCA projection of all filings into shared space. Tracks company trajectories toward distress zones |

---

## Project Structure

```
lexdrift/
├── src/lexdrift/
│   ├── main.py                     # FastAPI app — 29 API routes
│   ├── config.py                   # Pydantic settings (.env based)
│   ├── api/                        # REST endpoints
│   │   ├── companies.py            # Company search + lookup
│   │   ├── filings.py              # Ingest + section retrieval
│   │   ├── drift.py                # Analysis + drift timeline + screener
│   │   ├── alerts.py               # Watchlists + alert management
│   │   └── research.py             # Obfuscation, entropy, kinematics, contagion, latent space
│   ├── nlp/                        # 15 NLP modules
│   │   ├── drift.py                # Orchestrates all analysis with graceful degradation
│   │   ├── sentences.py            # Sentence-level alignment (capped at 300 for memory safety)
│   │   ├── embeddings.py           # Chunked mean-pooling, thread-safe model loading
│   │   ├── sentiment.py            # Loughran-McDonald 5-category scoring (3,861 words)
│   │   ├── risk.py                 # Per-sentence risk: critical/high/medium/low/boilerplate
│   │   ├── phrases.py              # Auto n-gram discovery + priority phrase tracking
│   │   ├── anomaly.py              # Company-specific z-scores + trend detection
│   │   ├── obfuscation.py          # Information density, specificity, readability, euphemisms
│   │   ├── entropy.py              # Shannon entropy, KL divergence, novelty scoring
│   │   ├── velocity.py             # Semantic velocity, acceleration, jerk, phase classification
│   │   ├── contagion.py            # Risk propagation graph with spectral analysis
│   │   ├── latent_space.py         # UMAP/PCA trajectory analysis, danger zone detection
│   │   ├── boilerplate.py          # Cross-company dedup + trained classifier
│   │   ├── diff.py                 # Unified text diffs
│   │   └── tokenizer.py            # Abbreviation-aware sentence splitting
│   ├── edgar/                      # SEC EDGAR integration
│   │   ├── client.py               # Rate-limited async HTTP (10 req/sec)
│   │   ├── tickers.py              # Ticker/CIK lookup (371K companies)
│   │   ├── filings.py              # Filing metadata + download
│   │   └── parser.py               # iXBRL → regex → title-based (3-pass)
│   ├── workers/                    # Celery background tasks
│   │   ├── ingest.py               # Filing download + parse
│   │   ├── analyze.py              # Full NLP pipeline
│   │   └── monitor.py              # EDGAR polling (every 30 min)
│   ├── training/                   # Self-supervised ML pipelines
│   │   ├── finetune.py             # Contrastive embedding fine-tuning
│   │   ├── data_quality.py         # 5-tier elite training data (text-level, no circular dependency)
│   │   ├── risk_classifier.py      # MLP risk severity classifier
│   │   └── boilerplate_classifier.py # Binary boilerplate classifier
│   └── db/
│       ├── models.py               # 9 SQLAlchemy tables
│       ├── engine.py               # Async engine (SQLite dev / PostgreSQL prod)
│       └── session.py              # FastAPI dependency
├── frontend/                       # Next.js 14 + Tailwind CSS
│   └── src/
│       ├── app/                    # Pages: dashboard, screener, company, alerts, watchlist
│       ├── components/             # Sidebar, header, theme toggle, data table, cards
│       └── lib/                    # Typed API client (20+ functions)
├── scripts/
│   ├── run_all.py                  # One-click: ingest → analyze → train → re-analyze
│   └── backfill.py                 # S&P 500 bulk ingestion (30 tickers)
├── tests/                          # 82 tests (unit + integration)
├── Dockerfile                      # Multi-stage Python 3.11
├── docker-compose.yml              # App + workers + Redis + PostgreSQL
└── data/
    ├── Loughran-McDonald_MasterDictionary_1993-2024.csv
    └── default_phrases.json
```

---

## API Endpoints (29 routes)

### Core
```
GET  /health
GET  /companies?q={search}
GET  /companies/{ticker}
GET  /companies/{ticker}/filings
GET  /companies/{ticker}/drift
GET  /companies/{ticker}/phrases
```

### Filing Operations
```
POST /filings/ingest/{ticker}?form_type=10-K&limit=3
GET  /filings/{id}
GET  /filings/{id}/sections/{type}
POST /filings/{id}/analyze?force=false
GET  /filings/{id}/diff?vs={prev_id}&section_type=risk_factors
```

### Drift Analysis
```
GET  /drift/screener?section_type=risk_factors&sort_by=cosine_distance
GET  /drift/{drift_score_id}/sentences?change_type=added
```

### Watchlists & Alerts
```
POST /watchlists
GET  /watchlists
POST /watchlists/{id}/companies
DELETE /watchlists/{id}/companies/{ticker}
GET  /alerts?unread=true
PUT  /alerts/{id}/read
```

### Research (Novel Features)
```
POST /research/obfuscation/{filing_id}
POST /research/entropy/{filing_id}
GET  /research/kinematics/{ticker}
GET  /research/overview/{ticker}
GET  /research/contagion/{ticker}
GET  /research/latent-space
```

---

## Database Schema

9 tables with proper indices:

- **companies** — CIK, ticker, name, SIC code
- **filings** — accession number, form type, dates, status
- **sections** — section text, word count, embedding (BLOB)
- **drift_scores** — cosine/Jaccard distance, sentiment delta, word changes
- **sentence_changes** — added/removed/changed sentences with similarity scores
- **key_phrases** — auto-discovered + priority phrase tracking
- **alerts** — drift_anomaly, critical_risk_language, phrase_change, obfuscation_detected
- **watchlists** + **watchlist_companies** — user watchlist management

---

## Training Pipelines

All three pipelines are self-supervised — zero human labels required.

### 1. Embedding Fine-Tuning
```bash
python -m lexdrift.training.finetune --elite --epochs 3
```
Uses 5-tier text-level training data (no circular model dependency):
- Tier 1: Exact match sentences across filings (label=1.0)
- Tier 2: High Jaccard overlap pairs (label=0.85-0.95)
- Tier 3: Cross-section hard negatives (label=0.0)
- Tier 4: Cross-company boilerplate positives (label=0.9)
- Tier 5: Outcome-anchored negatives (label=0.0)

### 2. Risk Classifier
```bash
python -m lexdrift.training.risk_classifier
```
2-layer MLP (384→128→4) bootstrapped from keyword system. Auto-loaded at inference time if `models/risk_classifier.pt` exists.

### 3. Boilerplate Classifier
```bash
python -m lexdrift.training.boilerplate_classifier
```
Binary MLP trained on cross-company sentence frequency. Sentences in 3+ companies = boilerplate.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + uvicorn (async) |
| Frontend | Next.js 14 + Tailwind CSS + Recharts |
| Database | SQLAlchemy 2.0 (SQLite dev / PostgreSQL prod) |
| Migrations | Alembic |
| NLP | sentence-transformers (all-MiniLM-L6-v2), spaCy, Loughran-McDonald lexicon |
| Task Queue | Celery + Redis |
| Deployment | Docker + docker-compose |
| HTTP Client | httpx (async, rate-limited) |

---

## Configuration

Copy `.env.example` to `.env` and configure:

```env
DATABASE_URL=sqlite+aiosqlite:///./lexdrift.db
SEC_USER_AGENT=LexDrift your-email@example.com
SEC_RATE_LIMIT=10
CELERY_BROKER_URL=redis://localhost:6379/0
EMBEDDING_MODEL=all-MiniLM-L6-v2
DRIFT_THRESHOLD=0.15
```

---

## Docker

```bash
docker-compose up
```

Runs: FastAPI app + 2 Celery workers + beat scheduler + Redis + PostgreSQL.

---

## Stats

| Metric | Count |
|---|---|
| Python files | 65 |
| Lines of code | 9,700+ |
| API routes | 29 |
| NLP modules | 15 |
| Tests | 82 (all passing) |
| DB tables | 9 |
| Training pipelines | 3 (self-supervised) |
| Frontend pages | 5 |

---

## Academic Foundation

| Paper | Key Finding |
|---|---|
| Cohen, Malloy & Nguyen (2020) "Lazy Prices" | Textual changes in 10-K/10-Q predict future stock returns |
| Loughran & McDonald (2011) | Financial-specific word lists outperform generic sentiment for SEC filings |
| Loughran & McDonald (2014) | Filing complexity correlates with worse outcomes |
| Hoberg & Lewis (2017) | 10-K text identifies peer firms and predicts competitive dynamics |
| Mayew, Sethuraman & Venkatachalam (2015) | MD&A linguistic features predict going-concern risk before auditors flag it |

---

## License

MIT
