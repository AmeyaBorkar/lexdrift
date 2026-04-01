# ---------------------------------------------------------------------------
# Stage 1 -- install Python dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed for building wheels (lxml, scipy, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

# Install project dependencies (prod extras included for asyncpg)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .[prod]

# Download the spaCy language model
RUN python -m spacy download en_core_web_sm

# ---------------------------------------------------------------------------
# Stage 2 -- final runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini .
COPY pyproject.toml .
COPY data/ data/

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "lexdrift.main:app", "--host", "0.0.0.0", "--port", "8000"]
