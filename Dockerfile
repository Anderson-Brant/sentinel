# syntax=docker/dockerfile:1.6
#
# Sentinel — market intelligence & stock prediction platform.
#
# Multi-stage build:
#   1. builder — compiles any native wheels (psycopg, lightgbm, xgboost) and
#      installs sentinel plus its most useful extras into a self-contained
#      venv under /opt/venv.
#   2. runtime — slim Python image that copies the venv, adds a non-root
#      user, mounts /data as a volume, and runs the scheduler daemon by
#      default.
#
# Build:
#     docker build -t sentinel:latest .
#
# Run (DuckDB, default — persistent volume):
#     docker run --rm -v sentinel-data:/data sentinel:latest demo SPY
#
# Run (Postgres/Timescale via docker-compose):
#     docker compose --profile postgres up
#
# Run the scheduler daemon (the container's default CMD):
#     docker run --rm -v sentinel-data:/data sentinel:latest

ARG PYTHON_VERSION=3.11


# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Native build deps for wheels that don't publish manylinux binaries everywhere
# (psycopg[c], lightgbm, occasionally xgboost on fresh Python minors).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        libomp-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only what's needed to install the package — maximizes layer caching
# so that code changes in src/ don't invalidate dep resolution.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Production extras: social ingestion, gradient boosting, MLflow tracking,
# Postgres driver, CCXT crypto, SHAP importance. Skip the `[transformers]`
# extra (torch+finBERT are ~2GB) and the `[dev]` extra (tests don't ship).
RUN pip install --upgrade pip \
    && pip install ".[social,ml-extra,tracking,postgres,crypto,explain]"


# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    SENTINEL_DB_PATH=/data/sentinel.duckdb \
    SENTINEL_LOG_LEVEL=INFO

# Runtime C libs for the compiled wheels we installed above.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
        libgomp1 \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /bin/bash sentinel \
    && mkdir -p /data /app \
    && chown -R sentinel:sentinel /data /app

COPY --from=builder --chown=sentinel:sentinel /opt/venv /opt/venv
COPY --chown=sentinel:sentinel config /app/config
COPY --chown=sentinel:sentinel pyproject.toml README.md LICENSE /app/

WORKDIR /app
USER sentinel

# /data holds the DuckDB file by default; mount a named volume or host path
# so state survives container restarts. When running against Postgres this
# volume is unused (but harmless).
VOLUME ["/data"]

# `sentinel version` exercises the CLI import graph + pyproject metadata.
# If any required dep failed to resolve at build time, this will fail.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD sentinel version > /dev/null 2>&1 || exit 1

# tini reaps zombies for the long-running scheduler daemon; harmless for
# one-shot CLI invocations (`docker run sentinel demo SPY`).
ENTRYPOINT ["/usr/bin/tini", "--", "sentinel"]
CMD ["schedule", "run", "--forever"]

# OCI labels — appear in `docker inspect`, used by registries for discovery.
LABEL org.opencontainers.image.title="sentinel" \
      org.opencontainers.image.description="Market intelligence & stock prediction platform" \
      org.opencontainers.image.source="https://github.com/Anderson-Brant/sentinel" \
      org.opencontainers.image.licenses="MIT"
