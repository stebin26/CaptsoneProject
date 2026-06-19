# ============================================================================
# Industrial Operations Intelligence Platform
# PostgreSQL container for the centralized Data Hub
# ============================================================================
# Based on the official Postgres image. Any .sql file placed in
# /docker-entrypoint-initdb.d/ is run automatically the FIRST time the
# container starts with an empty data directory. We copy the hub schema in
# so the database self-initializes — no manual setup, no extra step.
#
# Note: Alembic is the source of truth for schema *versioning* (migrations).
# This baked-in schema gives a clean, instantly-ready DB for `docker compose up`
# and demo-day seeding. For an existing/persisted volume, Alembic migrations
# drive forward changes. The two are kept identical by construction.
# ============================================================================

FROM postgres:16-alpine

# Standard Postgres env (overridden by docker-compose / .env at runtime)
ENV POSTGRES_DB=ops_hub
ENV POSTGRES_USER=ops
ENV POSTGRES_PASSWORD=ops

# Auto-run the hub schema on first initialization of an empty data dir.
# Files in this directory execute in alphabetical order, hence the 01_ prefix.
COPY data-hub/postgres/schema.sql /docker-entrypoint-initdb.d/01_schema.sql

EXPOSE 5432