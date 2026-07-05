from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ops_common.config import settings
from ops_common.logging import get_logger

logger = get_logger(__name__)

# Module-level singletons: one engine + one session factory shared across the app.
# The lock guards against two threads creating the engine at the same time.
_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None
_engine_lock = threading.Lock()


# ============================================================
# Postgres (the hub) — SQLAlchemy engine + session
# ============================================================

# Lazily build the Postgres engine once (thread-safe double-checked locking).
# pool_pre_ping checks dead connections; pool_size/max_overflow tune concurrency.
def get_engine() -> Engine:
    global _engine, _SessionFactory
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                logger.info("Creating Postgres engine", extra={"host": settings.postgres_host})
                _engine = create_engine(
                    settings.sqlalchemy_dsn,
                    pool_pre_ping=True,
                    pool_size=5,
                    max_overflow=10,
                    future=True,
                )
                # Session factory configured for explicit commits (autocommit off).
                _SessionFactory = sessionmaker(
                    bind=_engine,
                    autoflush=False,
                    autocommit=False,
                    expire_on_commit=False,
                    future=True,
                )
    return _engine


# Return the session factory, building the engine first if needed.
def get_session_factory() -> sessionmaker[Session]:
    if _SessionFactory is None:
        get_engine()
    assert _SessionFactory is not None
    return _SessionFactory


# Context manager for a unit of work: commit on success, rollback on error,
# always close. Use this for scripts/pipelines (with session_scope() as s:).
@contextmanager
def session_scope() -> Iterator[Session]:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Session rolled back due to error")
        raise
    finally:
        session.close()


# FastAPI dependency version: yields a session, no auto-commit (routes commit
# themselves). Used via Depends(get_db) in API endpoints.
def get_db() -> Iterator[Session]:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


# Startup helper: poll Postgres until it answers SELECT 1 or retries run out.
# Needed because in Docker the API can boot before Postgres is ready.
def wait_for_postgres(retries: int = 30, delay: float = 2.0) -> None:
    import time

    engine = get_engine()
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Postgres is ready", extra={"attempt": attempt})
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning(
                "Postgres not ready, retrying",
                extra={"attempt": attempt, "retries": retries},
            )
            time.sleep(delay)
    raise RuntimeError(f"Postgres unavailable after {retries} attempts") from last_err


# Run a .sql file against Postgres in one transaction. This is how the hub
# schema gets applied at startup (idempotent schema.sql).
def apply_schema(schema_path: str | Path) -> None:
    schema_path = Path(schema_path)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    sql = schema_path.read_text(encoding="utf-8")
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(sql))
    logger.info("Applied Postgres schema", extra={"path": str(schema_path)})


# ============================================================
# DuckDB (analytics) — attaches Postgres read-only, loads views
# ============================================================

# Open a DuckDB connection and wire it to Postgres: install+load the postgres
# extension, then attach the live Postgres DB so DuckDB can query hub tables.
# This is the "fast analytics reads off the same data" mechanism.
def get_duckdb(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    Path(settings.duckdb_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(settings.duckdb_path, read_only=read_only)
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    _attach_postgres(conn)
    return conn


# Attach Postgres into DuckDB as READ_ONLY — but only once. First checks if the
# alias is already attached (avoids the slow re-attach / duplicate error).
def _attach_postgres(conn: duckdb.DuckDBPyConnection) -> None:
    alias = settings.duckdb_pg_alias
    attached = conn.execute(
        "SELECT count(*) FROM duckdb_databases() WHERE database_name = ?",
        [alias],
    ).fetchone()
    if attached and attached[0] > 0:
        return
    conn.execute(
        f"ATTACH '{settings.duckdb_attach_dsn}' AS {alias} "
        f"(TYPE postgres, READ_ONLY);"
    )
    logger.info("Attached Postgres to DuckDB", extra={"alias": alias})


# Run the analytics .sql file (the 6 DuckDB views) against DuckDB once at startup.
# These views sit on top of the attached Postgres data for fast reads.
def load_analytics_views(analytics_sql_path: str | Path) -> None:
    analytics_sql_path = Path(analytics_sql_path)
    if not analytics_sql_path.exists():
        raise FileNotFoundError(f"Analytics SQL not found: {analytics_sql_path}")
    sql = analytics_sql_path.read_text(encoding="utf-8")
    conn = get_duckdb(read_only=False)
    try:
        conn.execute(sql)
        logger.info("Loaded DuckDB analytics views", extra={"path": str(analytics_sql_path)})
    finally:
        conn.close()


# Context manager for DuckDB reads: opens (read-only by default), yields, closes.
# Use this when the dashboard/API needs to run an analytics query.
@contextmanager
def duckdb_scope(read_only: bool = True) -> Iterator[duckdb.DuckDBPyConnection]:
    conn = get_duckdb(read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()