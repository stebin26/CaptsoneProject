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

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None
_engine_lock = threading.Lock()


# ============================================================
# Postgres (the hub) — SQLAlchemy engine + session
# ============================================================

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
                _SessionFactory = sessionmaker(
                    bind=_engine,
                    autoflush=False,
                    autocommit=False,
                    expire_on_commit=False,
                    future=True,
                )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    if _SessionFactory is None:
        get_engine()
    assert _SessionFactory is not None
    return _SessionFactory


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


# FastAPI dependency
def get_db() -> Iterator[Session]:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


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

def get_duckdb(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    Path(settings.duckdb_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(settings.duckdb_path, read_only=read_only)
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    _attach_postgres(conn)
    return conn


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


@contextmanager
def duckdb_scope(read_only: bool = True) -> Iterator[duckdb.DuckDBPyConnection]:
    conn = get_duckdb(read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()