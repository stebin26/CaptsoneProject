"""Database access for Postgres (the hub) and DuckDB (analytics).

Postgres is the system of record, reached through a single shared SQLAlchemy
engine and session factory. DuckDB attaches that same Postgres database
read-only, so analytical queries run fast against one copy of the data rather
than a second, drifting one. Engine creation is lazy and thread-safe, and both
databases expose context managers so callers cannot leak connections.
"""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
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
    """Return the shared Postgres engine, creating it on first use.

    Uses double-checked locking so concurrent callers cannot build two engines.
    The pool pre-pings to drop dead connections, and the session factory is built
    alongside it with autocommit off.

    Returns:
        The process-wide SQLAlchemy engine.
    """
    global _engine, _SessionFactory
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                logger.info(
                    "Creating Postgres engine", extra={"host": settings.postgres_host}
                )
                try:
                    engine = create_engine(
                        settings.sqlalchemy_dsn,
                        pool_pre_ping=True,
                        pool_size=5,
                        max_overflow=10,
                        future=True,
                    )
                    # Configured for explicit commits (autocommit off).
                    factory = sessionmaker(
                        bind=engine,
                        autoflush=False,
                        autocommit=False,
                        expire_on_commit=False,
                        future=True,
                    )
                except Exception:
                    logger.exception(
                        "Could not create the Postgres engine for %s:%s/%s",
                        settings.postgres_host,
                        settings.postgres_port,
                        settings.postgres_db,
                        extra={
                            "db_host": settings.postgres_host,
                            "db_port": settings.postgres_port,
                            "db_name": settings.postgres_db,
                        },
                    )
                    raise

                # Published together: a caller must never see an engine without
                # its session factory.
                _engine = engine
                _SessionFactory = factory
    return _engine


# Return the session factory, building the engine first if needed.
def get_session_factory() -> sessionmaker[Session]:
    """Return the shared session factory, building the engine if needed.

    Returns:
        The process-wide session factory.
    """
    if _SessionFactory is None:
        get_engine()
    if _SessionFactory is None:
        # A bare assert would be stripped under `python -O`, turning this into a
        # confusing NoneType error much further downstream.
        raise RuntimeError(
            "Session factory was not initialised; the Postgres engine is missing."
        )
    return _SessionFactory


# Context manager for a unit of work: commit on success, rollback on error,
# always close. Use this for scripts/pipelines (with session_scope() as s:).
@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional session scope for scripts and pipelines.

    Commits on success, rolls back and logs on failure, and always closes.

    Yields:
        An active session for the duration of the block.

    Raises:
        Exception: Re-raises whatever the block raised, after rolling back.
    """
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
    """Yield a request-scoped session for use as a FastAPI dependency.

    Unlike ``session_scope`` this does not commit; routes commit themselves.

    Yields:
        An active session, closed when the request ends.
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


# Startup helper: poll Postgres until it answers SELECT 1 or retries run out.
# Needed because in Docker the API can boot before Postgres is ready.
def wait_for_postgres(retries: int = 30, delay: float = 2.0) -> None:
    """Block until Postgres answers a trivial query, or give up.

    Needed because in Docker a service can start before Postgres is accepting
    connections.

    Args:
        retries: Maximum number of attempts.
        delay: Seconds to wait between attempts.

    Raises:
        RuntimeError: If Postgres is still unreachable after every attempt.
    """
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
            # The reason matters here: "not ready" reads the same whether the
            # container is still booting or the password is wrong.
            logger.warning(
                "Postgres not ready (%s), retrying",
                exc,
                extra={
                    "attempt": attempt,
                    "retries": retries,
                    "db_host": settings.postgres_host,
                },
            )
            time.sleep(delay)

    logger.error(
        "Postgres never became available at %s:%s/%s after %d attempts",
        settings.postgres_host,
        settings.postgres_port,
        settings.postgres_db,
        retries,
        extra={"db_host": settings.postgres_host, "retries": retries},
    )
    raise RuntimeError(f"Postgres unavailable after {retries} attempts") from last_err


# Run a .sql file against Postgres in one transaction. This is how the hub
# schema gets applied at startup (idempotent schema.sql).
def apply_schema(schema_path: str | Path) -> None:
    """Apply a ``.sql`` schema file to Postgres in a single transaction.

    Used at startup with idempotent schema files, so booting twice is harmless.

    Args:
        schema_path: Path to the SQL file to execute.

    Raises:
        FileNotFoundError: If the schema file does not exist.
    """
    schema_path = Path(schema_path)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    try:
        sql = schema_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.exception("Could not read schema file", extra={"path": str(schema_path)})
        raise OSError(f"Schema file could not be read: {schema_path}") from exc

    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text(sql))
    except SQLAlchemyError:
        # Naming the file is the point: several schemas are applied at startup
        # and the raw error says only that some statement failed.
        logger.exception(
            "Failed to apply schema file %s",
            schema_path.name,
            extra={"path": str(schema_path)},
        )
        raise

    logger.info("Applied Postgres schema", extra={"path": str(schema_path)})


# ============================================================
# DuckDB (analytics) — attaches Postgres read-only, loads views
# ============================================================


# Open a DuckDB connection and wire it to Postgres: install+load the postgres
# extension, then attach the live Postgres DB so DuckDB can query hub tables.
# This is the "fast analytics reads off the same data" mechanism.
def get_duckdb(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with Postgres attached read-only.

    Installs and loads the postgres extension, then attaches the live Postgres
    database so DuckDB can query the hub tables directly.

    Args:
        read_only: Open the DuckDB file itself read-only.

    Returns:
        An open DuckDB connection.

    Raises:
        RuntimeError: If the DuckDB file cannot be opened or the postgres
            extension cannot be loaded.
    """
    Path(settings.duckdb_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        conn = duckdb.connect(settings.duckdb_path, read_only=read_only)
    except Exception as exc:
        # DuckDB takes an exclusive lock on the file, so the usual cause is
        # another process in this stack still holding it.
        logger.exception(
            "Could not open the DuckDB file at %s (read_only=%s) — another "
            "process may still hold the lock",
            settings.duckdb_path,
            read_only,
            extra={"duckdb_path": settings.duckdb_path, "read_only": read_only},
        )
        raise RuntimeError(
            f"DuckDB could not be opened at {settings.duckdb_path}: {exc}"
        ) from exc

    try:
        conn.execute("INSTALL postgres;")
        conn.execute("LOAD postgres;")
        _attach_postgres(conn)
    except Exception as exc:
        # The connection is closed here because the caller never receives it and
        # would otherwise leave the file locked.
        logger.exception(
            "Could not prepare the DuckDB postgres extension",
            extra={"duckdb_path": settings.duckdb_path},
        )
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to close the DuckDB connection", exc_info=True)
        raise RuntimeError(f"DuckDB postgres extension unavailable: {exc}") from exc

    return conn


# Attach Postgres into DuckDB as READ_ONLY — but only once. First checks if the
# alias is already attached (avoids the slow re-attach / duplicate error).
def _attach_postgres(conn: duckdb.DuckDBPyConnection) -> None:
    """Attach Postgres into DuckDB read-only, at most once per connection.

    Checks whether the alias is already attached first, since re-attaching is both
    slow and an error.

    Args:
        conn: The DuckDB connection to attach into.
    """
    alias = settings.duckdb_pg_alias
    attached = conn.execute(
        "SELECT count(*) FROM duckdb_databases() WHERE database_name = ?",
        [alias],
    ).fetchone()
    if attached and attached[0] > 0:
        return

    try:
        conn.execute(
            f"ATTACH '{settings.duckdb_attach_dsn}' "
            f"AS {alias} (TYPE postgres, READ_ONLY);"
        )
    except Exception:
        # The DSN is built from the same settings as the SQLAlchemy one, so a
        # failure here is Postgres being unreachable rather than a typo.
        logger.exception(
            "Could not attach Postgres to DuckDB as %s",
            alias,
            extra={"alias": alias, "db_host": settings.postgres_host},
        )
        raise

    logger.info("Attached Postgres to DuckDB", extra={"alias": alias})


# Run the analytics .sql file (the 6 DuckDB views) against DuckDB once at startup.
# These views sit on top of the attached Postgres data for fast reads.
def load_analytics_views(analytics_sql_path: str | Path) -> None:
    """Execute the analytics SQL file to create the DuckDB views.

    Run once at startup; the views sit on top of the attached Postgres data.

    Args:
        analytics_sql_path: Path to the analytics SQL file.

    Raises:
        FileNotFoundError: If the analytics SQL file does not exist.
    """
    analytics_sql_path = Path(analytics_sql_path)
    if not analytics_sql_path.exists():
        raise FileNotFoundError(f"Analytics SQL not found: {analytics_sql_path}")

    try:
        sql = analytics_sql_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.exception(
            "Could not read the analytics SQL file",
            extra={"path": str(analytics_sql_path)},
        )
        raise OSError(
            f"Analytics SQL could not be read: {analytics_sql_path}"
        ) from exc

    conn = get_duckdb(read_only=False)
    try:
        conn.execute(sql)
        logger.info(
            "Loaded DuckDB analytics views", extra={"path": str(analytics_sql_path)}
        )
    except Exception:
        logger.exception(
            "Failed to create the DuckDB analytics views",
            extra={"path": str(analytics_sql_path)},
        )
        raise
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            # Never allowed to mask the real error above.
            logger.warning(
                "Failed to close the DuckDB connection cleanly", exc_info=True
            )


# Context manager for DuckDB reads: opens (read-only by default), yields, closes.
# Use this when the dashboard/API needs to run an analytics query.
@contextmanager
def duckdb_scope(read_only: bool = True) -> Iterator[duckdb.DuckDBPyConnection]:
    """Provide a DuckDB connection scope that always closes.

    Args:
        read_only: Open the DuckDB file read-only (the default for queries).

    Yields:
        An open DuckDB connection for the duration of the block.
    """
    conn = get_duckdb(read_only=read_only)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            # A close failure must not replace whatever the block raised, but an
            # unclosed DuckDB handle keeps the file locked, so it is recorded.
            logger.warning(
                "Failed to close the DuckDB connection cleanly", exc_info=True
            )
