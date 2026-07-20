"""FastAPI application entrypoint for the API gateway.

Wires the whole HTTP surface together: applies the Postgres schemas and DuckDB
analytics views at startup, configures logging and CORS, installs handlers that
translate common domain errors into clean HTTP responses, exposes liveness and
database health probes, and mounts every versioned router under ``/api/v1``.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from importlib import import_module
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from ops_common.config import settings
from ops_common.db import (
    apply_schema,
    load_analytics_views,
    wait_for_postgres,
)
from ops_common.logging import configure_logging, get_logger

from api_app.auth import routes as auth_routes
from api_app.routers.v1 import (
    agent,
    analytics,
    domains,
    evaluation,
    executive,
    features,
    intelligence,
    ml,
    onboard,
    rag,
)

logger = get_logger(__name__)

_SCHEMA_PATH = Path("/app/data-hub/postgres/schema.sql")
_ANALYTICS_PATH = Path("/app/data-hub/duckdb/analytics.sql")
_ML_SCHEMA_PATH = Path("/app/data-hub/postgres/ml_schema.sql")
_RAG_SCHEMA_PATH = Path("/app/data-hub/postgres/rag_schema.sql")
_AGENT_SCHEMA_PATH = Path("/app/data-hub/postgres/agent_schema.sql")
_AUTH_SCHEMA_PATH = Path("/app/data-hub/postgres/auth_schema.sql")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare and tear down application-wide resources.

    Runs once on startup: configures logging, waits for Postgres, applies the core,
    ML, RAG, agent, and auth schemas, loads the DuckDB analytics views, and ensures
    the configured working directories exist. Each optional schema is applied
    defensively so one missing or failing file cannot stop the service from booting.

    Args:
        app: The FastAPI application being started.

    Yields:
        Control back to FastAPI for the lifetime of the application.
    """
    configure_logging()
    logger.info("API gateway starting", extra={"env": settings.environment})

    try:
        wait_for_postgres()
    except Exception:
        # Nothing below can work without the database, so this is fatal — but it
        # is logged plainly, because a container that exits silently at boot is
        # the hardest kind of failure to diagnose.
        logger.exception(
            "Postgres did not become available; the API cannot start",
            extra={"db_host": settings.postgres_host, "db_name": settings.postgres_db},
        )
        raise

    if _SCHEMA_PATH.exists():
        try:
            apply_schema(_SCHEMA_PATH)
        except Exception:
            # Unlike the optional schemas below, the core schema is required:
            # every endpoint reads tables it defines.
            logger.exception(
                "Failed to apply the core schema; the API cannot start",
                extra={"path": str(_SCHEMA_PATH)},
            )
            raise
    else:
        logger.warning(
            "Schema file not found at startup", extra={"path": str(_SCHEMA_PATH)}
        )

    if _ML_SCHEMA_PATH.exists():
        try:
            apply_schema(_ML_SCHEMA_PATH)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to apply ML schema")
    else:
        logger.warning(
            "ML schema file not found at startup", extra={"path": str(_ML_SCHEMA_PATH)}
        )

    if _RAG_SCHEMA_PATH.exists():
        try:
            apply_schema(_RAG_SCHEMA_PATH)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to apply RAG schema")
    else:
        logger.warning(
            "RAG schema file not found at startup",
            extra={"path": str(_RAG_SCHEMA_PATH)},
        )

    if _AGENT_SCHEMA_PATH.exists():
        try:
            apply_schema(_AGENT_SCHEMA_PATH)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to apply agent schema")
    else:
        logger.warning(
            "Agent schema file not found at startup",
            extra={"path": str(_AGENT_SCHEMA_PATH)},
        )

    if _AUTH_SCHEMA_PATH.exists():
        try:
            apply_schema(_AUTH_SCHEMA_PATH)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to apply auth schema")
    else:
        logger.warning(
            "Auth schema file not found at startup",
            extra={"path": str(_AUTH_SCHEMA_PATH)},
        )

    if _ANALYTICS_PATH.exists():
        try:
            load_analytics_views(_ANALYTICS_PATH)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load DuckDB analytics views")
    else:
        logger.warning(
            "Analytics SQL not found at startup", extra={"path": str(_ANALYTICS_PATH)}
        )

    try:
        settings.ensure_dirs()
    except OSError:
        # Uploads land in these directories; starting without them would turn
        # every upload into a confusing runtime error instead.
        logger.exception(
            "Could not create the configured working directories",
            extra={"upload_dir": str(settings.upload_dir)},
        )
        raise

    logger.info("API gateway ready")

    yield

    logger.info("API gateway shutting down")


app = FastAPI(
    title="Operations Intelligence Platform",
    description="Industry-agnostic data onboarding and intelligence hub.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Domain errors -> HTTP status codes
#
# Translating these once, here, keeps every router free of the same repeated
# try/except and guarantees one consistent response shape. The packages are
# mounted into the image at fixed paths by the routers imported above, so the
# imports are defensive: a missing optional package must degrade the mapping,
# not stop the API from booting.
# ============================================================


def _domain_error_handlers() -> list[tuple[type[Exception], int, str]]:
    """Collect the domain exception classes that have an HTTP meaning.

    Returns:
        Tuples of exception class, HTTP status code, and a short label used in
        the log line. Classes that cannot be imported are skipped.
    """
    wanted = [
        # A malformed or unreadable upload is the client's problem, not a bug.
        ("app.connectors.base", "SourceValidationError", 400, "invalid upload"),
        ("extractor", "ExtractionError", 400, "unreadable document"),
        # A missing embedding model or unreachable local LLM is a dependency
        # being down, which is a 503 the caller can sensibly retry.
        ("embedder", "EmbeddingError", 503, "embedding unavailable"),
        ("agent.llm", "LLMTransportError", 503, "model unavailable"),
    ]

    resolved: list[tuple[type[Exception], int, str]] = []
    for module_name, class_name, status_code, label in wanted:
        try:
            module = import_module(module_name)
            error_class = getattr(module, class_name)
        except (ImportError, AttributeError):
            logger.warning(
                "Domain error %s.%s is unavailable; responses will fall back "
                "to the generic handler",
                module_name,
                class_name,
                extra={"error_module": module_name, "error_class": class_name},
            )
            continue
        resolved.append((error_class, status_code, label))
    return resolved


def _make_domain_handler(status_code: int, label: str):
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        logger.warning(
            "%s on %s %s: %s",
            label,
            request.method,
            request.url.path,
            exc,
            extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": status_code,
                "error_class": type(exc).__name__,
            },
        )
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    return handler


for _error_class, _status_code, _label in _domain_error_handlers():
    app.add_exception_handler(
        _error_class, _make_domain_handler(_status_code, _label)
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Log any unhandled error with its request context and return a clean 500.

    Without this, an unexpected failure is logged by the server with no record of
    which route or method produced it, and the caller receives an empty response
    body. The message returned is deliberately generic: internal details belong in
    the log, not in an HTTP response.

    Args:
        request: The request that raised the error.
        exc: The unhandled exception.

    Returns:
        A 500 JSON response with a generic message.
    """
    logger.exception(
        "Unhandled error on %s %s",
        request.method,
        request.url.path,
        extra={
            "path": request.url.path,
            "method": request.method,
            "error_class": type(exc).__name__,
        },
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Translate an unhandled ValueError into a 400 response.

    Args:
        request: The request that raised the error.
        exc: The raised ValueError.

    Returns:
        A 400 JSON response carrying the error detail.
    """
    logger.warning(
        "ValueError in request", extra={"path": request.url.path, "error": str(exc)}
    )
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(
    request: Request, exc: FileNotFoundError
) -> JSONResponse:
    """Translate an unhandled FileNotFoundError into a 404 response.

    Args:
        request: The request that raised the error.
        exc: The raised FileNotFoundError.

    Returns:
        A 404 JSON response carrying the error detail.
    """
    logger.warning(
        "FileNotFoundError in request",
        extra={"path": request.url.path, "error": str(exc)},
    )
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Report that the API gateway process is alive.

    Returns:
        The service name, version, and an ``ok`` status.
    """
    return {"status": "ok", "service": "api-gateway", "version": "0.1.0"}


@app.get("/health/db", tags=["health"])
async def health_db() -> dict[str, str]:
    """Report whether the database is reachable.

    Issues a trivial query so the probe fails fast when Postgres is down, and
    returns the failure as a payload rather than an exception so orchestrators can
    read the detail.

    Returns:
        A status payload describing database reachability.
    """
    from ops_common.db import get_engine
    from sqlalchemy import text

    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "database": "reachable"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "database": "unreachable", "detail": str(exc)}


app.include_router(onboard.router, prefix="/api/v1", tags=["onboarding"])
app.include_router(features.router, prefix="/api/v1", tags=["features"])
app.include_router(domains.router, prefix="/api/v1", tags=["domains"])
app.include_router(analytics.router, prefix="/api/v1", tags=["analytics"])
app.include_router(ml.router, prefix="/api/v1", tags=["ml"])
app.include_router(intelligence.router, prefix="/api/v1", tags=["intelligence"])
app.include_router(rag.router, prefix="/api/v1", tags=["rag"])
app.include_router(agent.router, prefix="/api/v1", tags=["agent"])
app.include_router(evaluation.router, prefix="/api/v1", tags=["evaluation"])
app.include_router(executive.router, prefix="/api/v1", tags=["executive"])
app.include_router(auth_routes.router, prefix="/api/v1", tags=["auth"])
