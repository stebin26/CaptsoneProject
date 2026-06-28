from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request

from ops_common.config import settings
from ops_common.db import (
    apply_schema,
    load_analytics_views,
    wait_for_postgres,
)
from ops_common.logging import configure_logging, get_logger
from api_app.routers.v1 import onboard, features, domains, analytics

logger = get_logger(__name__)

_SCHEMA_PATH = Path("/app/data-hub/postgres/schema.sql")
_ANALYTICS_PATH = Path("/app/data-hub/duckdb/analytics.sql")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("API gateway starting", extra={"env": settings.environment})

    wait_for_postgres()

    if _SCHEMA_PATH.exists():
        apply_schema(_SCHEMA_PATH)
    else:
        logger.warning("Schema file not found at startup", extra={"path": str(_SCHEMA_PATH)})

    if _ANALYTICS_PATH.exists():
        try:
            load_analytics_views(_ANALYTICS_PATH)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load DuckDB analytics views")
    else:
        logger.warning("Analytics SQL not found at startup", extra={"path": str(_ANALYTICS_PATH)})

    settings.ensure_dirs()
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


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning("ValueError in request", extra={"path": request.url.path, "error": str(exc)})
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    logger.warning("FileNotFoundError in request", extra={"path": request.url.path, "error": str(exc)})
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "api-gateway", "version": "0.1.0"}


@app.get("/health/db", tags=["health"])
async def health_db() -> dict[str, str]:
    from sqlalchemy import text
    from ops_common.db import get_engine

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