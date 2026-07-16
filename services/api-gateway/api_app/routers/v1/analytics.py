from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from ops_common.db import get_db
from ops_common.logging import get_logger
from api_app.auth.dependencies import require_permission

logger = get_logger(__name__)

router = APIRouter()


# ============================================================
# Response models
# ============================================================

class DomainMetricOut(BaseModel):
    dataset_id: int
    business_name: str | None
    industry: str | None
    domain: str
    metric_name: str
    row_count: int | None
    distinct_entities: int | None
    null_value_count: int | None
    sum_value: float | None
    avg_value: float | None
    min_value: float | None
    max_value: float | None


class TrendPointOut(BaseModel):
    domain: str
    metric_name: str
    day: str
    row_count: int | None
    sum_value: float | None
    avg_value: float | None


class FeatureOut(BaseModel):
    domain: str
    entity_ref: str
    metric_name: str
    obs_count: int | None
    avg_value: float | None
    std_value: float | None
    min_value: float | None
    max_value: float | None
    last_value: float | None
    trend_slope: float | None


class AnalyticsOverviewOut(BaseModel):
    dataset_id: int
    business_name: str | None
    industry: str | None
    domains: list[str]
    metric_count: int
    metrics: list[DomainMetricOut]


# ============================================================
# Helpers
# ============================================================

def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# Endpoints
# ============================================================

@router.get("/analytics/{dataset_id}/metrics", response_model=list[DomainMetricOut])
def dataset_metrics(
    dataset_id: int,
    session: Session = Depends(get_db),
    _user=Depends(require_permission("analytics:read")),
) -> list[DomainMetricOut]:
    query = text(
        """
        SELECT dataset_id, business_name, industry, domain, metric_name,
               row_count, distinct_entities, null_value_count,
               sum_value, avg_value, min_value, max_value
        FROM analytics.domain_metrics
        WHERE dataset_id = :dataset_id
        ORDER BY domain, metric_name
        """
    )
    try:
        rows = session.execute(query, {"dataset_id": dataset_id}).fetchall()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read analytics.domain_metrics")
        raise HTTPException(status_code=503, detail="Analytics layer unavailable.")

    return [
        DomainMetricOut(
            dataset_id=r[0],
            business_name=r[1],
            industry=r[2],
            domain=r[3],
            metric_name=r[4],
            row_count=_as_int(r[5]),
            distinct_entities=_as_int(r[6]),
            null_value_count=_as_int(r[7]),
            sum_value=_as_float(r[8]),
            avg_value=_as_float(r[9]),
            min_value=_as_float(r[10]),
            max_value=_as_float(r[11]),
        )
        for r in rows
    ]


@router.get("/analytics/{dataset_id}/overview", response_model=AnalyticsOverviewOut)
def dataset_overview(
    dataset_id: int,
    session: Session = Depends(get_db),
    _user=Depends(require_permission("analytics:read")),
) -> AnalyticsOverviewOut:
    metrics = dataset_metrics(dataset_id, session)

    business_name = metrics[0].business_name if metrics else None
    industry = metrics[0].industry if metrics else None
    domains = sorted({m.domain for m in metrics})

    return AnalyticsOverviewOut(
        dataset_id=dataset_id,
        business_name=business_name,
        industry=industry,
        domains=domains,
        metric_count=len(metrics),
        metrics=metrics,
    )


@router.get("/analytics/{dataset_id}/trend", response_model=list[TrendPointOut])
def dataset_trend(
    dataset_id: int,
    domain: str | None = Query(default=None),
    metric_name: str | None = Query(default=None),
    session: Session = Depends(get_db),
    _user=Depends(require_permission("analytics:read")),
) -> list[TrendPointOut]:
    clauses = ["dataset_id = :dataset_id"]
    params: dict[str, Any] = {"dataset_id": dataset_id}
    if domain:
        clauses.append("domain = :domain")
        params["domain"] = domain
    if metric_name:
        clauses.append("metric_name = :metric_name")
        params["metric_name"] = metric_name

    where = " AND ".join(clauses)
    query = text(
        f"""
        SELECT domain, metric_name, day, row_count, sum_value, avg_value
        FROM analytics.daily_trend
        WHERE {where}
        ORDER BY domain, metric_name, day
        """
    )
    try:
        rows = session.execute(query, params).fetchall()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read analytics.daily_trend")
        raise HTTPException(status_code=503, detail="Analytics layer unavailable.")

    return [
        TrendPointOut(
            domain=r[0],
            metric_name=r[1],
            day=str(r[2]),
            row_count=_as_int(r[3]),
            sum_value=_as_float(r[4]),
            avg_value=_as_float(r[5]),
        )
        for r in rows
    ]


@router.get("/analytics/{dataset_id}/features", response_model=list[FeatureOut])
def dataset_features(
    dataset_id: int,
    domain: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
    session: Session = Depends(get_db),
    _user=Depends(require_permission("analytics:read")),
) -> list[FeatureOut]:
    clauses = ["dataset_id = :dataset_id"]
    params: dict[str, Any] = {"dataset_id": dataset_id, "limit": limit}
    if domain:
        clauses.append("domain = :domain")
        params["domain"] = domain

    where = " AND ".join(clauses)
    query = text(
        f"""
        SELECT domain, entity_ref, metric_name, obs_count,
               avg_value, std_value, min_value, max_value,
               last_value, trend_slope
        FROM analytics.entity_features
        WHERE {where}
        ORDER BY domain, entity_ref, metric_name
        LIMIT :limit
        """
    )
    try:
        rows = session.execute(query, params).fetchall()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read analytics.entity_features")
        raise HTTPException(status_code=503, detail="Analytics layer unavailable.")

    return [
        FeatureOut(
            domain=r[0],
            entity_ref=str(r[1]),
            metric_name=str(r[2]),
            obs_count=_as_int(r[3]),
            avg_value=_as_float(r[4]),
            std_value=_as_float(r[5]),
            min_value=_as_float(r[6]),
            max_value=_as_float(r[7]),
            last_value=_as_float(r[8]),
            trend_slope=_as_float(r[9]),
        )
        for r in rows
    ]