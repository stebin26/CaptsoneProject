"""Machine-learning API endpoints backed by the ``ml.*`` tables.

Serves forecasts, anomalies, and risk scores, plus an ML overview and a
per-domain intelligence roll-up.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from ops_common.db import get_db
from ops_common.logging import get_logger
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from api_app.auth.dependencies import require_permission

logger = get_logger(__name__)

router = APIRouter()


# ============================================================
# Response models
# ============================================================


class ForecastOut(BaseModel):
    """One forecast point for a domain metric."""
    dataset_id: int
    domain: str
    metric_name: str
    forecast_date: str
    forecast_value: float | None
    lower_bound: float | None
    upper_bound: float | None
    model_name: str | None


class AnomalyOut(BaseModel):
    """One detected anomaly (alert) for a domain metric."""
    dataset_id: int
    domain: str
    entity_ref: str | None
    metric_name: str
    anomaly_date: str | None
    observed_value: float | None
    expected_value: float | None
    deviation: float | None
    severity: str | None
    method: str | None


class RiskScoreOut(BaseModel):
    """One entity risk score with its contributing factors."""
    dataset_id: int
    domain: str
    entity_ref: str | None
    risk_score: float | None
    risk_level: str | None
    contributing_factors: dict[str, Any] | None
    model_name: str | None


class DomainIntelligenceOut(BaseModel):
    """Per-domain roll-up of current, future, alerts, and risks."""
    domain: str
    current: list[dict[str, Any]]
    future: list[ForecastOut]
    alerts: list[AnomalyOut]
    risks: list[RiskScoreOut]


class MLOverviewOut(BaseModel):
    """ML overview counts for a dataset across its domains."""
    dataset_id: int
    business_name: str | None
    industry: str | None
    domains: list[str]
    forecast_count: int
    anomaly_count: int
    high_risk_count: int


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
# Endpoints — Future (forecasts)
# ============================================================


@router.get("/ml/{dataset_id}/forecasts", response_model=list[ForecastOut])
def dataset_forecasts(
    dataset_id: int,
    domain: str | None = Query(default=None),
    metric_name: str | None = Query(default=None),
    session: Session = Depends(get_db),
    _user=Depends(require_permission("ml:read")),
) -> list[ForecastOut]:
    """Return forecasts for a dataset.

    Args:
        dataset_id: Id of the dataset to read.
        domain: Optional domain filter.
        metric_name: Optional metric filter.
        session: Active database session.
        _user: Authenticated caller, injected to enforce ``ml:read``.

    Returns:
        Forecast points, ordered by domain, metric, and date.

    Raises:
        HTTPException: 503 if the ML store cannot be read.
    """
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
        SELECT dataset_id, domain, metric_name, forecast_date,
               forecast_value, lower_bound, upper_bound, model_name
        FROM ml.forecasts
        WHERE {where}
        ORDER BY domain, metric_name, forecast_date
        """
    )
    try:
        rows = session.execute(query, params).fetchall()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read ml.forecasts")
        raise HTTPException(status_code=503, detail="ML layer unavailable.") from None

    return [
        ForecastOut(
            dataset_id=r[0],
            domain=r[1],
            metric_name=r[2],
            forecast_date=str(r[3]),
            forecast_value=_as_float(r[4]),
            lower_bound=_as_float(r[5]),
            upper_bound=_as_float(r[6]),
            model_name=r[7],
        )
        for r in rows
    ]


# ============================================================
# Endpoints — Alerts (anomalies)
# ============================================================


@router.get("/ml/{dataset_id}/anomalies", response_model=list[AnomalyOut])
def dataset_anomalies(
    dataset_id: int,
    domain: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    session: Session = Depends(get_db),
    _user=Depends(require_permission("ml:read")),
) -> list[AnomalyOut]:
    """Return anomalies for a dataset.

    Args:
        dataset_id: Id of the dataset to read.
        domain: Optional domain filter.
        severity: Optional severity filter.
        limit: Maximum number of anomalies to return.
        session: Active database session.
        _user: Authenticated caller, injected to enforce ``ml:read``.

    Returns:
        Anomalies, most severe first.

    Raises:
        HTTPException: 503 if the ML store cannot be read.
    """
    clauses = ["dataset_id = :dataset_id"]
    params: dict[str, Any] = {"dataset_id": dataset_id, "limit": limit}
    if domain:
        clauses.append("domain = :domain")
        params["domain"] = domain
    if severity:
        clauses.append("severity = :severity")
        params["severity"] = severity

    where = " AND ".join(clauses)
    query = text(
        f"""
        SELECT dataset_id, domain, entity_id, metric_name, anomaly_date,
               observed_value, expected_value, deviation, severity, method
        FROM ml.anomalies
        WHERE {where}
        ORDER BY
            CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
            domain, metric_name
        LIMIT :limit
        """
    )
    try:
        rows = session.execute(query, params).fetchall()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read ml.anomalies")
        raise HTTPException(status_code=503, detail="ML layer unavailable.") from None

    return [
        AnomalyOut(
            dataset_id=r[0],
            domain=r[1],
            entity_ref=None if r[2] is None else str(r[2]),
            metric_name=str(r[3]),
            anomaly_date=None if r[4] is None else str(r[4]),
            observed_value=_as_float(r[5]),
            expected_value=_as_float(r[6]),
            deviation=_as_float(r[7]),
            severity=r[8],
            method=r[9],
        )
        for r in rows
    ]


# ============================================================
# Endpoints — Risk scores
# ============================================================


@router.get("/ml/{dataset_id}/risk-scores", response_model=list[RiskScoreOut])
def dataset_risk_scores(
    dataset_id: int,
    domain: str | None = Query(default=None),
    risk_level: str | None = Query(default=None),
    session: Session = Depends(get_db),
    _user=Depends(require_permission("ml:read")),
) -> list[RiskScoreOut]:
    """Return risk scores for a dataset.

    Args:
        dataset_id: Id of the dataset to read.
        domain: Optional domain filter.
        risk_level: Optional risk-level filter.
        session: Active database session.
        _user: Authenticated caller, injected to enforce ``ml:read``.

    Returns:
        Risk scores, highest first.

    Raises:
        HTTPException: 503 if the ML store cannot be read.
    """
    clauses = ["dataset_id = :dataset_id"]
    params: dict[str, Any] = {"dataset_id": dataset_id}
    if domain:
        clauses.append("domain = :domain")
        params["domain"] = domain
    if risk_level:
        clauses.append("risk_level = :risk_level")
        params["risk_level"] = risk_level

    where = " AND ".join(clauses)
    query = text(
        f"""
        SELECT dataset_id, domain, entity_id, risk_score, risk_level,
               contributing_factors, model_name
        FROM ml.risk_scores
        WHERE {where}
        ORDER BY risk_score DESC NULLS LAST
        """
    )
    try:
        rows = session.execute(query, params).fetchall()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read ml.risk_scores")
        raise HTTPException(status_code=503, detail="ML layer unavailable.") from None

    return [
        RiskScoreOut(
            dataset_id=r[0],
            domain=r[1],
            entity_ref=None if r[2] is None else str(r[2]),
            risk_score=_as_float(r[3]),
            risk_level=r[4],
            contributing_factors=r[5] if isinstance(r[5], dict) else None,
            model_name=r[6],
        )
        for r in rows
    ]


# ============================================================
# Endpoints — Overview + per-domain intelligence
# ============================================================


@router.get("/ml/{dataset_id}/overview", response_model=MLOverviewOut)
def ml_overview(
    dataset_id: int,
    session: Session = Depends(get_db),
    _user=Depends(require_permission("ml:read")),
) -> MLOverviewOut:
    """Return ML overview counts for a dataset across its domains.

    Args:
        dataset_id: Id of the dataset to summarize.
        session: Active database session.
        _user: Authenticated caller, injected to enforce ``ml:read``.

    Returns:
        Forecast, anomaly, and high-risk counts plus the active domains.

    Raises:
        HTTPException: 503 if the ML store cannot be read.
    """
    query = text(
        """
        SELECT
            (SELECT COUNT(*) FROM ml.forecasts  WHERE dataset_id = :d),
            (SELECT COUNT(*) FROM ml.anomalies  WHERE dataset_id = :d),
            (SELECT COUNT(*) FROM ml.risk_scores WHERE dataset_id = :d AND risk_level = 'high'),
            (SELECT business_name FROM ml.forecasts WHERE dataset_id = :d LIMIT 1),
            (SELECT industry FROM ml.forecasts WHERE dataset_id = :d LIMIT 1)
        """
    )
    domains_query = text(
        """
        SELECT DISTINCT domain FROM (
            SELECT domain FROM ml.forecasts   WHERE dataset_id = :d
            UNION SELECT domain FROM ml.anomalies   WHERE dataset_id = :d
            UNION SELECT domain FROM ml.risk_scores WHERE dataset_id = :d
        ) t
        ORDER BY domain
        """
    )
    try:
        row = session.execute(query, {"d": dataset_id}).fetchone()
        domain_rows = session.execute(domains_query, {"d": dataset_id}).fetchall()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read ML overview")
        raise HTTPException(status_code=503, detail="ML layer unavailable.") from None

    return MLOverviewOut(
        dataset_id=dataset_id,
        business_name=row[3] if row else None,
        industry=row[4] if row else None,
        domains=[str(d[0]) for d in domain_rows],
        forecast_count=_as_int(row[0]) or 0 if row else 0,
        anomaly_count=_as_int(row[1]) or 0 if row else 0,
        high_risk_count=_as_int(row[2]) or 0 if row else 0,
    )


@router.get("/ml/{dataset_id}/domain/{domain}", response_model=DomainIntelligenceOut)
def domain_intelligence(
    dataset_id: int,
    domain: str,
    session: Session = Depends(get_db),
    _user=Depends(require_permission("ml:read")),
) -> DomainIntelligenceOut:
    """Return a per-domain roll-up of current metrics, forecasts, alerts, and risks.

    Args:
        dataset_id: Id of the dataset to read.
        domain: Domain to roll up.
        session: Active database session.
        _user: Authenticated caller, injected to enforce ``ml:read``.

    Returns:
        The domain's current metrics, forecasts, alerts, and risks.

    Raises:
        HTTPException: 503 if the analytics store cannot be read.
    """
    current_query = text(
        """
        SELECT domain, metric_name, row_count, distinct_entities,
               sum_value, avg_value, min_value, max_value
        FROM analytics.domain_metrics
        WHERE dataset_id = :d AND domain = :domain
        ORDER BY metric_name
        """
    )
    try:
        current_rows = session.execute(
            current_query, {"d": dataset_id, "domain": domain}
        ).fetchall()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read analytics.domain_metrics")
        raise HTTPException(status_code=503, detail="Analytics layer unavailable.") from None

    current = [
        {
            "domain": r[0],
            "metric_name": r[1],
            "row_count": _as_int(r[2]),
            "distinct_entities": _as_int(r[3]),
            "sum_value": _as_float(r[4]),
            "avg_value": _as_float(r[5]),
            "min_value": _as_float(r[6]),
            "max_value": _as_float(r[7]),
        }
        for r in current_rows
    ]

    future = dataset_forecasts(dataset_id, domain=domain, session=session)
    alerts = dataset_anomalies(dataset_id, domain=domain, session=session)
    risks = dataset_risk_scores(dataset_id, domain=domain, session=session)

    return DomainIntelligenceOut(
        domain=domain,
        current=current,
        future=future,
        alerts=alerts,
        risks=risks,
    )
