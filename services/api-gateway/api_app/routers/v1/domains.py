from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ops_common.db import get_db, duckdb_scope
from ops_common.domain.models import Dataset, Domain
from ops_common.domain.registry import DOMAIN_REGISTRY, get_spec
from ops_common.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

_VALID_DOMAINS = set(Domain.values())


# ============================================================
# Response models
# ============================================================

class DomainInfoOut(BaseModel):
    domain: str
    description: str
    features: list[str]


class MetricSummaryOut(BaseModel):
    domain: str
    metric_name: str
    observations: int
    metric_sum: float | None
    metric_avg: float | None
    metric_min: float | None
    metric_max: float | None


class DomainSummaryOut(BaseModel):
    dataset_id: int
    business_name: str
    metrics: list[MetricSummaryOut]


class MetricPointOut(BaseModel):
    entity_ref: str
    metric_name: str
    metric_value: float | None
    recorded_at: str | None


class DomainDataOut(BaseModel):
    dataset_id: int
    domain: str
    points: list[MetricPointOut]


# ============================================================
# Static registry endpoints
# ============================================================

@router.get("/domains", response_model=list[DomainInfoOut])
def list_domains() -> list[DomainInfoOut]:
    out: list[DomainInfoOut] = []
    for spec in DOMAIN_REGISTRY.values():
        out.append(
            DomainInfoOut(
                domain=spec.domain.value,
                description=spec.description,
                features=spec.feature_names(),
            )
        )
    return out


@router.get("/domains/{domain}", response_model=DomainInfoOut)
def domain_info(domain: str) -> DomainInfoOut:
    if domain not in _VALID_DOMAINS:
        raise HTTPException(status_code=404, detail=f"Unknown domain {domain!r}")
    spec = get_spec(domain)
    return DomainInfoOut(
        domain=spec.domain.value,
        description=spec.description,
        features=spec.feature_names(),
    )


# ============================================================
# Hub data endpoints (read through DuckDB analytics views)
# ============================================================

def _require_dataset(session: Session, dataset_id: int) -> Dataset:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return dataset


@router.get("/datasets/{dataset_id}/summary", response_model=DomainSummaryOut)
def dataset_summary(
    dataset_id: int,
    session: Session = Depends(get_db),
) -> DomainSummaryOut:
    dataset = _require_dataset(session, dataset_id)

    query = """
        SELECT domain, metric_name, observations,
               metric_sum, metric_avg, metric_min, metric_max
        FROM v_domain_metric_summary
        WHERE dataset_id = ?
        ORDER BY domain, metric_name
    """

    metrics: list[MetricSummaryOut] = []
    try:
        with duckdb_scope(read_only=True) as conn:
            rows = conn.execute(query, [dataset_id]).fetchall()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read domain summary from DuckDB")
        raise HTTPException(status_code=503, detail="Analytics layer unavailable.")

    for r in rows:
        metrics.append(
            MetricSummaryOut(
                domain=r[0],
                metric_name=r[1],
                observations=int(r[2]) if r[2] is not None else 0,
                metric_sum=_as_float(r[3]),
                metric_avg=_as_float(r[4]),
                metric_min=_as_float(r[5]),
                metric_max=_as_float(r[6]),
            )
        )

    return DomainSummaryOut(
        dataset_id=dataset_id,
        business_name=dataset.business_name,
        metrics=metrics,
    )


@router.get("/datasets/{dataset_id}/domains/{domain}", response_model=DomainDataOut)
def domain_data(
    dataset_id: int,
    domain: str,
    limit: int = Query(default=200, ge=1, le=2000),
    session: Session = Depends(get_db),
) -> DomainDataOut:
    _require_dataset(session, dataset_id)

    if domain not in _VALID_DOMAINS:
        raise HTTPException(status_code=404, detail=f"Unknown domain {domain!r}")

    query = """
        SELECT entity_ref, metric_name, metric_value, recorded_at
        FROM v_hub_metrics
        WHERE dataset_id = ? AND domain = ?
        ORDER BY recorded_at NULLS LAST, entity_ref
        LIMIT ?
    """

    try:
        with duckdb_scope(read_only=True) as conn:
            rows = conn.execute(query, [dataset_id, domain, limit]).fetchall()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read domain data from DuckDB")
        raise HTTPException(status_code=503, detail="Analytics layer unavailable.")

    points = [
        MetricPointOut(
            entity_ref=str(r[0]),
            metric_name=str(r[1]),
            metric_value=_as_float(r[2]),
            recorded_at=str(r[3]) if r[3] is not None else None,
        )
        for r in rows
    ]

    return DomainDataOut(dataset_id=dataset_id, domain=domain, points=points)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None