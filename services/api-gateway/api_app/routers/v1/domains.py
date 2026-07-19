"""Domain registry and hub-data API endpoints.

Serves the static universal-domain catalog and reads entity-level hub data
through the DuckDB analytics views, plus a dataset listing for the browse page.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from ops_common.db import duckdb_scope, get_db
from ops_common.domain.models import Dataset, Domain
from ops_common.domain.registry import DOMAIN_REGISTRY, get_spec
from ops_common.logging import get_logger
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from api_app.auth.dependencies import require_permission

logger = get_logger(__name__)

router = APIRouter()

_VALID_DOMAINS = set(Domain.values())


# ============================================================
# Response models
# ============================================================


class DomainInfoOut(BaseModel):
    """Static catalog entry for one universal domain."""
    domain: str
    description: str
    features: list[str]


class MetricSummaryOut(BaseModel):
    """Aggregate summary for one metric within a domain."""
    domain: str
    metric_name: str
    observations: int
    metric_sum: float | None
    metric_avg: float | None
    metric_min: float | None
    metric_max: float | None


class DomainSummaryOut(BaseModel):
    """Per-domain metric summary for a dataset."""
    dataset_id: int
    business_name: str
    metrics: list[MetricSummaryOut]


class MetricPointOut(BaseModel):
    """One entity-level metric reading from the hub."""
    entity_ref: str
    metric_name: str
    metric_value: float | None
    recorded_at: str | None


class DomainDataOut(BaseModel):
    """Entity-level hub readings for one domain of a dataset."""
    dataset_id: int
    domain: str
    points: list[MetricPointOut]


# ============================================================
# Static registry endpoints
# ============================================================


@router.get("/domains", response_model=list[DomainInfoOut])
def list_domains(
    _user=Depends(require_permission("dataset:read")),
) -> list[DomainInfoOut]:
    """Return the static catalog of universal domains and their features.

    Args:
        _user: Authenticated caller, injected to enforce ``dataset:read``.

    Returns:
        One catalog entry per universal domain.
    """
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
def domain_info(
    domain: str,
    _user=Depends(require_permission("dataset:read")),
) -> DomainInfoOut:
    """Return the catalog entry for one domain.

    Args:
        domain: Name of the domain to describe.
        _user: Authenticated caller, injected to enforce ``dataset:read``.

    Returns:
        The domain's description and feature list.

    Raises:
        HTTPException: 404 if the domain is unknown.
    """
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
    _user=Depends(require_permission("dataset:read")),
) -> DomainSummaryOut:
    """Return a per-domain metric summary for a dataset from the hub.

    Args:
        dataset_id: Id of the dataset to summarize.
        session: Active database session.
        _user: Authenticated caller, injected to enforce ``dataset:read``.

    Returns:
        The dataset's per-domain metric summary.

    Raises:
        HTTPException: 404 if the dataset is missing, 503 if analytics is unavailable.
    """
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
        raise HTTPException(status_code=503, detail="Analytics layer unavailable.") from None

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
    _user=Depends(require_permission("dataset:read")),
) -> DomainDataOut:
    """Return entity-level hub readings for one domain of a dataset.

    Args:
        dataset_id: Id of the dataset to read.
        domain: Domain to read readings from.
        limit: Maximum number of readings to return.
        session: Active database session.
        _user: Authenticated caller, injected to enforce ``dataset:read``.

    Returns:
        Entity-level readings for the domain.

    Raises:
        HTTPException: 404 if the dataset or domain is unknown, 503 if analytics is
            unavailable.
    """
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
        raise HTTPException(status_code=503, detail="Analytics layer unavailable.") from None

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


# ============================================================
# Dataset listing (for the Datasets browse page)
# ============================================================


class DatasetListItem(BaseModel):
    """One row in the dataset browse listing."""
    dataset_id: int
    business_name: str
    industry: str | None
    source_filename: str
    row_count: int | None
    uploaded_at: str
    features_collected: int
    features_skipped: int


@router.get("/datasets", response_model=list[DatasetListItem])
def list_datasets(
    session: Session = Depends(get_db),
    _user=Depends(require_permission("dataset:read")),
) -> list[DatasetListItem]:
    """Return all datasets with their collected/skipped feature counts.

    Args:
        session: Active database session.
        _user: Authenticated caller, injected to enforce ``dataset:read``.

    Returns:
        One listing row per dataset, newest first.
    """
    from ops_common.domain.models import Dataset, FeatureRecord, FeatureStatus
    from sqlalchemy import func

    datasets = (
        session.execute(select(Dataset).order_by(Dataset.id.desc())).scalars().all()
    )

    counts: dict[int, dict[str, int]] = {}
    rows = session.execute(
        select(
            FeatureRecord.dataset_id,
            FeatureRecord.status,
            func.count().label("n"),
        ).group_by(FeatureRecord.dataset_id, FeatureRecord.status)
    ).all()
    for dataset_id, status, n in rows:
        bucket = counts.setdefault(dataset_id, {"collected": 0, "skipped": 0})
        if status in (FeatureStatus.COLLECTED.value, FeatureStatus.ADDED_LATER.value):
            bucket["collected"] += int(n)
        elif status == FeatureStatus.SKIPPED.value:
            bucket["skipped"] += int(n)

    result: list[DatasetListItem] = []
    for d in datasets:
        c = counts.get(d.id, {"collected": 0, "skipped": 0})
        result.append(
            DatasetListItem(
                dataset_id=d.id,
                business_name=d.business_name,
                industry=d.industry,
                source_filename=d.source_filename,
                row_count=d.row_count,
                uploaded_at=str(d.uploaded_at),
                features_collected=c["collected"],
                features_skipped=c["skipped"],
            )
        )
    return result
