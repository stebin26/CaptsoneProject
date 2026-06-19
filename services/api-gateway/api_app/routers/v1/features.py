from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ops_common.db import get_db
from ops_common.domain.models import (
    ColumnProfile,
    Dataset,
    FeatureRecord,
    FeatureStatus,
    MappingStatus,
    Domain,
    model_for_domain,
)
from ops_common.domain.registry import features_for_domain
from ops_common.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

_VALID_DOMAINS = set(Domain.values())


# ============================================================
# Response models
# ============================================================

class FeatureOut(BaseModel):
    domain: str
    feature_name: str
    source_column: str
    status: str


class MissedColumnOut(BaseModel):
    column_name: str
    data_type: str
    distinct_count: int | None
    null_count: int | None
    sample_values: Any | None
    suggested_domain: str | None


class DomainCoverageOut(BaseModel):
    domain: str
    features_collected: int
    features_skipped: int


class FeatureReviewOut(BaseModel):
    dataset_id: int
    business_name: str
    industry: str | None
    row_count: int | None
    collected: list[FeatureOut]
    missed: list[MissedColumnOut]
    coverage: list[DomainCoverageOut]


class AddFeatureRequest(BaseModel):
    dataset_id: int
    column_name: str
    domain: str
    metric_name: str = Field(..., min_length=1)


class AddFeatureResponse(BaseModel):
    dataset_id: int
    column_name: str
    domain: str
    features_added: int


# ============================================================
# Helpers
# ============================================================

def _require_dataset(session: Session, dataset_id: int) -> Dataset:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return dataset


# ============================================================
# Endpoints
# ============================================================

@router.get("/features/{dataset_id}/review", response_model=FeatureReviewOut)
def feature_review(dataset_id: int, session: Session = Depends(get_db)) -> FeatureReviewOut:
    dataset = _require_dataset(session, dataset_id)

    collected_stmt = (
        select(FeatureRecord)
        .where(
            FeatureRecord.dataset_id == dataset_id,
            FeatureRecord.status.in_(
                [FeatureStatus.COLLECTED.value, FeatureStatus.ADDED_LATER.value]
            ),
        )
        .order_by(FeatureRecord.domain, FeatureRecord.feature_name)
    )
    collected = [
        FeatureOut(
            domain=f.domain,
            feature_name=f.feature_name,
            source_column=f.source_column,
            status=f.status,
        )
        for f in session.execute(collected_stmt).scalars().all()
    ]

    missed_stmt = (
        select(ColumnProfile)
        .where(
            ColumnProfile.dataset_id == dataset_id,
            ColumnProfile.mapping_status == MappingStatus.SKIPPED.value,
        )
        .order_by(ColumnProfile.column_name)
    )
    missed = [
        MissedColumnOut(
            column_name=c.column_name,
            data_type=c.data_type,
            distinct_count=c.distinct_count,
            null_count=c.null_count,
            sample_values=c.sample_values,
            suggested_domain=c.suggested_domain,
        )
        for c in session.execute(missed_stmt).scalars().all()
    ]

    coverage = _build_coverage(session, dataset_id)

    return FeatureReviewOut(
        dataset_id=dataset_id,
        business_name=dataset.business_name,
        industry=dataset.industry,
        row_count=dataset.row_count,
        collected=collected,
        missed=missed,
        coverage=coverage,
    )


def _build_coverage(session: Session, dataset_id: int) -> list[DomainCoverageOut]:
    stmt = select(FeatureRecord).where(FeatureRecord.dataset_id == dataset_id)
    records = session.execute(stmt).scalars().all()

    agg: dict[str, dict[str, int]] = {}
    for r in records:
        bucket = agg.setdefault(r.domain, {"collected": 0, "skipped": 0})
        if r.status in (FeatureStatus.COLLECTED.value, FeatureStatus.ADDED_LATER.value):
            bucket["collected"] += 1
        elif r.status == FeatureStatus.SKIPPED.value:
            bucket["skipped"] += 1

    return [
        DomainCoverageOut(
            domain=domain,
            features_collected=counts["collected"],
            features_skipped=counts["skipped"],
        )
        for domain, counts in sorted(agg.items())
    ]


@router.post("/features/add", response_model=AddFeatureResponse)
def add_missed_feature(
    payload: AddFeatureRequest,
    session: Session = Depends(get_db),
) -> AddFeatureResponse:
    dataset = _require_dataset(session, payload.dataset_id)

    if payload.domain not in _VALID_DOMAINS:
        raise HTTPException(status_code=400, detail=f"Invalid domain {payload.domain!r}")

    profile_stmt = select(ColumnProfile).where(
        ColumnProfile.dataset_id == payload.dataset_id,
        ColumnProfile.column_name == payload.column_name,
    )
    profile = session.execute(profile_stmt).scalar_one_or_none()
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"Column {payload.column_name!r} not found for this dataset.",
        )
    if profile.mapping_status != MappingStatus.SKIPPED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Column {payload.column_name!r} is not in skipped state.",
        )

    try:
        added = _ingest_single_column(session, dataset, payload)
        profile.mapping_status = MappingStatus.ADDED_LATER.value
        profile.suggested_domain = payload.domain
        profile.suggested_metric = payload.metric_name
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("add_missed_feature failed")
        raise

    return AddFeatureResponse(
        dataset_id=payload.dataset_id,
        column_name=payload.column_name,
        domain=payload.domain,
        features_added=added,
    )


def _ingest_single_column(
    session: Session,
    dataset: Dataset,
    payload: AddFeatureRequest,
) -> int:
    import pandas as pd
    from pathlib import Path
    from ops_common.config import settings
    from app.transforms import transform_to_hub_rows
    from app.loaders import _write_hub_rows

    matches = sorted(settings.upload_dir.glob("*.csv"))
    if not matches:
        raise HTTPException(status_code=404, detail="Source file unavailable for re-ingest.")
    df = pd.read_csv(matches[-1], low_memory=False)
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    if payload.column_name not in df.columns:
        raise HTTPException(
            status_code=404,
            detail=f"Column {payload.column_name!r} no longer present in source.",
        )

    entity_col = _find_entity_column(session, dataset.id)

    mapping = [{"column_name": payload.column_name, "domain": payload.domain,
                "metric_name": payload.metric_name, "role": "metric"}]
    if entity_col:
        mapping.append({"column_name": entity_col, "domain": payload.domain,
                        "metric_name": None, "role": "entity"})

    transform = transform_to_hub_rows(df, mapping)
    _write_hub_rows(session, dataset.id, transform.rows)

    domain_features = features_for_domain(payload.domain)
    for feature_def in domain_features:
        session.add(
            FeatureRecord(
                dataset_id=dataset.id,
                domain=payload.domain,
                feature_name=f"{payload.metric_name}.{feature_def.name}",
                source_column=payload.metric_name,
                status=FeatureStatus.ADDED_LATER.value,
            )
        )
    return len(domain_features)


def _find_entity_column(session: Session, dataset_id: int) -> str | None:
    stmt = select(ColumnProfile).where(ColumnProfile.dataset_id == dataset_id)
    for col in session.execute(stmt).scalars().all():
        name = col.column_name.lower()
        if name.endswith("_id") or name == "id" or name.endswith("_ref"):
            return col.column_name
    return None