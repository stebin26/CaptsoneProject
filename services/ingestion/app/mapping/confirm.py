from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ops_common.domain.models import (
    ColumnProfile as ColumnProfileModel,
    Dataset,
    MappingConfig,
    MappingStatus,
)
from ops_common.domain.models import Domain
from ops_common.logging import get_logger

logger = get_logger(__name__)

_VALID_DOMAINS = set(Domain.values())


@dataclass
class ConfirmedColumn:
    column_name: str
    domain: str | None
    metric_name: str | None
    role: str  # "metric", "entity", "skip"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfirmedColumn":
        return cls(
            column_name=data["column_name"],
            domain=data.get("domain"),
            metric_name=data.get("metric_name"),
            role=data.get("role", "skip"),
        )

    def validate(self) -> None:
        if self.role not in ("metric", "entity", "skip"):
            raise ValueError(f"Invalid role {self.role!r} for column {self.column_name!r}")
        if self.role != "skip":
            if self.domain not in _VALID_DOMAINS:
                raise ValueError(
                    f"Invalid domain {self.domain!r} for column {self.column_name!r}"
                )
        if self.role == "metric" and not self.metric_name:
            raise ValueError(f"Metric column {self.column_name!r} needs a metric_name")


@dataclass
class ConfirmationResult:
    dataset_id: int
    business_name: str
    config_version: int
    confirmed_count: int
    skipped_count: int
    entity_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "business_name": self.business_name,
            "config_version": self.config_version,
            "confirmed_count": self.confirmed_count,
            "skipped_count": self.skipped_count,
            "entity_count": self.entity_count,
        }


def _status_for_role(role: str) -> str:
    if role == "skip":
        return MappingStatus.SKIPPED.value
    return MappingStatus.CONFIRMED.value


def _build_config(business_name: str, columns: list[ConfirmedColumn]) -> dict[str, Any]:
    return {
        "business_name": business_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "columns": [
            {
                "column_name": c.column_name,
                "domain": c.domain,
                "metric_name": c.metric_name,
                "role": c.role,
            }
            for c in columns
        ],
    }


def _next_config_version(session: Session, business_name: str) -> int:
    stmt = (
        select(MappingConfig.version)
        .where(MappingConfig.business_name == business_name)
        .order_by(MappingConfig.version.desc())
        .limit(1)
    )
    latest = session.execute(stmt).scalar_one_or_none()
    return (latest or 0) + 1


def confirm_mappings(
    session: Session,
    dataset_id: int,
    columns: list[ConfirmedColumn],
) -> ConfirmationResult:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} not found")

    for col in columns:
        col.validate()

    profile_stmt = select(ColumnProfileModel).where(
        ColumnProfileModel.dataset_id == dataset_id
    )
    profiles = {
        p.column_name: p for p in session.execute(profile_stmt).scalars().all()
    }

    confirmed = skipped = entities = 0

    for col in columns:
        profile = profiles.get(col.column_name)
        if profile is None:
            logger.warning(
                "Confirmed column has no profile row, skipping",
                extra={"dataset_id": dataset_id, "column": col.column_name},
            )
            continue

        profile.mapping_status = _status_for_role(col.role)
        if col.role != "skip":
            profile.suggested_domain = col.domain
            profile.suggested_metric = col.metric_name

        if col.role == "skip":
            skipped += 1
        elif col.role == "entity":
            entities += 1
        else:
            confirmed += 1

    version = _next_config_version(session, dataset.business_name)
    config_row = MappingConfig(
        business_name=dataset.business_name,
        config=_build_config(dataset.business_name, columns),
        version=version,
    )
    session.add(config_row)
    session.flush()

    logger.info(
        "Confirmed mappings",
        extra={
            "dataset_id": dataset_id,
            "business": dataset.business_name,
            "version": version,
            "confirmed": confirmed,
            "entities": entities,
            "skipped": skipped,
        },
    )

    return ConfirmationResult(
        dataset_id=dataset_id,
        business_name=dataset.business_name,
        config_version=version,
        confirmed_count=confirmed,
        skipped_count=skipped,
        entity_count=entities,
    )