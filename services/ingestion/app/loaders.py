"""Loading -- writing transformed rows and feature records into the hub.

Routes each row to one of the eight domain tables by its domain, inserting in
batches so a large upload does not build one enormous statement. Loading is
idempotent per dataset: existing hub rows and feature records for the dataset
are cleared first, so re-onboarding the same dataset replaces its data rather
than duplicating it. Skipped columns are recorded too, which is what lets the
review screen show what was *not* collected and offer to add it later.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ops_common.domain.models import (
    ColumnProfile as ColumnProfileModel,
)
from ops_common.domain.models import (
    Dataset,
    FeatureRecord,
    FeatureStatus,
    MappingStatus,
    model_for_domain,
)
from ops_common.domain.registry import features_for_domain
from ops_common.logging import get_logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.transforms import HubRow, TransformResult

logger = get_logger(__name__)

_INSERT_BATCH_SIZE = 1000


@dataclass
class LoadResult:
    """Counts describing what a load wrote: hub rows and features."""
    dataset_id: int
    hub_rows_written: int
    features_collected: int
    features_skipped: int

    def to_dict(self) -> dict[str, Any]:
        """Return this result as a plain dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "hub_rows_written": self.hub_rows_written,
            "features_collected": self.features_collected,
            "features_skipped": self.features_skipped,
        }


def _batched(rows: list[HubRow], size: int) -> Iterable[list[HubRow]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def _write_hub_rows(session: Session, dataset_id: int, rows: list[HubRow]) -> int:
    written = 0
    grouped: dict[str, list[HubRow]] = {}
    for row in rows:
        grouped.setdefault(row.domain, []).append(row)

    for domain, domain_rows in grouped.items():
        model = model_for_domain(domain)
        for batch in _batched(domain_rows, _INSERT_BATCH_SIZE):
            mappings = [
                {
                    "dataset_id": dataset_id,
                    "entity_ref": r.entity_ref,
                    "metric_name": r.metric_name,
                    "metric_value": r.metric_value,
                    "attributes": r.attributes,
                    "recorded_at": r.recorded_at,
                }
                for r in batch
            ]
            session.bulk_insert_mappings(model, mappings)
            written += len(mappings)

        logger.info(
            "Wrote hub rows",
            extra={"dataset_id": dataset_id, "domain": domain, "rows": len(domain_rows)},
        )
    return written


def _generate_feature_records(
    session: Session,
    dataset_id: int,
    transform: TransformResult,
) -> tuple[int, int]:
    collected = 0

    for domain, metric_names in transform.metrics_by_domain.items():
        domain_features = features_for_domain(domain)
        for metric_name in sorted(metric_names):
            for feature_def in domain_features:
                session.add(
                    FeatureRecord(
                        dataset_id=dataset_id,
                        domain=domain,
                        feature_name=f"{metric_name}.{feature_def.name}",
                        source_column=metric_name,
                        status=FeatureStatus.COLLECTED.value,
                    )
                )
                collected += 1

    skipped = _record_skipped_features(session, dataset_id)
    return collected, skipped


def _record_skipped_features(session: Session, dataset_id: int) -> int:
    stmt = select(ColumnProfileModel).where(
        ColumnProfileModel.dataset_id == dataset_id,
        ColumnProfileModel.mapping_status == MappingStatus.SKIPPED.value,
    )
    skipped_columns = session.execute(stmt).scalars().all()

    for col in skipped_columns:
        session.add(
            FeatureRecord(
                dataset_id=dataset_id,
                domain=col.suggested_domain or "unmapped",
                feature_name=col.column_name,
                source_column=col.column_name,
                status=FeatureStatus.SKIPPED.value,
            )
        )
    return len(skipped_columns)


def load_to_hub(
    session: Session,
    dataset_id: int,
    transform: TransformResult,
    row_count: int | None = None,
) -> LoadResult:
    """Write a dataset's transformed rows and feature records to the hub.

    Clears any previous data for the dataset first, so a re-run replaces rather
    than duplicates.

    Args:
        session: Active database session.
        dataset_id: Dataset being loaded.
        transform: The transformed rows and their per-domain metrics.
        row_count: Source row count to record on the dataset.

    Returns:
        Counts of what was written.

    Raises:
        ValueError: If the dataset does not exist.
    """
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} not found")

    _clear_existing_hub_data(session, dataset_id)
    _clear_existing_features(session, dataset_id)

    written = _write_hub_rows(session, dataset_id, transform.rows)
    collected, skipped = _generate_feature_records(session, dataset_id, transform)

    if row_count is not None:
        dataset.row_count = row_count

    session.flush()

    logger.info(
        "Load complete",
        extra={
            "dataset_id": dataset_id,
            "hub_rows": written,
            "features_collected": collected,
            "features_skipped": skipped,
        },
    )

    return LoadResult(
        dataset_id=dataset_id,
        hub_rows_written=written,
        features_collected=collected,
        features_skipped=skipped,
    )


def _clear_existing_hub_data(session: Session, dataset_id: int) -> None:
    from ops_common.domain.models import DOMAIN_MODELS

    for model in DOMAIN_MODELS.values():
        session.query(model).filter(model.dataset_id == dataset_id).delete(
            synchronize_session=False
        )


def _clear_existing_features(session: Session, dataset_id: int) -> None:
    session.query(FeatureRecord).filter(FeatureRecord.dataset_id == dataset_id).delete(
        synchronize_session=False
    )
