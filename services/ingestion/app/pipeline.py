"""The onboarding pipeline -- one CSV from upload to queryable hub data.

Deliberately split into two stages around a human decision point. The first
stage profiles the file and suggests a mapping; the user reviews it; the second
stage validates, transforms, and loads. That pause is what makes automatic
onboarding trustworthy -- the system proposes, a person confirms once, and the
confirmed mapping is saved so it never has to be proposed again.

Both stages are plain callable functions rather than framework handlers, so the
same pipeline serves the API's interactive upload and Airflow's unattended
ingestion without being rewritten.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ops_common.domain.models import (
    ColumnProfile as ColumnProfileModel,
)
from ops_common.domain.models import (
    Dataset,
    MappingStatus,
)
from ops_common.logging import get_logger
from sqlalchemy.orm import Session

from app.connectors.csv_connector import CSVConnector
from app.loaders import load_to_hub
from app.mapping.confirm import ConfirmedColumn, confirm_mappings
from app.mapping.suggester import suggest_mappings
from app.profiling.profiler import profile_dataframe
from app.transforms import transform_to_hub_rows
from app.validation import validate_dataframe

logger = get_logger(__name__)


@dataclass
class OnboardStartResult:
    """Result of stage one: the profiled dataset and its suggested mapping."""
    dataset_id: int
    business_name: str
    industry: str | None
    row_count: int
    suggestions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return this result as a plain dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "business_name": self.business_name,
            "industry": self.industry,
            "row_count": self.row_count,
            "suggestions": self.suggestions,
        }


@dataclass
class OnboardCompleteResult:
    """Result of stage two: what was loaded, collected, skipped, and validated."""
    dataset_id: int
    config_version: int
    hub_rows_written: int
    features_collected: int
    features_skipped: int
    validation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return this result as a plain dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "config_version": self.config_version,
            "hub_rows_written": self.hub_rows_written,
            "features_collected": self.features_collected,
            "features_skipped": self.features_skipped,
            "validation": self.validation,
        }


# ============================================================
# STAGE 1 — upload → profile → suggest (returns for user review)
# ============================================================


def start_onboarding(
    session: Session,
    csv_path: str | Path,
    business_name: str,
    industry: str | None = None,
) -> OnboardStartResult:
    """Run stage one: read the CSV, profile it, and suggest a mapping.

    Registers the dataset and stores each column's profile so the review screen has
    everything it needs, then returns the suggestions for the user to confirm.

    Args:
        session: Active database session.
        csv_path: Path to the stored upload.
        business_name: Business the dataset belongs to.
        industry: Optional industry label.

    Returns:
        The registered dataset and its suggested column mapping.
    """
    connector = CSVConnector.from_upload(csv_path, business_name, industry)
    connector.validate_source()
    df = connector.read_dataframe()

    profile = profile_dataframe(df, connector.metadata.source_name)

    dataset = Dataset(
        business_name=business_name,
        industry=industry,
        source_filename=connector.metadata.source_name,
        row_count=profile.row_count,
    )
    session.add(dataset)
    session.flush()

    suggestion_result = suggest_mappings(profile)
    suggestion_by_col = {s.column_name: s for s in suggestion_result.suggestions}

    for col in profile.columns:
        suggestion = suggestion_by_col.get(col.column_name)
        session.add(
            ColumnProfileModel(
                dataset_id=dataset.id,
                column_name=col.column_name,
                data_type=col.data_type,
                sample_values=col.sample_values,
                distinct_count=col.distinct_count,
                null_count=col.null_count,
                suggested_domain=suggestion.suggested_domain if suggestion else None,
                suggested_metric=suggestion.suggested_metric if suggestion else None,
                mapping_status=MappingStatus.SUGGESTED.value,
            )
        )

    session.flush()

    logger.info(
        "Onboarding started",
        extra={
            "dataset_id": dataset.id,
            "business": business_name,
            "rows": profile.row_count,
            "columns": len(profile.columns),
        },
    )

    enriched = []
    for col in profile.columns:
        s = suggestion_by_col.get(col.column_name)
        enriched.append(
            {
                "column_name": col.column_name,
                "data_type": col.data_type,
                "distinct_count": col.distinct_count,
                "null_count": col.null_count,
                "sample_values": col.sample_values,
                "is_numeric": col.is_numeric,
                "is_datetime": col.is_datetime,
                "is_identifier": col.is_identifier,
                "suggested_domain": s.suggested_domain if s else None,
                "suggested_metric": s.suggested_metric if s else None,
                "role": s.role if s else "skip",
                "confidence": s.confidence if s else 0.0,
                "source": s.source if s else "none",
            }
        )

    return OnboardStartResult(
        dataset_id=dataset.id,
        business_name=business_name,
        industry=industry,
        row_count=profile.row_count,
        suggestions=enriched,
    )


# ============================================================
# STAGE 2 — confirm → validate → transform → load into hub
# ============================================================


def complete_onboarding(
    session: Session,
    dataset_id: int,
    csv_path: str | Path,
    confirmed: list[dict[str, Any]],
) -> OnboardCompleteResult:
    """Run stage two: confirm the mapping, validate, transform, and load.

    Saves the confirmed mapping as a new config version, validates the frame
    against it, unpivots it into hub rows, and loads them.

    Args:
        session: Active database session.
        dataset_id: Dataset being completed.
        csv_path: Path to the stored upload.
        confirmed: The user's confirmed column decisions.

    Returns:
        Load counts and the validation report.

    Raises:
        ValueError: If the dataset is missing or validation fails.
    """
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} not found")

    confirmed_columns = [ConfirmedColumn.from_dict(c) for c in confirmed]
    confirmation = confirm_mappings(session, dataset_id, confirmed_columns)

    connector = CSVConnector.from_upload(csv_path, dataset.business_name, dataset.industry)
    df = connector.read_dataframe()

    mapping = [
        {
            "column_name": c.column_name,
            "domain": c.domain,
            "metric_name": c.metric_name,
            "role": c.role,
        }
        for c in confirmed_columns
    ]

    report = validate_dataframe(df, mapping)
    if not report.ok:
        logger.warning(
            "Validation failed during onboarding",
            extra={"dataset_id": dataset_id, "errors": len(report.errors)},
        )
        return OnboardCompleteResult(
            dataset_id=dataset_id,
            config_version=confirmation.config_version,
            hub_rows_written=0,
            features_collected=0,
            features_skipped=0,
            validation=report.to_dict(),
        )

    transform = transform_to_hub_rows(df, mapping)
    load_result = load_to_hub(session, dataset_id, transform, row_count=len(df))

    logger.info(
        "Onboarding complete",
        extra={
            "dataset_id": dataset_id,
            "hub_rows": load_result.hub_rows_written,
            "collected": load_result.features_collected,
            "skipped": load_result.features_skipped,
        },
    )

    return OnboardCompleteResult(
        dataset_id=dataset_id,
        config_version=confirmation.config_version,
        hub_rows_written=load_result.hub_rows_written,
        features_collected=load_result.features_collected,
        features_skipped=load_result.features_skipped,
        validation=report.to_dict(),
    )
