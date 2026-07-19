"""Transformation -- reshaping source rows into the hub's universal shape.

This is where an industry-specific file becomes industry-agnostic data. A wide
source frame, where each mapped column is its own metric, is unpivoted into hub
rows of ``entity_ref``, ``metric_name``, and ``metric_value``. Because every one
of the eight hub tables shares that shape, everything downstream can treat data
from any industry identically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
from ops_common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HubRow:
    """One measurement in the hub's universal shape."""
    domain: str
    entity_ref: str
    metric_name: str
    metric_value: float | None
    attributes: dict[str, Any] | None
    recorded_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        """Return this row as a plain dictionary."""
        return {
            "domain": self.domain,
            "entity_ref": self.entity_ref,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "attributes": self.attributes,
            "recorded_at": self.recorded_at,
        }


@dataclass
class TransformResult:
    """The transformed rows plus which metrics landed in which domain."""
    rows: list[HubRow] = field(default_factory=list)
    metrics_by_domain: dict[str, set[str]] = field(default_factory=dict)

    def record_metric(self, domain: str, metric_name: str) -> None:
        """Note that a metric was produced for a domain.

        Args:
            domain: The domain the metric belongs to.
            metric_name: The metric that was produced.
        """
        self.metrics_by_domain.setdefault(domain, set()).add(metric_name)

    def summary(self) -> dict[str, Any]:
        """Return a summary of what the transform produced.

        Returns:
            The row count and the metrics found per domain.
        """
        return {
            "row_count": len(self.rows),
            "domains": {d: sorted(m) for d, m in self.metrics_by_domain.items()},
        }


@dataclass
class _MappingSpec:
    column_name: str
    domain: str | None
    metric_name: str | None
    role: str


def _normalize_specs(mapping: list[dict[str, Any]]) -> list[_MappingSpec]:
    specs: list[_MappingSpec] = []
    for index, m in enumerate(mapping):
        try:
            column_name = m["column_name"]
        except (KeyError, TypeError) as exc:
            # Naming the position makes a malformed mapping payload findable;
            # a bare KeyError would not say which entry was at fault.
            logger.error(
                "Mapping entry %d has no column_name: %r",
                index,
                m,
                extra={"entry_index": index},
            )
            raise ValueError(
                f"Mapping entry {index} is missing 'column_name': {m!r}"
            ) from exc

        specs.append(
            _MappingSpec(
                column_name=column_name,
                domain=m.get("domain"),
                metric_name=m.get("metric_name"),
                role=m.get("role", "skip"),
            )
        )
    return specs


def _pick_entity_column(specs: list[_MappingSpec]) -> str | None:
    entities = [s for s in specs if s.role == "entity"]
    if not entities:
        return None
    return entities[0].column_name


def _pick_time_column(df: pd.DataFrame, specs: list[_MappingSpec]) -> str | None:
    for s in specs:
        name = s.column_name.lower()
        if any(tok in name for tok in ("date", "time", "timestamp", "_at", "_on")):
            if s.column_name in df.columns:
                return s.column_name
    return None


def _coerce_timestamp(value: Any) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except (ValueError, TypeError):
        return None


def _coerce_numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        num = pd.to_numeric(value, errors="coerce")
        if pd.isna(num):
            return None
        return float(num)
    except (ValueError, TypeError):
        return None


def transform_to_hub_rows(
    df: pd.DataFrame,
    mapping: list[dict[str, Any]],
) -> TransformResult:
    """Unpivot a source frame into hub rows using its confirmed mapping.

    Each mapped metric column becomes one hub row per source row, tagged with the
    entity it belongs to and the timestamp it was recorded at.

    Args:
        df: The validated source frame.
        mapping: The confirmed column decisions.

    Returns:
        The hub rows and the metrics recorded per domain.
    """
    specs = _normalize_specs(mapping)
    metric_specs = [s for s in specs if s.role == "metric" and s.domain]
    entity_specs = [s for s in specs if s.role == "entity"]

    entity_col = _pick_entity_column(specs)
    time_col = _pick_time_column(df, specs)

    result = TransformResult()

    records = df.to_dict(orient="records")
    for idx, row in enumerate(records):
        entity_ref = _resolve_entity_ref(row, entity_col, idx)
        recorded_at = _coerce_timestamp(row.get(time_col)) if time_col else None
        attributes = _build_attributes(row, entity_specs, entity_col)

        for spec in metric_specs:
            raw_value = row.get(spec.column_name)
            metric_value = _coerce_numeric(raw_value)

            hub_row = HubRow(
                domain=spec.domain,  # type: ignore[arg-type]
                entity_ref=entity_ref,
                metric_name=spec.metric_name or spec.column_name,
                metric_value=metric_value,
                attributes=attributes or None,
                recorded_at=recorded_at,
            )
            result.rows.append(hub_row)
            result.record_metric(spec.domain, hub_row.metric_name)  # type: ignore[arg-type]

    logger.info("Transformed to hub rows", extra=result.summary())
    return result


def _resolve_entity_ref(row: dict[str, Any], entity_col: str | None, idx: int) -> str:
    if entity_col:
        value = row.get(entity_col)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            return str(value)
    return f"row_{idx}"


def _build_attributes(
    row: dict[str, Any],
    entity_specs: list[_MappingSpec],
    primary_entity_col: str | None,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for spec in entity_specs:
        if spec.column_name == primary_entity_col:
            continue
        value = row.get(spec.column_name)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            attributes[spec.column_name] = str(value)
    return attributes
