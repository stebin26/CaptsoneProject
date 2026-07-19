"""Data validation -- checking a mapping against the data before loading it.

Runs between confirmation and loading, so a mapping that does not match the
actual file is caught before anything reaches the hub. Issues are separated into
errors, which stop the load, and warnings, which are reported but allow it: a
missing mapped column is fatal, whereas a mostly-null entity column is worth
flagging without blocking an otherwise usable dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd
from ops_common.domain.models import Domain
from ops_common.logging import get_logger

logger = get_logger(__name__)

_VALID_DOMAINS = set(Domain.values())


class Severity(str, Enum): # noqa: UP042
    """Whether an issue blocks the load or is merely reported."""
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    """One validation finding, with its severity and the column involved."""
    severity: Severity
    code: str
    message: str
    column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return this issue as a plain dictionary."""
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "column": self.column,
        }


@dataclass
class ValidationReport:
    """Every issue found while validating a dataset against its mapping."""
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Return the issues that block the load."""
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Return the issues that are reported but do not block the load."""
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """Return whether the dataset passed validation."""
        return len(self.errors) == 0

    def add(self, severity: Severity, code: str, message: str, column: str | None = None) -> None:
        """Record one validation issue.

        Args:
            severity: Whether the issue blocks the load.
            code: Short machine-readable identifier for the issue.
            message: Human-readable description.
            column: The column involved, when the issue is column-specific.
        """
        self.issues.append(ValidationIssue(severity, code, message, column))

    def to_dict(self) -> dict[str, Any]:
        """Return this issue as a plain dictionary."""
        return {
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class MappingSpec:
    """One column's mapping decision, in the shape validation needs."""
    column_name: str
    domain: str | None
    metric_name: str | None
    role: str  # "metric", "entity", "skip"


def _normalize_specs(mapping: list[dict[str, Any]]) -> list[MappingSpec]:
    specs: list[MappingSpec] = []
    for m in mapping:
        specs.append(
            MappingSpec(
                column_name=m["column_name"],
                domain=m.get("domain"),
                metric_name=m.get("metric_name"),
                role=m.get("role", "skip"),
            )
        )
    return specs


def validate_dataframe(
    df: pd.DataFrame,
    mapping: list[dict[str, Any]],
) -> ValidationReport:
    """Validate a frame against its confirmed mapping.

    Checks that mapped columns exist, that metric columns carry usable values, and
    that entity columns are populated enough to be meaningful.

    Args:
        df: The frame about to be loaded.
        mapping: The confirmed column decisions.

    Returns:
        A report of every issue found.
    """
    report = ValidationReport()
    specs = _normalize_specs(mapping)

    if df.empty:
        report.add(Severity.ERROR, "empty_dataframe", "Dataframe has no rows.")
        return report

    df_columns = set(df.columns)

    active = [s for s in specs if s.role != "skip"]
    if not active:
        report.add(
            Severity.ERROR,
            "no_active_columns",
            "No columns are mapped to a domain; nothing to load.",
        )

    entities = [s for s in specs if s.role == "entity"]
    if not entities:
        report.add(
            Severity.WARNING,
            "no_entity_column",
            "No entity column mapped; entity_ref will fall back to row index.",
        )

    for spec in active:
        if spec.column_name not in df_columns:
            report.add(
                Severity.ERROR,
                "missing_column",
                f"Mapped column not present in data: {spec.column_name}",
                column=spec.column_name,
            )
            continue

        if spec.domain not in _VALID_DOMAINS:
            report.add(
                Severity.ERROR,
                "invalid_domain",
                f"Domain {spec.domain!r} is not a recognized universal domain.",
                column=spec.column_name,
            )

        if spec.role == "metric":
            _validate_metric_column(df, spec, report)

        if spec.role == "entity":
            _validate_entity_column(df, spec, report)

    return report


def _validate_metric_column(df: pd.DataFrame, spec: MappingSpec, report: ValidationReport) -> None:
    series = df[spec.column_name]
    coerced = pd.to_numeric(series, errors="coerce")
    parse_rate = coerced.notna().mean()

    if parse_rate == 0:
        report.add(
            Severity.ERROR,
            "non_numeric_metric",
            f"Metric column {spec.column_name!r} has no numeric values.",
            column=spec.column_name,
        )
    elif parse_rate < 0.8:
        report.add(
            Severity.WARNING,
            "low_numeric_rate",
            f"Metric column {spec.column_name!r} is only "
            f"{parse_rate:.0%} numeric; non-numeric rows become null.",
            column=spec.column_name,
        )

    if not spec.metric_name:
        report.add(
            Severity.ERROR,
            "missing_metric_name",
            f"Metric column {spec.column_name!r} has no metric_name.",
            column=spec.column_name,
        )


def _validate_entity_column(df: pd.DataFrame, spec: MappingSpec, report: ValidationReport) -> None:
    series = df[spec.column_name]
    null_rate = series.isna().mean()
    if null_rate > 0.5:
        report.add(
            Severity.WARNING,
            "sparse_entity",
            f"Entity column {spec.column_name!r} is {null_rate:.0%} null.",
            column=spec.column_name,
        )


def raise_if_invalid(report: ValidationReport) -> None:
    """Stop the load if validation found blocking errors.

    Args:
        report: The validation report to check.

    Raises:
        ValueError: If the report contains any errors.
    """
    if not report.ok:
        messages = "; ".join(f"[{i.code}] {i.message}" for i in report.errors)
        raise ValueError(f"Validation failed: {messages}")
