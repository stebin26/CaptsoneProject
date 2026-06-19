from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ops_common.logging import get_logger

logger = get_logger(__name__)

_MAX_SAMPLE_VALUES = 5
_DATETIME_HINT_TOKENS = ("date", "time", "timestamp", "_at", "_on", "dt")


@dataclass
class ColumnProfile:
    column_name: str
    data_type: str
    distinct_count: int
    null_count: int
    total_count: int
    sample_values: list[Any] = field(default_factory=list)
    is_numeric: bool = False
    is_datetime: bool = False
    is_identifier: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_name": self.column_name,
            "data_type": self.data_type,
            "distinct_count": self.distinct_count,
            "null_count": self.null_count,
            "total_count": self.total_count,
            "sample_values": self.sample_values,
            "is_numeric": self.is_numeric,
            "is_datetime": self.is_datetime,
            "is_identifier": self.is_identifier,
        }


@dataclass
class DatasetProfile:
    source_filename: str
    row_count: int
    columns: list[ColumnProfile] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_filename": self.source_filename,
            "row_count": self.row_count,
            "columns": [c.to_dict() for c in self.columns],
        }


def _infer_logical_type(series: pd.Series, column_name: str) -> tuple[str, bool, bool]:
    non_null = series.dropna()

    if non_null.empty:
        return "empty", False, False

    if pd.api.types.is_bool_dtype(series):
        return "boolean", False, False

    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        return ("integer" if pd.api.types.is_integer_dtype(series) else "float"), True, False

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime", False, True

    name_lower = column_name.lower()
    if any(tok in name_lower for tok in _DATETIME_HINT_TOKENS):
        parsed = pd.to_datetime(non_null, errors="coerce", utc=False)
        if parsed.notna().mean() >= 0.8:
            return "datetime", False, True

    coerced = pd.to_numeric(non_null, errors="coerce")
    if coerced.notna().mean() >= 0.9:
        is_int = (coerced.dropna() % 1 == 0).all()
        return ("integer" if is_int else "float"), True, False

    return "string", False, False


def _is_identifier(series: pd.Series, distinct_count: int, total_count: int, column_name: str) -> bool:
    name_lower = column_name.lower()
    if name_lower.endswith("_id") or name_lower == "id" or name_lower.endswith("_ref"):
        return True
    if total_count == 0:
        return False
    uniqueness = distinct_count / total_count
    return uniqueness >= 0.95


def _sample_values(series: pd.Series) -> list[Any]:
    non_null = series.dropna()
    if non_null.empty:
        return []
    uniques = non_null.unique()[:_MAX_SAMPLE_VALUES]
    out: list[Any] = []
    for v in uniques:
        if hasattr(v, "item"):
            try:
                out.append(v.item())
                continue
            except (ValueError, AttributeError):
                pass
        out.append(str(v) if not isinstance(v, (int, float, bool, str)) else v)
    return out


def profile_dataframe(df: pd.DataFrame, source_filename: str) -> DatasetProfile:
    total = len(df)
    columns: list[ColumnProfile] = []

    for col in df.columns:
        series = df[col]
        data_type, is_numeric, is_datetime = _infer_logical_type(series, str(col))
        distinct = int(series.nunique(dropna=True))
        nulls = int(series.isna().sum())
        identifier = _is_identifier(series, distinct, total, str(col))

        profile = ColumnProfile(
            column_name=str(col),
            data_type=data_type,
            distinct_count=distinct,
            null_count=nulls,
            total_count=total,
            sample_values=_sample_values(series),
            is_numeric=is_numeric,
            is_datetime=is_datetime,
            is_identifier=identifier,
        )
        columns.append(profile)

    logger.info(
        "Profiled dataset",
        extra={"file": source_filename, "rows": total, "columns": len(columns)},
    )
    return DatasetProfile(
        source_filename=source_filename,
        row_count=total,
        columns=columns,
    )


def profile_csv(path: str | Path, source_filename: str | None = None) -> DatasetProfile:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return profile_dataframe(df, source_filename or path.name)