"""Column profiling -- reading a dataset's shape before anything is mapped.

Walks every column and records its inferred type, distinct and null counts, and
a few sample values, plus the flags (numeric, datetime, identifier) the mapping
suggester relies on. Profiling first is what makes automatic onboarding
possible: the suggester classifies columns from evidence about the data rather
than from the column name alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from ops_common.logging import get_logger

logger = get_logger(__name__)

# Keep at most 5 sample values per column. Column-name tokens that hint the
# column is a date/time (used to decide datetime type).
_MAX_SAMPLE_VALUES = 5
_DATETIME_HINT_TOKENS = ("date", "time", "timestamp", "_at", "_on", "dt")


# The profiling result for ONE column: its type, distinct/null counts, samples,
# and three boolean flags (numeric? datetime? identifier?) the suggester uses.
@dataclass
class ColumnProfile:
    """The profile of one column: its type, counts, samples, and flags."""
    column_name: str
    data_type: str
    distinct_count: int
    null_count: int
    total_count: int
    sample_values: list[Any] = field(default_factory=list)
    is_numeric: bool = False
    is_datetime: bool = False
    is_identifier: bool = False

    # Serialize to a plain dict (for JSON storage / API response).
    def to_dict(self) -> dict[str, Any]:
        """Return this profile as a plain dictionary."""
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


# The profiling result for the WHOLE dataset: filename, row count, and the
# list of per-column profiles above.
@dataclass
class DatasetProfile:
    """The profile of a whole dataset: its source, size, and column profiles."""
    source_filename: str
    row_count: int
    columns: list[ColumnProfile] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return this profile as a plain dictionary."""
        return {
            "source_filename": self.source_filename,
            "row_count": self.row_count,
            "columns": [c.to_dict() for c in self.columns],
        }


# The core type-detection logic. Returns (type_name, is_numeric, is_datetime).
# Goes in priority order so the most reliable signal wins.
def _infer_logical_type(series: pd.Series, column_name: str) -> tuple[str, bool, bool]:
    non_null = series.dropna()

    # All-null column → "empty", nothing more to infer.
    if non_null.empty:
        return "empty", False, False

    # True/False column.
    if pd.api.types.is_bool_dtype(series):
        return "boolean", False, False

    # Already a proper numeric dtype → int or float, flag as numeric.
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        return ("integer" if pd.api.types.is_integer_dtype(series) else "float"), True, False

    # Already a proper datetime dtype.
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime", False, True

    # Name looks like a date column → try parsing; if ≥80% parse, call it datetime.
    # This catches dates stored as text.
    name_lower = column_name.lower()
    if any(tok in name_lower for tok in _DATETIME_HINT_TOKENS):
        # Even with errors="coerce", exotic dtypes can still raise. Treating the
        # column as text is the correct fallback: type inference is a hint for
        # the mapping suggester, not something worth failing an upload over.
        try:
            parsed = pd.to_datetime(non_null, errors="coerce", utc=False)
        except Exception:
            logger.warning(
                "Datetime inference failed for column %s, treating as text",
                column_name,
                extra={"column": column_name},
                exc_info=True,
            )
        else:
            if parsed.notna().mean() >= 0.8:
                return "datetime", False, True

    # Numbers stored as text → if ≥90% coerce to number, treat as numeric.
    # Then check if all whole numbers to pick integer vs float.
    try:
        coerced = pd.to_numeric(non_null, errors="coerce")
    except Exception:
        logger.warning(
            "Numeric inference failed for column %s, treating as text",
            column_name,
            extra={"column": column_name},
            exc_info=True,
        )
    else:
        if coerced.notna().mean() >= 0.9:
            is_int = (coerced.dropna() % 1 == 0).all()
            return ("integer" if is_int else "float"), True, False

    # Default: plain text.
    return "string", False, False


# Decide if a column is an identifier (an entity key like machine_id).
# Rule: name ends in _id/_ref OR the values are ≥95% unique.
# Identifiers become the entity_ref in the hub, not a metric.
def _is_identifier(
    series: pd.Series, distinct_count: int, total_count: int, column_name: str
) -> bool:
    name_lower = column_name.lower()
    if name_lower.endswith("_id") or name_lower == "id" or name_lower.endswith("_ref"):
        return True
    if total_count == 0:
        return False
    uniqueness = distinct_count / total_count
    return uniqueness >= 0.95


# Grab up to 5 distinct example values, converted to clean Python scalars
# (numpy types → native) so they serialize to JSON safely.
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


# The main entry: walk every column, build its ColumnProfile, and assemble
# the DatasetProfile. This is what the suggester consumes next.
def profile_dataframe(df: pd.DataFrame, source_filename: str) -> DatasetProfile:
    """Profile every column of a frame.

    Args:
        df: The frame to profile.
        source_filename: Name of the file the frame came from.

    Returns:
        The dataset profile the suggester consumes next.
    """
    total = len(df)
    columns: list[ColumnProfile] = []

    for col in df.columns:
        series = df[col]
        try:
            data_type, is_numeric, is_datetime = _infer_logical_type(series, str(col))
            distinct = int(series.nunique(dropna=True))
            nulls = int(series.isna().sum())
            identifier = _is_identifier(series, distinct, total, str(col))
        except Exception:
            # Naming the offending column is the whole point: without it the
            # user sees a pandas error with no idea which column caused it.
            logger.exception(
                "Could not profile column %s of %s",
                col,
                source_filename,
                extra={"column": str(col), "file": source_filename},
            )
            raise

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


# Convenience: profile straight from a CSV path. Reads the file, strips
# whitespace from headers, then runs profile_dataframe.
def profile_csv(path: str | Path, source_filename: str | None = None) -> DatasetProfile:
    """Profile a CSV file directly from disk.

    Args:
        path: Path to the CSV file.
        source_filename: Name to record; defaults to the file's own name.

    Returns:
        The dataset profile.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If the CSV exists but cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    try:
        df = pd.read_csv(path, low_memory=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as exc:
        logger.exception("Could not read CSV for profiling", extra={"file": str(path)})
        raise ValueError(f"CSV could not be read for profiling: {path} ({exc})") from exc

    df.columns = [str(c).strip() for c in df.columns]
    return profile_dataframe(df, source_filename or path.name)
