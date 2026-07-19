"""Unit tests for the ingestion layer.

An uploaded CSV is the least trustworthy input the platform accepts, so these
tests focus on what happens when it is wrong: malformed, empty, missing, or
accompanied by a mapping payload that does not describe it. The contract being
protected is that every one of those cases produces the connector's own error
with a message naming the file or the offending entry, never a raw pandas
traceback escaping to the API.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pandas")

from app.connectors.base import ConnectorMetadata, SourceValidationError  # noqa: E402
from app.connectors.csv_connector import CSVConnector  # noqa: E402
from app.profiling.profiler import profile_dataframe  # noqa: E402
from app.transforms import _normalize_specs  # noqa: E402


def _connector(path, delimiter=None):
    """Build a connector for a file without going through the upload helper."""
    metadata = ConnectorMetadata(
        source_name=path.name,
        source_type="csv",
        business_name="Acme Manufacturing",
    )
    return CSVConnector(path, metadata, delimiter=delimiter)


# ============================================================
# Reading a CSV
# ============================================================


def test_missing_file_is_rejected_by_name(tmp_path):
    """A path that does not exist fails validation, naming the file."""
    missing = tmp_path / "not_here.csv"
    with pytest.raises(SourceValidationError, match="not found"):
        _connector(missing).validate_source()


def test_empty_file_is_rejected(csv_file):
    """A zero-byte upload is rejected before any parsing is attempted."""
    with pytest.raises(SourceValidationError, match="empty"):
        _connector(csv_file("")).validate_source()


def test_malformed_csv_raises_source_validation_error(csv_file):
    """A ragged CSV surfaces as SourceValidationError, not a pandas ParserError.

    This is the case that previously reached the API as an unhandled 500.
    """
    path = csv_file("a,b,c\n1,2,3\n4,5,6,7,8,9\n", name="ragged.csv")
    with pytest.raises(SourceValidationError) as exc_info:
        _connector(path, delimiter=",").read_dataframe()

    assert "ragged.csv" in str(exc_info.value)


def test_valid_csv_loads_and_reports_its_shape(csv_file):
    """A well-formed CSV loads, and the connector reports its row count."""
    path = csv_file("machine_id,temp_celsius\nM-1,70.5\nM-2,71.0\n")
    connector = _connector(path)
    connector.validate_source()

    frame = connector.read_dataframe()
    assert list(frame.columns) == ["machine_id", "temp_celsius"]
    assert connector.row_count() == 2


def test_column_names_are_cleaned_on_read(csv_file):
    """Whitespace and casing in headers are normalised at the boundary."""
    path = csv_file("  Machine ID ,Temp Celsius\nM-1,70.5\n")
    frame = _connector(path).read_dataframe()

    assert all(c == c.strip() for c in frame.columns)
    assert all(" " not in c for c in frame.columns)


# ============================================================
# Mapping payloads
# ============================================================


def test_mapping_entry_without_column_name_names_its_position():
    """A malformed mapping entry is rejected with its index, not a bare KeyError."""
    mapping = [
        {"column_name": "temp_celsius", "domain": "assets", "role": "metric"},
        {"domain": "quality"},
    ]
    with pytest.raises(ValueError, match="entry 1"):
        _normalize_specs(mapping)


def test_valid_mapping_normalises_to_specs():
    """A well-formed mapping produces one spec per entry, defaults applied."""
    specs = _normalize_specs(
        [
            {"column_name": "temp_celsius", "domain": "assets", "role": "metric"},
            {"column_name": "notes"},
        ]
    )

    assert [s.column_name for s in specs] == ["temp_celsius", "notes"]
    assert specs[1].role == "skip"
    assert specs[1].domain is None


# ============================================================
# Profiling
# ============================================================


def test_profiling_classifies_numeric_and_text_columns(csv_file):
    """Type inference distinguishes numbers from free text."""
    path = csv_file("machine_id,temp_celsius\nM-1,70.5\nM-2,71.0\nM-3,69.8\n")
    frame = _connector(path).read_dataframe()

    profile = profile_dataframe(frame, "sample.csv")
    by_name = {c.column_name: c for c in profile.columns}

    assert by_name["temp_celsius"].is_numeric is True
    assert by_name["machine_id"].is_numeric is False


def test_profiling_counts_rows_and_nulls(csv_file):
    """The profile reports the dataset shape and per-column null counts."""
    path = csv_file("machine_id,temp_celsius\nM-1,70.5\nM-2,\nM-3,69.8\n")
    frame = _connector(path).read_dataframe()

    profile = profile_dataframe(frame, "sample.csv")
    by_name = {c.column_name: c for c in profile.columns}

    assert profile.row_count == 3
    assert by_name["temp_celsius"].null_count == 1
