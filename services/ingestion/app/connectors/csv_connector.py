"""CSV connector -- the concrete reader for uploaded CSV files.

Inherits the shared connector contract, so CSV is simply the first supported
source type rather than a special case. Reading is defensive about the messy
files real businesses actually upload: the delimiter is sniffed when not given,
a failed UTF-8 decode retries in latin-1, near-empty columns are dropped, and
duplicate headers are made unique. The parsed frame is cached so repeated reads
do not re-parse the file.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from ops_common.logging import get_logger

from app.connectors.base import (
    BaseConnector,
    ConnectorMetadata,
    SourceValidationError,
)

logger = get_logger(__name__)

# A column that is ≥99% empty is treated as dead and dropped.
_MAX_EMPTY_RATIO = 0.99


# The concrete CSV reader — inherits BaseConnector so every source type
# (CSV now, API/ERP later) shares the same interface. source_type tags it.
class CSVConnector(BaseConnector):
    """Reads an uploaded CSV file into the platform's uniform frame shape."""
    source_type = "csv"

    # Store the file path + read options + metadata. _cache holds the parsed
    # DataFrame so repeated reads don't re-parse the file.
    def __init__(
        self,
        path: str | Path,
        metadata: ConnectorMetadata,
        delimiter: str | None = None,
        encoding: str = "utf-8",
    ) -> None:
        """Store the file path, read options, and source metadata.

        Args:
            path: Path to the CSV file.
            metadata: Identifying details of the source.
            delimiter: Field separator; sniffed automatically when omitted.
            encoding: Text encoding to try first.
        """
        super().__init__(metadata)
        self.path = Path(path)
        self.delimiter = delimiter
        self.encoding = encoding
        self._cache: pd.DataFrame | None = None

    # Cheap pre-checks before parsing: file exists, is a file, not zero-byte.
    # Fails fast with a clear error instead of a confusing pandas crash.
    def validate_source(self) -> None:
        """Check the file exists, is a file, and is not empty.

        These cheap checks run before parsing so a bad upload fails with a clear
        message rather than a confusing parser error.

        Raises:
            SourceValidationError: If the file is missing, not a file, or zero-byte.
        """
        if not self.path.exists():
            raise SourceValidationError(f"CSV file not found: {self.path}")
        if not self.path.is_file():
            raise SourceValidationError(f"Path is not a file: {self.path}")
        if self.path.stat().st_size == 0:
            raise SourceValidationError(f"CSV file is empty: {self.path}")

    # The actual pandas read. If no delimiter is given, use the python engine
    # with sep=None so pandas auto-sniffs the separator (comma/tab/semicolon).
    def _read_raw(self) -> pd.DataFrame:
        read_kwargs: dict = {
            "encoding": self.encoding,
            "skip_blank_lines": True,
        }
        if self.delimiter:
            read_kwargs["sep"] = self.delimiter
            read_kwargs["low_memory"] = False
        else:
            read_kwargs["sep"] = None
            read_kwargs["engine"] = "python"

        # If UTF-8 fails on a messy file, retry once with latin-1 (common fix).
        try:
            return pd.read_csv(self.path, **read_kwargs)
        except UnicodeDecodeError:
            logger.warning(
                "UTF-8 decode failed, retrying with latin-1",
                extra={"file": str(self.path)},
            )
            read_kwargs["encoding"] = "latin-1"
            return pd.read_csv(self.path, **read_kwargs)

    # The main public method: validate → read → clean → cache → return.
    # Cleaning = normalize headers, drop dead columns, dedupe duplicate names.
    def read_dataframe(self) -> pd.DataFrame:
        """Read, clean, and cache the CSV as a normalized frame.

        Validates the file, parses it, normalizes headers, drops near-empty columns,
        and makes duplicate column names unique.

        Returns:
            The cleaned frame.

        Raises:
            SourceValidationError: If the file is invalid or yields no rows or columns.
        """
        if self._cache is not None:
            return self._cache

        self.validate_source()
        df = self._read_raw()

        if df.empty or len(df.columns) == 0:
            raise SourceValidationError(f"CSV produced no rows or columns: {self.path}")

        df = self.normalize_columns(df)
        df = self._drop_dead_columns(df)
        df = self._dedupe_columns(df)

        self._cache = df
        logger.info(
            "CSV loaded",
            extra={
                "file": self.path.name,
                "rows": len(df),
                "columns": len(df.columns),
                "business": self.metadata.business_name,
            },
        )
        return df

    # Drop columns that are almost entirely empty (≥99% NaN) — they carry no
    # signal and would just clutter profiling/mapping.
    def _drop_dead_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = []
        for col in df.columns:
            empty_ratio = df[col].isna().mean()
            if empty_ratio >= _MAX_EMPTY_RATIO:
                logger.info("Dropping near-empty column", extra={"column": col})
                continue
            keep.append(col)
        return df[keep]

    # Make duplicate column names unique (col, col_1, col_2...) so downstream
    # code doesn't break on two columns sharing a name.
    def _dedupe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        seen: dict[str, int] = {}
        new_cols: list[str] = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        df.columns = new_cols
        return df

    # Convenience constructor: build the connector straight from an upload,
    # packaging path + business name + industry into the metadata object.
    @classmethod
    def from_upload(
        cls,
        path: str | Path,
        business_name: str,
        industry: str | None = None,
    ) -> CSVConnector:
        """Build a connector directly from an uploaded file.

        Args:
            path: Path to the stored upload.
            business_name: Business the dataset belongs to.
            industry: Optional industry label.

        Returns:
            A connector configured for that upload.
        """
        path = Path(path)
        metadata = ConnectorMetadata(
            source_name=path.name,
            source_type=cls.source_type,
            business_name=business_name,
            industry=industry,
        )
        return cls(path=path, metadata=metadata)
