"""The connector contract -- the platform's portability seam.

Every source format has its own connector; everything downstream (profiling,
mapping, loading, hub, dashboard) consumes the uniform output of
``read_dataframe`` and ``iter_records`` and never learns which format the data
came from. That is what lets a new industry or source be onboarded with a new
subclass and a mapping config rather than changes throughout the codebase.
"""
from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class SourceRecord:
    """One raw source row with its position in the source."""
    raw: dict[str, Any]
    row_index: int


@dataclass
class ConnectorMetadata:
    """Identifying details of a source: what it is and whose it is."""
    source_name: str
    source_type: str
    business_name: str
    industry: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return this metadata as a plain dictionary."""
        return {
            "source_name": self.source_name,
            "source_type": self.source_type,
            "business_name": self.business_name,
            "industry": self.industry,
            "extra": self.extra,
        }


class BaseConnector(abc.ABC):
    """Base class every source connector inherits from -- the portability seam.

    A new industry or source format is a new subclass. Everything downstream
    (profiling, mapping, loading, hub, dashboard) consumes the uniform output of
    ``read_dataframe`` and ``iter_records`` and never needs to change.
    """

    source_type: str = "base"

    def __init__(self, metadata: ConnectorMetadata) -> None:
        """Store the metadata describing this source.

        Args:
            metadata: Identifying details of the source being read.
        """
        self.metadata = metadata

    @abc.abstractmethod
    def read_dataframe(self) -> pd.DataFrame:
        """Return the full source as a normalized pandas DataFrame."""
        raise NotImplementedError

    @abc.abstractmethod
    def validate_source(self) -> None:
        """Raise if the source is missing, unreadable, or malformed."""
        raise NotImplementedError

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the frame with its column names cleaned to a common form.

        Args:
            df: The frame to normalize.

        Returns:
            A copy with normalized column names.
        """
        df = df.copy()
        df.columns = [self._clean_column_name(str(c)) for c in df.columns]
        return df

    @staticmethod
    def _clean_column_name(name: str) -> str:
        cleaned = name.strip().lower().replace(" ", "_").replace("-", "_")
        cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")
        return cleaned.strip("_")

    def iter_records(self) -> Iterator[SourceRecord]:
        """Iterate the source one record at a time.

        Yields:
            Each source row with its index.
        """
        df = self.read_dataframe()
        for idx, row in enumerate(df.to_dict(orient="records")):
            yield SourceRecord(raw=row, row_index=idx)

    def row_count(self) -> int:
        """Return the number of rows in the source."""
        return len(self.read_dataframe())

    def describe(self) -> dict[str, Any]:
        """Return a short description of this connector and its source.

        Returns:
            The source type and its metadata.
        """
        return {
            "source_type": self.source_type,
            "metadata": self.metadata.to_dict(),
        }


class ConnectorError(Exception):
    """Base error for any connector failure."""
    pass


class SourceValidationError(ConnectorError):
    """Raised when a source is missing, unreadable, or malformed."""
    pass
