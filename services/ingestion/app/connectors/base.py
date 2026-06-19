from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Iterator

import pandas as pd


@dataclass
class SourceRecord:
    raw: dict[str, Any]
    row_index: int


@dataclass
class ConnectorMetadata:
    source_name: str
    source_type: str
    business_name: str
    industry: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_type": self.source_type,
            "business_name": self.business_name,
            "industry": self.industry,
            "extra": self.extra,
        }


class BaseConnector(abc.ABC):
    """
    Reusability seam. A new industry/source = a new subclass.
    Everything downstream (profiling, mapping, loading, hub, dashboard)
    consumes the uniform output of read_dataframe() / iter_records()
    and never needs to change.
    """

    source_type: str = "base"

    def __init__(self, metadata: ConnectorMetadata) -> None:
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
        df = self.read_dataframe()
        for idx, row in enumerate(df.to_dict(orient="records")):
            yield SourceRecord(raw=row, row_index=idx)

    def row_count(self) -> int:
        return len(self.read_dataframe())

    def describe(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "metadata": self.metadata.to_dict(),
        }


class ConnectorError(Exception):
    pass


class SourceValidationError(ConnectorError):
    pass