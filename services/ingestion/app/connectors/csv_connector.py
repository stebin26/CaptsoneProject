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

_MAX_EMPTY_RATIO = 0.99


class CSVConnector(BaseConnector):
    source_type = "csv"

    def __init__(
        self,
        path: str | Path,
        metadata: ConnectorMetadata,
        delimiter: str | None = None,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__(metadata)
        self.path = Path(path)
        self.delimiter = delimiter
        self.encoding = encoding
        self._cache: pd.DataFrame | None = None

    def validate_source(self) -> None:
        if not self.path.exists():
            raise SourceValidationError(f"CSV file not found: {self.path}")
        if not self.path.is_file():
            raise SourceValidationError(f"Path is not a file: {self.path}")
        if self.path.stat().st_size == 0:
            raise SourceValidationError(f"CSV file is empty: {self.path}")

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

        try:
            return pd.read_csv(self.path, **read_kwargs)
        except UnicodeDecodeError:
            logger.warning(
                "UTF-8 decode failed, retrying with latin-1",
                extra={"file": str(self.path)},
            )
            read_kwargs["encoding"] = "latin-1"
            return pd.read_csv(self.path, **read_kwargs)

    def read_dataframe(self) -> pd.DataFrame:
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

    def _drop_dead_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = []
        for col in df.columns:
            empty_ratio = df[col].isna().mean()
            if empty_ratio >= _MAX_EMPTY_RATIO:
                logger.info("Dropping near-empty column", extra={"column": col})
                continue
            keep.append(col)
        return df[keep]

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

    @classmethod
    def from_upload(
        cls,
        path: str | Path,
        business_name: str,
        industry: str | None = None,
    ) -> "CSVConnector":
        path = Path(path)
        metadata = ConnectorMetadata(
            source_name=path.name,
            source_type=cls.source_type,
            business_name=business_name,
            industry=industry,
        )
        return cls(path=path, metadata=metadata)
