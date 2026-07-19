"""Shared pytest configuration for the platform's unit tests.

The services are separate packages mounted into the runtime images at fixed
paths rather than installed as one distribution, so the paths the routers add at
import time are added here too. Anything genuinely unavailable in the image a
test runs in is skipped by that test rather than failing the suite, which keeps
the same suite meaningful whether it is run inside the API container or from a
checkout.

These are unit tests: nothing here touches Postgres, DuckDB, Spark, Airflow, or
the network. Those layers are integration concerns and are verified by running
the stack, not by this suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Repository root, whether the suite runs from a checkout or from /app in an image.
_ROOT = Path(__file__).resolve().parent.parent

# Mirrors the sys.path inserts the API routers perform: the RAG modules are
# imported flat, the agent as a package under services/, and the ML and
# intelligence packages from their own directories.
_SERVICE_PATHS = [
    _ROOT / "services",
    _ROOT / "services" / "rag",
    _ROOT / "services" / "intelligence",
    _ROOT / "services" / "ml" / "jobs",
    _ROOT / "services" / "ingestion",
    _ROOT / "packages" / "common",
    _ROOT / "packages",
]

for _path in _SERVICE_PATHS:
    resolved = str(_path)
    if _path.exists() and resolved not in sys.path:
        sys.path.insert(0, resolved)


@pytest.fixture
def csv_file(tmp_path):
    """Return a factory that writes CSV text to a temporary file.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A callable taking the file contents and an optional name, returning the
        path it was written to.
    """

    def _write(content: str, name: str = "sample.csv") -> Path:
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        return path

    return _write


@pytest.fixture
def clean_env(monkeypatch):
    """Remove the ``OPS_`` settings a test might otherwise inherit.

    Environment-driven defaults are only meaningful if the surrounding process
    has not already set the variable under test.

    Args:
        monkeypatch: pytest's environment patcher.

    Returns:
        The monkeypatch fixture, for setting deliberately invalid values.
    """
    for key in (
        "OPS_TARGET_DATASET_ID",
        "OPS_AGENT_TEMPERATURE",
        "OPS_AGENT_LLM_RETRIES",
        "OPS_AGENT_NUM_CTX",
        "OPS_AGENT_MEMORY_TURNS",
        "OPS_AGENT_MAX_STEPS",
    ):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch
