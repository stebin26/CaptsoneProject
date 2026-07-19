"""Unit tests for the shared package and the API's error boundary.

Two things are protected here. The first is ``ops_common``: every service
depends on it, so a mistake in the settings or the engine lifecycle shows up
everywhere at once. The second is the mapping from domain errors to HTTP status
codes -- the boundary that decides whether a caller sees a useful 400 or an
opaque 500.

No test opens a database connection. The engine tests exercise the lifecycle
around ``create_engine``, not a live server.
"""

from __future__ import annotations

import pytest
from ops_common import db as ops_db
from ops_common.config import Settings
from ops_common.domain.models import Domain, model_for_domain
from ops_common.domain.registry import (
    DOMAIN_REGISTRY,
    features_for_domain,
    get_spec,
    match_domain_by_keyword,
)
from pydantic import ValidationError

# ============================================================
# Settings
# ============================================================


def test_connection_strings_are_derived_from_one_set_of_values():
    """Host and credentials cannot drift between the three connection styles.

    They are derived rather than configured separately precisely so that a
    change in one place cannot leave another pointing somewhere else.
    """
    settings = Settings(
        postgres_host="db.internal",
        postgres_port=6543,
        postgres_db="ops_test",
        postgres_user="tester",
        postgres_password="secret",
    )

    for dsn in (
        settings.postgres_dsn,
        settings.sqlalchemy_dsn,
        settings.duckdb_attach_dsn,
    ):
        assert "db.internal" in dsn
        assert "6543" in dsn
        assert "ops_test" in dsn

    assert settings.sqlalchemy_dsn.startswith("postgresql+psycopg://")
    assert settings.postgres_dsn.startswith("postgresql://")


def test_settings_fall_back_to_documented_defaults(monkeypatch):
    """With nothing configured, the defaults match the compose file."""
    for key in ("OPS_POSTGRES_HOST", "OPS_POSTGRES_PORT", "OPS_POSTGRES_DB"):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)

    assert settings.postgres_host == "postgres"
    assert settings.postgres_port == 5432
    assert settings.postgres_db == "ops"


def test_a_non_numeric_port_is_rejected(monkeypatch):
    """A port that is not a number fails validation rather than being coerced."""
    with pytest.raises(ValidationError):
        Settings(postgres_port="not-a-port")


def test_ensure_dirs_creates_every_working_directory(tmp_path):
    """A fresh container never fails on a missing upload path."""
    settings = Settings(
        upload_dir=tmp_path / "uploads",
        rag_upload_dir=tmp_path / "rag",
        mapping_config_dir=tmp_path / "configs",
        duckdb_path=str(tmp_path / "duck" / "analytics.duckdb"),
    )

    settings.ensure_dirs()

    assert settings.upload_dir.is_dir()
    assert settings.rag_upload_dir.is_dir()
    assert settings.mapping_config_dir.is_dir()
    assert (tmp_path / "duck").is_dir()


# ============================================================
# Engine lifecycle
# ============================================================


def test_a_missing_session_factory_raises_instead_of_asserting(monkeypatch):
    """The guard survives ``python -O``, where a bare assert would vanish.

    An assert here would be stripped from an optimised build and the failure
    would resurface much later as an unexplained NoneType error.
    """
    monkeypatch.setattr(ops_db, "_engine", object())
    monkeypatch.setattr(ops_db, "_SessionFactory", None)

    with pytest.raises(RuntimeError, match="Session factory"):
        ops_db.get_session_factory()


def test_a_failed_engine_creation_leaves_no_partial_state(monkeypatch):
    """A failure must not publish an engine without its session factory.

    Publishing them separately once left the engine set and the factory None,
    which made the next call skip creation entirely and fail downstream.
    """
    monkeypatch.setattr(ops_db, "_engine", None)
    monkeypatch.setattr(ops_db, "_SessionFactory", None)

    def boom(*args, **kwargs):
        raise RuntimeError("bad DSN")

    monkeypatch.setattr(ops_db, "create_engine", boom)

    with pytest.raises(RuntimeError, match="bad DSN"):
        ops_db.get_engine()

    assert ops_db._engine is None
    assert ops_db._SessionFactory is None


# ============================================================
# Universal domain model
# ============================================================


def test_every_domain_has_a_registry_entry():
    """The eight universal domains are all described; none is a stub."""
    assert len(DOMAIN_REGISTRY) == len(list(Domain))

    for domain in Domain:
        spec = get_spec(domain)
        assert spec.aliases, f"{domain} has no aliases to map columns by"
        assert features_for_domain(domain), f"{domain} defines no features"


def test_every_domain_maps_to_a_hub_table():
    """A confirmed mapping always has somewhere to load into."""
    for domain in Domain:
        assert model_for_domain(domain.value) is not None


def test_an_unknown_domain_is_rejected():
    """A value outside the universal model cannot reach the hub."""
    with pytest.raises((KeyError, ValueError)):
        model_for_domain("teleportation")


@pytest.mark.parametrize(
    ("column_name", "expected"),
    [
        ("machine_id", Domain.ASSETS),
        ("defect_count", Domain.QUALITY),
        ("repair_cost", Domain.MAINTENANCE),
    ],
)
def test_industry_wording_maps_onto_universal_domains(column_name, expected):
    """Keyword matching is what makes the platform industry-agnostic."""
    assert match_domain_by_keyword(column_name) == expected


def test_an_unrecognised_column_matches_nothing():
    """No match is returned rather than a wrong one guessed."""
    assert match_domain_by_keyword("xyzzy_42") is None


# ============================================================
# Domain errors to HTTP status codes
# ============================================================


def test_invalid_input_is_a_client_error_not_a_server_error():
    """A malformed upload answers 400, and a bug answers 500.

    Before this boundary existed both were 500, so a user could not tell a bad
    file from a broken service.
    """
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from api_app.main import _make_domain_handler

    class SourceValidationError(Exception):
        pass

    app = fastapi.FastAPI()
    app.add_exception_handler(
        SourceValidationError, _make_domain_handler(400, "invalid upload")
    )

    @app.get("/bad-upload")
    def bad_upload():
        raise SourceValidationError("CSV is malformed and could not be parsed")

    response = TestClient(app, raise_server_exceptions=False).get("/bad-upload")

    assert response.status_code == 400
    assert "malformed" in response.json()["detail"]


def test_an_unavailable_dependency_is_a_retryable_503():
    """A model or embedding backend being down is not the caller's fault."""
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from api_app.main import _make_domain_handler

    class EmbeddingError(Exception):
        pass

    app = fastapi.FastAPI()
    app.add_exception_handler(
        EmbeddingError, _make_domain_handler(503, "embedding unavailable")
    )

    @app.get("/embed")
    def embed():
        raise EmbeddingError("model could not be loaded")

    assert TestClient(app, raise_server_exceptions=False).get("/embed").status_code == (
        503
    )


def test_an_unexpected_error_returns_a_generic_message():
    """Internal detail belongs in the log, not in the HTTP response."""
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from api_app.main import unhandled_exception_handler

    app = fastapi.FastAPI()
    app.add_exception_handler(Exception, unhandled_exception_handler)

    @app.get("/boom")
    def boom():
        raise RuntimeError("connection string password=hunter2")

    response = TestClient(app, raise_server_exceptions=False).get("/boom")

    assert response.status_code == 500
    assert "hunter2" not in response.text
    assert "internal error" in response.json()["detail"].lower()
