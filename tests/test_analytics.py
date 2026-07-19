"""Unit tests for the analytics, ML, and intelligence layers.

Covered here is the logic that decides *what* gets computed and *whether it is
safe to compute it*: how a job resolves its scope, how scores are banded, what
happens to a delete whose identifiers cannot be trusted, and how the knowledge
graph behaves when the file backing it is wrong.

The Spark and ML jobs themselves are not executed. Running them needs a cluster
and a populated hub, which makes them integration territory; what is unit-tested
is the pure logic they depend on.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pandas")

# Skipped rather than failed where the ML jobs are not mounted, so the same
# suite stays meaningful in every image it runs in.
ml_common = pytest.importorskip("ml_common")


# ============================================================
# Job scope resolution
# ============================================================


def test_absent_dataset_id_means_a_full_batch(clean_env):
    """With nothing set, a job processes every dataset."""
    assert ml_common.target_dataset_id(argv=["job.py"]) is None
    assert ml_common.announce_mode(None) == "all"


def test_dataset_id_from_argument_scopes_the_run(clean_env):
    """A numeric argument scopes the job to one dataset."""
    assert ml_common.target_dataset_id(argv=["job.py", "42"]) == 42
    assert ml_common.announce_mode(42) == "42"


def test_dataset_id_from_environment_scopes_the_run(clean_env):
    """The environment is used when no argument is given."""
    clean_env.setenv("OPS_TARGET_DATASET_ID", "7")
    assert ml_common.target_dataset_id(argv=["job.py"]) == 7


def test_non_numeric_dataset_id_falls_back_to_full_batch(clean_env, caplog):
    """A malformed scope runs the full batch, and says so.

    Silently widening an intended incremental run to every dataset is the kind
    of thing that goes unnoticed, so the fallback has to be logged.
    """
    clean_env.setenv("OPS_TARGET_DATASET_ID", "not-a-number")

    with caplog.at_level("WARNING"):
        assert ml_common.target_dataset_id(argv=["job.py"]) is None

    assert "not-a-number" in caplog.text


def test_an_orchestrator_command_line_is_not_read_as_a_dataset_id(
    clean_env, monkeypatch
):
    """A job imported by a worker ignores that worker's arguments.

    Under Airflow the jobs are imported and run in-process, so sys.argv is the
    Celery worker's command line. A numeric value anywhere on it -- a
    concurrency flag, for example -- would otherwise scope the run to a dataset
    nobody asked for.
    """
    monkeypatch.setattr(
        ml_common.sys, "argv", ["airflow", "celery", "worker", "--concurrency", "16"]
    )

    assert ml_common.target_dataset_id() is None


def test_an_orchestrator_command_line_does_not_hide_the_environment(
    clean_env, monkeypatch
):
    """Ignoring the worker's arguments still leaves the configured scope intact."""
    monkeypatch.setattr(
        ml_common.sys, "argv", ["airflow", "celery", "worker", "--concurrency", "16"]
    )
    clean_env.setenv("OPS_TARGET_DATASET_ID", "7")

    assert ml_common.target_dataset_id() == 7


def test_invalid_database_port_is_rejected_by_name(clean_env):
    """A bad port names the variable to fix, not just the failed conversion."""
    clean_env.setenv("OPS_POSTGRES_PORT", "abcd")

    with pytest.raises(ValueError, match="OPS_POSTGRES_PORT"):
        ml_common._db_config()


# ============================================================
# Score banding
# ============================================================


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.0, "low"), (32.9, "low"), (33.0, "medium"), (65.9, "medium"),
     (66.0, "high"), (100.0, "high")],
)
def test_scores_are_banded_at_the_shared_thresholds(score, expected):
    """Every job bands scores identically, including on the boundaries."""
    assert ml_common.bucket_level(score) == expected


# ============================================================
# Spark job guards
# ============================================================


def test_untrusted_dataset_ids_never_reach_a_delete():
    """A non-integer identifier aborts the delete instead of being interpolated.

    The identifiers are formatted into the SQL, so this is the check that stops
    an unusable value from widening or corrupting the statement.
    """
    spark_session = pytest.importorskip("spark_session")

    with pytest.raises(ValueError, match="integers"):
        spark_session._delete_existing(
            "analytics.domain_metrics", [1, "not-an-id"], "assets"
        )


# ============================================================
# Knowledge graph
# ============================================================


def _write_graph(tmp_path, payload):
    path = tmp_path / "relationships.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _use_graph(monkeypatch, engine, path):
    """Point the engine at a test graph, bypassing the process-wide cache.

    ``load_graph`` is lru_cached, so the uncached function is captured before
    the attribute is replaced; reading it back afterwards would find the
    replacement instead.
    """
    uncached = engine.load_graph.__wrapped__
    monkeypatch.setattr(engine, "load_graph", lambda *a, **k: uncached(path))


def test_missing_graph_file_is_reported_not_silently_empty(tmp_path):
    """A missing relationships file fails loudly rather than yielding no edges."""
    engine = pytest.importorskip("inference_engine")

    with pytest.raises(FileNotFoundError):
        engine.load_graph.__wrapped__(str(tmp_path / "absent.json"))


def test_invalid_graph_json_is_reported_as_a_value_error(tmp_path):
    """A corrupt relationships file gives a message naming the file."""
    engine = pytest.importorskip("inference_engine")
    path = tmp_path / "broken.json"
    path.write_text("{edges: oops", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        engine.load_graph.__wrapped__(str(path))


def test_one_malformed_edge_does_not_discard_the_graph(tmp_path, caplog):
    """A bad edge is dropped and logged; the usable edges still load."""
    engine = pytest.importorskip("inference_engine")
    path = _write_graph(
        tmp_path,
        {
            "edges": [
                {"source": "Assets", "target": "Maintenance", "strength": "strong"},
                {"target": "Quality"},
                {"source": "Finance", "target": "Customers"},
            ]
        },
    )

    with caplog.at_level("WARNING"):
        edges = engine.load_graph.__wrapped__(path)

    assert len(edges) == 2
    assert "malformed edge 1" in caplog.text


def test_graph_edges_are_normalised_to_lowercase(tmp_path):
    """Domain names are case-insensitive once loaded."""
    engine = pytest.importorskip("inference_engine")
    path = _write_graph(
        tmp_path, {"edges": [{"source": "Assets", "target": "Maintenance"}]}
    )

    edge = engine.load_graph.__wrapped__(path)[0]
    assert (edge.source, edge.target) == ("assets", "maintenance")


def test_a_cycle_in_the_graph_terminates_the_walk(tmp_path, monkeypatch):
    """Mutually dependent domains do not send the traversal into a loop.

    Assets drives Maintenance and Maintenance drives Assets. Without the visited
    set this walk would not terminate.
    """
    engine = pytest.importorskip("inference_engine")
    path = _write_graph(
        tmp_path,
        {
            "edges": [
                {"source": "assets", "target": "maintenance", "strength": "strong"},
                {"source": "maintenance", "target": "assets", "strength": "strong"},
            ]
        },
    )
    _use_graph(monkeypatch, engine, path)

    signals = {
        "assets": engine.DomainSignal(domain="assets", strength=0.9, risk=0.8),
        "maintenance": engine.DomainSignal(
            domain="maintenance", strength=0.7, risk=0.6
        ),
    }

    insights = engine.infer(signals)

    assert insights, "a lit cycle should still produce insights"
    for insight in insights:
        targets = [i.target for i in insight.impacts]
        assert insight.root not in targets, (
            "the walk stepped back into the domain it started from"
        )
        assert len(targets) == len(set(targets)), (
            f"a domain was reached twice from one root: {targets}"
        )
        assert all(i.hop < engine.MAX_HOPS for i in insight.impacts)


def test_domains_below_the_signal_threshold_produce_no_insight(tmp_path, monkeypatch):
    """An edge is only lit when both ends carry a real signal."""
    engine = pytest.importorskip("inference_engine")
    path = _write_graph(
        tmp_path, {"edges": [{"source": "assets", "target": "quality"}]}
    )
    _use_graph(monkeypatch, engine, path)

    quiet = {
        "assets": engine.DomainSignal(domain="assets", strength=0.01),
        "quality": engine.DomainSignal(domain="quality", strength=0.01),
    }

    assert engine.infer(quiet) == []
