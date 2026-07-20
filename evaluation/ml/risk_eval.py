"""Evaluation of the unsupervised risk scoring job.

There are no failure labels for the scored entities, so precision, recall and F1
are not computed here and no threshold is chosen. Manufacturing labels in order
to produce a classification score would measure the labelling, not the model.
What can be established without labels is measured instead, in four parts.

**Distribution.** What the scores and bands actually look like, including how
much of the 0-100 range is used.

**Sensitivity.** Each of the three weighted components is zeroed in turn and the
entities are rescored, with the rank agreement against the baseline reported. A
component that can be removed without changing the ordering is not contributing,
whatever weight it carries in the formula.

**Stability.** The score is min-max normalised across the scored population, so
it is relative by construction: the worst entity present receives a high score
whether or not it is in poor condition absolutely. Each entity is dropped in
turn and the remainder rescored, which quantifies how far the population
determines an individual result.

**Convergent validity.** The ranking is compared against an observable the score
never reads -- mean downtime and breakdown counts straight from the hub. Agreement
does not prove the score is right, but disagreement would be evidence it is
wrong, and that is worth knowing either way.

A caveat runs through all of it: the scored populations here are small. With a
handful of entities per domain, rank statistics are coarse and a single position
change moves a coefficient a long way. The population size is reported beside
every figure rather than left in a footnote.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Candidate locations for the ML job package, most likely first. The jobs use
# flat imports (``from ml_common import ...``), so the jobs directory itself has
# to be on the path rather than its parent.
_JOB_DIR_CANDIDATES = (
    Path("/app/services/ml/jobs"),
    Path(__file__).resolve().parents[2] / "services" / "ml" / "jobs",
)

# Rank statistics below this population size are reported but flagged, because
# a single swapped position dominates the coefficient.
SMALL_POPULATION = 6

# Metrics used for convergent validity, and whether a higher value means worse
# condition. None of these are inputs to the risk score itself.
VALIDITY_METRICS = {
    "downtime_minutes": True,
    "breakdowns": True,
}


def _add_jobs_to_path() -> Path:
    """Put the ML jobs directory on ``sys.path``.

    Returns:
        The directory that was added.

    Raises:
        FileNotFoundError: If no candidate directory exists, which means the
            evaluation is running somewhere the ML code was never copied to.

    """
    for candidate in _JOB_DIR_CANDIDATES:
        if candidate.is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise FileNotFoundError(
        "Could not locate the ML jobs directory; looked in "
        + ", ".join(str(c) for c in _JOB_DIR_CANDIDATES)
    )


# ---------------------------------------------------------------------------
# Rank statistics, implemented directly to avoid a scipy dependency
# ---------------------------------------------------------------------------


def _ranks(values: np.ndarray) -> np.ndarray:
    """Return average ranks, with ties sharing the mean of their positions."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Return the Spearman rank correlation between two sequences.

    Args:
        a: First sequence.
        b: Second sequence.

    Returns:
        The coefficient, or None when either sequence is constant or too short
        for a correlation to be defined.

    """
    if len(a) < 3 or len(a) != len(b):
        return None
    ra, rb = _ranks(np.asarray(a, dtype=float)), _ranks(np.asarray(b, dtype=float))
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float | None:
    """Return Kendall's tau-b between two sequences.

    Used for rank agreement rather than Spearman because it degrades more
    gracefully on the very small populations scored here.

    Args:
        a: First sequence.
        b: Second sequence.

    Returns:
        The coefficient, or None when it is undefined.

    """
    n = len(a)
    if n < 2 or n != len(b):
        return None
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    concordant = discordant = tie_a = tie_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            product = da * db
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            else:
                if da == 0:
                    tie_a += 1
                if db == 0:
                    tie_b += 1

    denom = np.sqrt(
        (concordant + discordant + tie_a) * (concordant + discordant + tie_b)
    )
    if denom < 1e-12:
        return None
    return float((concordant - discordant) / denom)


# ---------------------------------------------------------------------------
# Rescoring helpers
# ---------------------------------------------------------------------------


def _score_frame(
    risk_scoring,
    domain: str,
    frame: pd.DataFrame,
    anomaly_weights: dict,
) -> pd.DataFrame:
    """Score one domain's entities using the production scoring function.

    Args:
        risk_scoring: The imported risk scoring module.
        domain: Domain being scored.
        frame: Entity feature rows for that domain.
        anomaly_weights: Per-entity anomaly weights.

    Returns:
        A frame of entity ids, scores and bands, ordered by descending score.

    """
    rows = risk_scoring._score_domain(domain, frame, anomaly_weights, "eval")
    if not rows:
        return pd.DataFrame(columns=["entity_id", "risk_score", "risk_level"])
    out = pd.DataFrame(
        [
            {
                "entity_id": r["entity_id"],
                "risk_score": r["risk_score"],
                "risk_level": r["risk_level"],
            }
            for r in rows
        ]
    )
    return out.sort_values("risk_score", ascending=False).reset_index(drop=True)


def _weight_ablation(
    risk_scoring,
    domain: str,
    frame: pd.DataFrame,
    anomaly_weights: dict,
    baseline: pd.DataFrame,
) -> dict[str, Any]:
    """Zero each component weight in turn and measure the rank change.

    The module-level weights are restored afterwards, so the production job is
    unaffected by having been evaluated.

    Args:
        risk_scoring: The imported risk scoring module.
        domain: Domain being scored.
        frame: Entity feature rows.
        anomaly_weights: Per-entity anomaly weights.
        baseline: The unmodified scoring result.

    Returns:
        Per-component rank agreement against the baseline.

    """
    originals = {
        "trend": risk_scoring.W_TREND,
        "volatility": risk_scoring.W_VOLATILITY,
        "anomaly": risk_scoring.W_ANOMALY,
    }
    attributes = {
        "trend": "W_TREND",
        "volatility": "W_VOLATILITY",
        "anomaly": "W_ANOMALY",
    }

    results: dict[str, Any] = {}
    base_lookup = dict(zip(baseline["entity_id"], baseline["risk_score"], strict=False))

    for component, attribute in attributes.items():
        setattr(risk_scoring, attribute, 0.0)
        try:
            ablated = _score_frame(risk_scoring, domain, frame, anomaly_weights)
        finally:
            setattr(risk_scoring, attribute, originals[component])

        if ablated.empty:
            results[component] = {"weight": originals[component], "tau": None}
            continue

        shared = [e for e in ablated["entity_id"] if e in base_lookup]
        base_scores = np.array([base_lookup[e] for e in shared], dtype=float)
        ablated_lookup = dict(
            zip(ablated["entity_id"], ablated["risk_score"], strict=False)
        )
        new_scores = np.array([ablated_lookup[e] for e in shared], dtype=float)

        results[component] = {
            "weight": originals[component],
            "rank_agreement_tau": kendall_tau(base_scores, new_scores),
            "mean_absolute_score_change": float(
                np.mean(np.abs(base_scores - new_scores))
            ),
        }

    return results


def _leave_one_out_stability(
    risk_scoring,
    domain: str,
    frame: pd.DataFrame,
    anomaly_weights: dict,
    baseline: pd.DataFrame,
) -> dict[str, Any]:
    """Drop each entity in turn and measure how far the survivors move.

    Args:
        risk_scoring: The imported risk scoring module.
        domain: Domain being scored.
        frame: Entity feature rows.
        anomaly_weights: Per-entity anomaly weights.
        baseline: The unmodified scoring result.

    Returns:
        The mean and worst score movement across all leave-one-out runs, plus
        the rank agreement among the surviving entities.

    """
    entities = list(baseline["entity_id"])
    if len(entities) < 3:
        return {"runs": 0, "note": "population too small for leave-one-out"}

    base_lookup = dict(zip(baseline["entity_id"], baseline["risk_score"], strict=False))
    shifts: list[float] = []
    taus: list[float] = []

    for dropped in entities:
        subset = frame[frame["entity_id"] != dropped]
        if subset.empty:
            continue
        rescored = _score_frame(risk_scoring, domain, subset, anomaly_weights)
        if rescored.empty:
            continue
        lookup = dict(zip(rescored["entity_id"], rescored["risk_score"], strict=False))
        shared = [e for e in entities if e != dropped and e in lookup]
        if len(shared) < 2:
            continue
        before = np.array([base_lookup[e] for e in shared], dtype=float)
        after = np.array([lookup[e] for e in shared], dtype=float)
        shifts.append(float(np.mean(np.abs(before - after))))
        tau = kendall_tau(before, after)
        if tau is not None:
            taus.append(tau)

    return {
        "runs": len(shifts),
        "mean_score_shift": float(np.mean(shifts)) if shifts else None,
        "max_score_shift": float(np.max(shifts)) if shifts else None,
        "mean_rank_agreement_tau": float(np.mean(taus)) if taus else None,
    }


def _convergent_validity(
    conn,
    dataset_ids: list[int],
    baseline: pd.DataFrame,
    read_daily_trend,
) -> dict[str, Any]:
    """Compare the risk ranking against observables the score never reads.

    Args:
        conn: An open database connection.
        dataset_ids: Datasets covered by the scored population.
        baseline: The scoring result to validate.
        read_daily_trend: The production trend reader.

    Returns:
        Per-metric rank correlations, or a note explaining why none could be
        computed.

    """
    entity_scores = dict(
        zip(baseline["entity_id"], baseline["risk_score"], strict=False)
    )

    sql = """
        SELECT entity_ref AS entity_id, metric_name, avg_value
        FROM analytics.entity_features
        WHERE dataset_id = ANY(%s)
    """
    try:
        features = pd.read_sql(sql, conn, params=(dataset_ids,))
    except Exception:
        logger.warning("Could not read entity features for validity", exc_info=True)
        return {"available": False}

    if features.empty:
        return {"available": False}

    results: dict[str, Any] = {}
    for metric, higher_is_worse in VALIDITY_METRICS.items():
        subset = features[features["metric_name"] == metric]
        if subset.empty:
            continue
        observed = subset.groupby("entity_id")["avg_value"].mean()
        shared = [e for e in observed.index if e in entity_scores]
        if len(shared) < 3:
            results[metric] = {"entities": len(shared), "note": "too few entities"}
            continue
        scores = np.array([entity_scores[e] for e in shared], dtype=float)
        values = np.array([observed[e] for e in shared], dtype=float)
        if not higher_is_worse:
            values = -values
        results[metric] = {
            "entities": len(shared),
            "spearman": spearman(scores, values),
            "kendall_tau": kendall_tau(scores, values),
        }

    return {"available": bool(results), "metrics": results}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_evaluation(dataset_id: int | None) -> dict[str, Any]:
    """Evaluate the risk scoring job without labels.

    Args:
        dataset_id: Dataset to scope to, or None for every dataset.

    Returns:
        A report containing the run configuration and, per scored domain, the
        score distribution, weight sensitivity, population stability, and
        convergent validity against independent observables.

    """
    jobs_dir = _add_jobs_to_path()

    import risk_scoring  # noqa: PLC0415 - path valid only after the call above
    from ml_common import (  # noqa: PLC0415
        db_conn,
        read_daily_trend,
        read_entity_features,
    )

    logger.info(
        "Loaded production risk scoring module",
        extra={"jobs_dir": str(jobs_dir)},
    )

    domains: dict[str, Any] = {}

    with db_conn() as conn:
        features = read_entity_features(conn, dataset_id)
        if features.empty:
            logger.warning(
                "No entity_features rows for scope — nothing to evaluate",
                extra={"dataset_id": dataset_id},
            )
            return {
                "configuration": {"dataset_id": dataset_id},
                "domains": {},
            }

        features["domain"] = features["domain"].astype(str).str.lower()
        anomaly_weights = risk_scoring._load_anomaly_weights(conn, dataset_id)

        for domain in sorted(risk_scoring.TARGET_DOMAINS):
            frame = features[features["domain"] == domain]
            if frame.empty:
                domains[domain] = {"entities": 0, "note": "no rows in this domain"}
                continue

            baseline = _score_frame(risk_scoring, domain, frame, anomaly_weights)
            if baseline.empty:
                domains[domain] = {"entities": 0, "note": "scoring produced no rows"}
                continue

            scores = baseline["risk_score"].values.astype(float)
            dataset_ids = sorted({int(d) for d in frame["dataset_id"].unique()})

            domains[domain] = {
                "entities": int(len(baseline)),
                "small_population": bool(len(baseline) < SMALL_POPULATION),
                "distribution": {
                    "min": float(np.min(scores)),
                    "median": float(np.median(scores)),
                    "max": float(np.max(scores)),
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "range_used": float(np.max(scores) - np.min(scores)),
                    "by_level": baseline["risk_level"].value_counts().to_dict(),
                },
                "ranking": [
                    {
                        "entity_id": r.entity_id,
                        "risk_score": r.risk_score,
                        "risk_level": r.risk_level,
                    }
                    for r in baseline.itertuples()
                ],
                "weight_sensitivity": _weight_ablation(
                    risk_scoring, domain, frame, anomaly_weights, baseline
                ),
                "population_stability": _leave_one_out_stability(
                    risk_scoring, domain, frame, anomaly_weights, baseline
                ),
                "convergent_validity": _convergent_validity(
                    conn, dataset_ids, baseline, read_daily_trend
                ),
            }

    logger.info(
        "Risk scoring evaluation complete",
        extra={"dataset_id": dataset_id, "domains": len(domains)},
    )

    return {
        "configuration": {
            "dataset_id": dataset_id,
            "weights": {
                "trend": risk_scoring.W_TREND,
                "volatility": risk_scoring.W_VOLATILITY,
                "anomaly": risk_scoring.W_ANOMALY,
            },
            "target_domains": sorted(risk_scoring.TARGET_DOMAINS),
            "validity_metrics": sorted(VALIDITY_METRICS),
            "classification_metrics": (
                "not computed — the scored entities carry no failure labels"
            ),
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "domains": domains,
    }


def _fmt(value: float | None, suffix: str = "") -> str:
    """Format an optional number for the console report."""
    return "n/a" if value is None else f"{value:.4f}{suffix}"


def _print_report(report: dict[str, Any]) -> None:
    """Print a short human-readable summary of a report."""
    config = report["configuration"]

    print("=" * 66)
    print(" Risk Scoring Evaluation — unsupervised, no labels")
    print("=" * 66)
    print(f" dataset={config.get('dataset_id') or 'all'}")
    print(f" weights={config.get('weights')}")
    print(" Precision/recall/F1 are not reported: there are no failure labels.")

    if not report["domains"]:
        print("\n No domains scored.")
        print("=" * 66)
        return

    for domain, block in report["domains"].items():
        print(f"\n Domain: {domain}")
        print(" " + "-" * 62)
        if not block.get("entities"):
            print(f"   {block.get('note', 'nothing scored')}")
            continue

        note = (
            "  (small population — rank statistics are coarse)"
            if block["small_population"]
            else ""
        )
        print(f"   Entities scored       {block['entities']}{note}")

        dist = block["distribution"]
        print(
            f"   Score range           {_fmt(dist['min'])} – {_fmt(dist['max'])}"
            f"   median {_fmt(dist['median'])}"
        )
        print(f"   Bands                 {dist['by_level']}")

        print("   Ranking")
        for row in block["ranking"]:
            print(
                f"     {str(row['entity_id']):<10} "
                f"{row['risk_score']:>7.2f}  {row['risk_level']}"
            )

        print("   Weight sensitivity  (tau near 1 means the component is inert)")
        for component, stats in block["weight_sensitivity"].items():
            print(
                f"     {component:<12} w={stats['weight']:<6} "
                f"tau {_fmt(stats.get('rank_agreement_tau')):>8}   "
                f"mean score change {_fmt(stats.get('mean_absolute_score_change'))}"
            )

        stability = block["population_stability"]
        print("   Population stability  (leave-one-out)")
        if not stability.get("runs"):
            print(f"     {stability.get('note', 'not computed')}")
        else:
            print(
                f"     runs {stability['runs']}   "
                f"mean shift {_fmt(stability['mean_score_shift'])}   "
                f"max shift {_fmt(stability['max_score_shift'])}   "
                f"tau {_fmt(stability['mean_rank_agreement_tau'])}"
            )

        validity = block["convergent_validity"]
        print("   Convergent validity  (against metrics the score never reads)")
        if not validity.get("available"):
            print("     no comparable observable available")
        else:
            for metric, stats in validity["metrics"].items():
                if "note" in stats:
                    print(f"     {metric:<20} {stats['note']}")
                else:
                    print(
                        f"     {metric:<20} n={stats['entities']}   "
                        f"spearman {_fmt(stats['spearman'])}   "
                        f"tau {_fmt(stats['kendall_tau'])}"
                    )

    print("\n" + "=" * 66)


def main() -> None:
    """Parse arguments, run the evaluation, and write the report."""
    parser = argparse.ArgumentParser(
        description="Label-free evaluation of the unsupervised risk scoring job."
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=None,
        help="Scope to one dataset; omit to evaluate every dataset.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the full JSON report to this path.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    report = run_evaluation(args.dataset_id)
    _print_report(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        logger.info("Report written", extra={"path": str(args.out)})


if __name__ == "__main__":
    main()
