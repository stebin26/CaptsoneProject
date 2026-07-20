"""Evaluation of the anomaly detection job.

The detector is unsupervised and the hub holds no labelled anomalies, so there
is no honest way to compute precision and recall against the production data as
it stands. This module therefore reports two separate things and keeps them
separate:

* A **synthetic injection benchmark**. Real series are copied, anomalies of a
  known magnitude are injected at known positions, and the production detector
  is run over the modified series. The injected positions are the ground truth.
  Because the labels were manufactured, these numbers describe the detector's
  sensitivity to a controlled disturbance -- they are not a claim about
  real-world accuracy.
* A **descriptive profile** of the detector's behaviour on the unmodified data:
  how much it flags, which method handled each group, and how severity is
  distributed. No accuracy is claimed here at all, because none can be.

One subtlety drives the precision reporting. A real series may already contain
natural outliers, so a detection that does not coincide with an injected point
is not automatically wrong. Each series is therefore scored twice -- once clean
and once injected -- and detections present in both runs are excluded from the
false-positive count. Raw and adjusted precision are both reported.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
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

# Grouping keys that identify one detectable series in analytics.daily_trend.
SERIES_KEYS = ("dataset_id", "business_name", "industry", "domain", "metric_name")

# Injection magnitudes, expressed as multiples of the series' own standard
# deviation. A 2-sigma disturbance is deliberately near the detector's z-score
# threshold of 2.5, so the benchmark shows where sensitivity begins to fall off
# rather than only confirming that a huge spike is obvious.
INJECTION_SIGMAS = (2.0, 3.0, 5.0)

# Anomalies injected per series per run.
INJECTIONS_PER_SERIES = 3

# Points at each end of a series left untouched, since an anomaly at the very
# first or last observation confounds detection with edge effects.
EDGE_MARGIN = 3

# A series needs enough points for the isolation forest path to be reachable and
# for injections to be spaced apart.
MIN_SERIES_POINTS = 15

# Fixed so the benchmark is reproducible run to run.
RANDOM_SEED = 42


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
# Injection
# ---------------------------------------------------------------------------


def _injection_positions(
    length: int, count: int, rng: np.random.Generator
) -> list[int]:
    """Choose spaced positions in a series to disturb.

    Args:
        length: Number of points in the series.
        count: Number of positions wanted.
        rng: Seeded random generator.

    Returns:
        Sorted index positions, excluding the series edges.

    """
    usable = np.arange(EDGE_MARGIN, length - EDGE_MARGIN)
    if len(usable) == 0:
        return []
    take = min(count, len(usable))
    return sorted(int(p) for p in rng.choice(usable, size=take, replace=False))


def _inject(series: pd.Series, positions: list[int], sigma: float) -> pd.Series:
    """Return a copy of a series with spikes injected at given positions.

    The disturbance is scaled by the series' own standard deviation so the
    benchmark means the same thing for a metric measured in units and one
    measured in thousands. Direction alternates so the benchmark covers both
    spikes and dips rather than only testing one tail.

    Args:
        series: The original observed series.
        positions: Index positions to disturb.
        sigma: Magnitude of the disturbance in standard deviations.

    Returns:
        The modified series.

    """
    values = series.values.astype(float).copy()
    std = float(np.std(values))
    if std < 1e-9:
        # A flat series has no scale to work from; fall back to a fraction of
        # the level so the injection is still detectable in principle.
        std = max(abs(float(np.mean(values))) * 0.1, 1.0)

    for i, pos in enumerate(positions):
        direction = 1.0 if i % 2 == 0 else -1.0
        values[pos] = values[pos] + direction * sigma * std

    return pd.Series(values, index=series.index)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _detected_positions(rows: list[dict], index: pd.Index) -> set[int]:
    """Map detected anomaly dates back to index positions in the series.

    Args:
        rows: Rows emitted by the production detector.
        index: The series index the detector ran over.

    Returns:
        The set of flagged index positions.

    """
    lookup = {pd.Timestamp(d).date(): i for i, d in enumerate(index)}
    positions = set()
    for row in rows:
        date = row.get("anomaly_date")
        if date is None:
            continue
        pos = lookup.get(date)
        if pos is not None:
            positions.add(pos)
    return positions


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    """Compute precision, recall, and F1 from a confusion count.

    Returns:
        The three scores, with None where the denominator is zero rather than a
        misleading zero.

    """
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision and recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0 if (precision is not None and recall is not None) else None
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_series_injection(
    meta: dict,
    series: pd.Series,
    sigma: float,
    rng: np.random.Generator,
    rows_from_group,
    version: str,
) -> dict[str, Any] | None:
    """Run the injection benchmark for one series at one magnitude.

    Args:
        meta: Identifying keys for the series.
        series: The observed series.
        sigma: Injection magnitude in standard deviations.
        rng: Seeded random generator.
        rows_from_group: The production detection entry point.
        version: Model version string passed through to the detector.

    Returns:
        The per-series confusion counts, or None when the series is too short.

    """
    if len(series) < MIN_SERIES_POINTS:
        return None

    positions = _injection_positions(len(series), INJECTIONS_PER_SERIES, rng)
    if not positions:
        return None

    clean_rows = rows_from_group(meta, series, None, version)
    clean_hits = _detected_positions(clean_rows, series.index)

    injected = _inject(series, positions, sigma)
    injected_rows = rows_from_group(meta, injected, None, version)
    injected_hits = _detected_positions(injected_rows, injected.index)

    truth = set(positions)
    tp = len(injected_hits & truth)
    fn = len(truth - injected_hits)
    raw_fp = len(injected_hits - truth)
    # A flag that the detector also raised on the untouched series reflects a
    # genuine feature of the data, not an error introduced by the injection.
    adjusted_fp = len(injected_hits - truth - clean_hits)

    methods = Counter(r["method"] for r in injected_rows)

    return {
        **{k: meta.get(k) for k in SERIES_KEYS},
        "series_points": int(len(series)),
        "injected": len(truth),
        "detected": len(injected_hits),
        "detected_on_clean_series": len(clean_hits),
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives_raw": raw_fp,
        "false_positives_adjusted": adjusted_fp,
        "method": methods.most_common(1)[0][0] if methods else None,
    }


# ---------------------------------------------------------------------------
# Descriptive profile on unmodified data
# ---------------------------------------------------------------------------


def profile_real_data(
    df: pd.DataFrame, prepare_series, rows_from_group, version: str
) -> dict[str, Any]:
    """Describe the detector's behaviour on the unmodified series.

    No accuracy is computed here. The purpose is to show the flag rate, which
    detection path each group took, and how severity is distributed, so the
    benchmark numbers can be read against how the detector actually behaves in
    production.

    Args:
        df: Rows read from analytics.daily_trend.
        prepare_series: The production series preparation helper.
        rows_from_group: The production detection entry point.
        version: Model version string passed through to the detector.

    Returns:
        Counts and distributions describing the run.

    """
    total_points = 0
    all_rows: list[dict] = []
    series_count = 0

    for keys, group in df.groupby(list(SERIES_KEYS), dropna=False):
        meta = dict(zip(SERIES_KEYS, keys, strict=False))
        series = prepare_series(group)
        if series.empty:
            continue
        series_count += 1
        total_points += len(series)
        all_rows.extend(rows_from_group(meta, series, None, version))

    severity = Counter(r["severity"] for r in all_rows)
    methods = Counter(r["method"] for r in all_rows)
    by_domain = Counter(str(r["domain"]) for r in all_rows)

    return {
        "series_scanned": series_count,
        "observations_scanned": total_points,
        "anomalies_flagged": len(all_rows),
        "flag_rate": (len(all_rows) / total_points) if total_points else None,
        "by_severity": dict(severity),
        "by_method": dict(methods),
        "by_domain": dict(by_domain),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarise_injection(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-series injection results into the reported metrics.

    Confusion counts are pooled across series rather than averaged, so a series
    where nothing was detectable cannot drag a mean around.

    Args:
        results: One entry per series per magnitude, each carrying its sigma.

    Returns:
        Overall and per-magnitude scores.

    """

    def block(subset: list[dict[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {"series": 0}
        tp = sum(r["true_positives"] for r in subset)
        fn = sum(r["false_negatives"] for r in subset)
        fp_raw = sum(r["false_positives_raw"] for r in subset)
        fp_adj = sum(r["false_positives_adjusted"] for r in subset)
        return {
            "series": len(subset),
            "injected": sum(r["injected"] for r in subset),
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives_raw": fp_raw,
            "false_positives_adjusted": fp_adj,
            "raw": _prf(tp, fp_raw, fn),
            "adjusted": _prf(tp, fp_adj, fn),
        }

    by_sigma = {}
    for sigma in sorted({r["sigma"] for r in results}):
        by_sigma[f"{sigma:g}_sigma"] = block(
            [r for r in results if r["sigma"] == sigma]
        )

    by_method: dict[str, Any] = {}
    for method in sorted({str(r["method"]) for r in results if r["method"]}):
        by_method[method] = block([r for r in results if str(r["method"]) == method])

    return {
        "overall": block(results),
        "by_magnitude": by_sigma,
        "by_method": by_method,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_evaluation(dataset_id: int | None) -> dict[str, Any]:
    """Evaluate the anomaly detector over the selected scope.

    Args:
        dataset_id: Dataset to scope to, or None for every dataset.

    Returns:
        A report containing the run configuration, the injection benchmark, the
        descriptive profile, and the per-series detail behind both.

    """
    jobs_dir = _add_jobs_to_path()

    import anomaly_detection as ad  # noqa: PLC0415 - path valid only after setup
    from ml_common import db_conn, read_daily_trend  # noqa: PLC0415

    logger.info(
        "Loaded production anomaly detection module",
        extra={"jobs_dir": str(jobs_dir), "sklearn": ad._HAS_SKLEARN},
    )

    with db_conn() as conn:
        df = read_daily_trend(conn, dataset_id)

    config = {
        "dataset_id": dataset_id,
        "injection_sigmas": list(INJECTION_SIGMAS),
        "injections_per_series": INJECTIONS_PER_SERIES,
        "min_series_points": MIN_SERIES_POINTS,
        "random_seed": RANDOM_SEED,
        "sklearn_available": ad._HAS_SKLEARN,
        "detector_params": {
            "contamination": ad.IFOREST_CONTAMINATION,
            "zscore_threshold": ad.ZSCORE_THRESHOLD,
            "min_points_iforest": ad.MIN_POINTS_IFOREST,
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }

    if df.empty:
        logger.warning(
            "No daily_trend rows for scope — nothing to evaluate",
            extra={"dataset_id": dataset_id},
        )
        return {
            "configuration": config,
            "injection_benchmark": summarise_injection([]),
            "real_data_profile": {},
            "series": [],
        }

    version = "evaluation"
    profile = profile_real_data(df, ad._prepare_series, ad._rows_from_group, version)

    rng = np.random.default_rng(RANDOM_SEED)
    series_results: list[dict[str, Any]] = []

    for sigma in INJECTION_SIGMAS:
        for keys, group in df.groupby(list(SERIES_KEYS), dropna=False):
            meta = dict(zip(SERIES_KEYS, keys, strict=False))
            series = ad._prepare_series(group)
            result = evaluate_series_injection(
                meta, series, sigma, rng, ad._rows_from_group, version
            )
            if result is not None:
                result["sigma"] = sigma
                series_results.append(result)

    summary = summarise_injection(series_results)

    logger.info(
        "Anomaly evaluation complete",
        extra={
            "dataset_id": dataset_id,
            "series_evaluated": len(series_results),
            "flag_rate": profile.get("flag_rate"),
        },
    )

    return {
        "configuration": config,
        "injection_benchmark": summary,
        "real_data_profile": profile,
        "series": series_results,
    }


def _fmt(value: float | None, suffix: str = "") -> str:
    """Format an optional number for the console report."""
    return "n/a" if value is None else f"{value:.4f}{suffix}"


def _print_report(report: dict[str, Any]) -> None:
    """Print a short human-readable summary of a report."""
    config = report["configuration"]
    bench = report["injection_benchmark"]
    profile = report["real_data_profile"]

    print("=" * 66)
    print(" Anomaly Detection Evaluation")
    print("=" * 66)
    print(
        f" dataset={config['dataset_id'] or 'all'} "
        f"sklearn={config['sklearn_available']} "
        f"contamination={config['detector_params']['contamination']}"
    )

    if profile:
        print("\n Behaviour on unmodified data  (no accuracy claimed)")
        print(" " + "-" * 62)
        print(
            f"   {profile['anomalies_flagged']} flagged from "
            f"{profile['observations_scanned']} observations across "
            f"{profile['series_scanned']} series"
        )
        print(f"   Flag rate             {_fmt(profile['flag_rate'])}")
        print(f"   By method             {profile['by_method']}")
        print(f"   By severity           {profile['by_severity']}")

    overall = bench["overall"]
    if not overall.get("series"):
        print("\n No series were long enough for the injection benchmark.")
        print("=" * 66)
        return

    print("\n Synthetic injection benchmark  (labels are manufactured)")
    print(" " + "-" * 62)
    print(f"   Injected anomalies    {overall['injected']}")
    print(
        f"   Recall                {_fmt(overall['raw']['recall'])}  "
        f"({overall['true_positives']} of {overall['injected']} recovered)"
    )
    print(f"   Precision (raw)       {_fmt(overall['raw']['precision'])}")
    print(f"   Precision (adjusted)  {_fmt(overall['adjusted']['precision'])}")
    print(f"   F1 (adjusted)         {_fmt(overall['adjusted']['f1'])}")

    print("\n By injection magnitude")
    print(" " + "-" * 62)
    for label, stats in bench["by_magnitude"].items():
        print(
            f"   {label:<10} recall {_fmt(stats['raw']['recall']):>8}   "
            f"precision(adj) {_fmt(stats['adjusted']['precision']):>8}   "
            f"F1 {_fmt(stats['adjusted']['f1']):>8}"
        )

    if bench["by_method"]:
        print("\n By detection method")
        print(" " + "-" * 62)
        for method, stats in bench["by_method"].items():
            print(
                f"   {method:<18} n={stats['series']:<4} "
                f"recall {_fmt(stats['raw']['recall']):>8}   "
                f"F1(adj) {_fmt(stats['adjusted']['f1']):>8}"
            )

    print("\n" + "=" * 66)


def main() -> None:
    """Parse arguments, run the evaluation, and write the report."""
    parser = argparse.ArgumentParser(
        description="Evaluate the anomaly detection job by synthetic injection."
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
