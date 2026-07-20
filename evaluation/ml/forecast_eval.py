"""Rolling-origin backtest for the production forecasting job.

The forecasting job writes future-dated rows to ``ml.forecasts``, so those rows
can never be scored -- the actuals do not exist yet. This module instead holds
out the tail of each historical series, calls the *same* forecasting functions
the job uses, and compares the projection against the observed values that were
withheld.

Two design choices matter for the reported numbers:

* The production model selector ``forecasting._forecast_series`` is imported and
  reused rather than reimplemented, so the evaluation measures the deployed
  model and not a lookalike.
* Every error metric is reported beside three naive baselines. A mean absolute
  error has no meaning on its own, since its magnitude depends entirely on the
  scale of the metric being forecast; the skill score against a baseline is the
  number that says whether the model earned its place.
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

# Grouping keys that identify one forecastable series in analytics.daily_trend.
SERIES_KEYS = ("dataset_id", "business_name", "industry", "domain", "metric_name")

# A fold needs enough history left over for the model to fit after the holdout
# is removed; below this the fold is skipped instead of fitted on noise.
MIN_TRAIN_POINTS = 10


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
# Error metrics
# ---------------------------------------------------------------------------


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return the mean absolute error."""
    return float(np.mean(np.abs(actual - predicted)))


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return the root mean squared error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _mape(actual: np.ndarray, predicted: np.ndarray) -> tuple[float | None, int]:
    """Return the mean absolute percentage error and the points it used.

    MAPE is undefined wherever the actual value is zero. Those points are
    excluded rather than clamped, and the count of usable points is returned so
    a MAPE computed from a small remainder is never mistaken for a full one.

    Args:
        actual: Observed values.
        predicted: Forecast values.

    Returns:
        A tuple of the percentage error and the number of points it covers, or
        ``(None, 0)`` when every actual was zero.

    """
    mask = np.abs(actual) > 1e-9
    usable = int(mask.sum())
    if usable == 0:
        return None, 0
    errors = np.abs((actual[mask] - predicted[mask]) / actual[mask])
    return float(np.mean(errors) * 100.0), usable


def _smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return the symmetric mean absolute percentage error.

    Reported alongside MAPE because it stays defined when actuals reach zero,
    which happens on count metrics such as breakdowns.
    """
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0
    mask = denom > 1e-9
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100.0)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _baseline_naive(train: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed value across the horizon."""
    return np.full(horizon, float(train[-1]))


def _baseline_seasonal_naive(
    train: np.ndarray, horizon: int, period: int = 7
) -> np.ndarray:
    """Repeat the values observed one seasonal period earlier.

    Falls back to the plain naive forecast when the training window is shorter
    than one period.
    """
    if len(train) < period:
        return _baseline_naive(train, horizon)
    season = train[-period:]
    return np.array([float(season[i % period]) for i in range(horizon)])


def _baseline_mean(train: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the training mean across the horizon."""
    return np.full(horizon, float(np.mean(train)))


BASELINES = {
    "naive_last_value": _baseline_naive,
    "seasonal_naive_7d": _baseline_seasonal_naive,
    "train_mean": _baseline_mean,
}


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


def _fold_cutpoints(length: int, horizon: int, folds: int) -> list[int]:
    """Return the training-window sizes for each rolling-origin fold.

    The origins step backwards from the end of the series by one horizon each
    time, so the folds cover distinct, non-overlapping holdout windows.

    Args:
        length: Number of points in the full series.
        horizon: Points held out per fold.
        folds: Maximum number of folds requested.

    Returns:
        Training sizes, earliest origin first. Empty when the series is too
        short for even one fold.

    """
    cuts = []
    for i in range(folds):
        cut = length - horizon * (i + 1)
        if cut < MIN_TRAIN_POINTS:
            break
        cuts.append(cut)
    return sorted(cuts)


def _evaluate_fold(
    series: pd.Series,
    cut: int,
    horizon: int,
    forecast_series,
) -> dict[str, Any] | None:
    """Fit on the head of a series and score the projection against the tail.

    Args:
        series: The full observed series.
        cut: Number of leading points used for training.
        horizon: Number of points held out and forecast.
        forecast_series: The production model selector.

    Returns:
        A record of the fold's errors, or None when the model declined to fit.

    """
    train = series.iloc[:cut]
    test = series.iloc[cut : cut + horizon]
    if len(test) == 0:
        return None

    result = forecast_series(train, horizon)
    if result is None:
        return None

    _dates, values, lower, upper, method = result
    actual = test.values.astype(float)
    predicted = np.asarray(values, dtype=float)[: len(actual)]
    lo = np.asarray(lower, dtype=float)[: len(actual)]
    hi = np.asarray(upper, dtype=float)[: len(actual)]

    mape, mape_points = _mape(actual, predicted)
    train_values = train.values.astype(float)

    baselines = {}
    for name, fn in BASELINES.items():
        base_pred = fn(train_values, horizon)[: len(actual)]
        baselines[name] = {
            "mae": _mae(actual, base_pred),
            "rmse": _rmse(actual, base_pred),
        }

    return {
        "train_points": int(cut),
        "test_points": int(len(actual)),
        "method": method,
        "mae": _mae(actual, predicted),
        "rmse": _rmse(actual, predicted),
        "mape": mape,
        "mape_points": mape_points,
        "smape": _smape(actual, predicted),
        "interval_coverage": float(np.mean((actual >= lo) & (actual <= hi))),
        "baselines": baselines,
    }


def evaluate_series(
    meta: dict,
    series: pd.Series,
    horizon: int,
    folds: int,
    forecast_series,
) -> dict[str, Any]:
    """Backtest every fold of a single series.

    Args:
        meta: The identifying keys for this series.
        series: The prepared daily series.
        horizon: Forecast horizon in days.
        folds: Maximum rolling-origin folds to attempt.
        forecast_series: The production model selector.

    Returns:
        The series metadata together with its per-fold results.

    """
    cuts = _fold_cutpoints(len(series), horizon, folds)
    fold_results = []
    for cut in cuts:
        fold = _evaluate_fold(series, cut, horizon, forecast_series)
        if fold is not None:
            fold_results.append(fold)

    return {
        **{k: meta.get(k) for k in SERIES_KEYS},
        "series_points": int(len(series)),
        "folds_attempted": len(cuts),
        "folds_scored": len(fold_results),
        "folds": fold_results,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean_or_none(values: list[float]) -> float | None:
    """Average a list, returning None when it is empty."""
    return float(np.mean(values)) if values else None


def _skill(model_error: float | None, baseline_error: float | None) -> float | None:
    """Return the skill score of a model against a baseline.

    A positive score means the model beat the baseline; zero means it merely
    matched it; negative means the baseline was better and the model is not
    earning its complexity.
    """
    if model_error is None or not baseline_error:
        return None
    return float(1.0 - (model_error / baseline_error))


def summarise(series_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate every scored fold into the reported metrics.

    Folds are pooled rather than averaged per series, so a series with more
    history does not carry disproportionate weight through a nested mean.

    Args:
        series_results: One entry per evaluated series.

    Returns:
        A nested summary covering the overall picture, the split by forecasting
        method, and the split by business domain.

    """
    folds = [(s, f) for s in series_results for f in s["folds"]]

    def block(subset: list[tuple[dict, dict]]) -> dict[str, Any]:
        if not subset:
            return {"folds": 0}
        maes = [f["mae"] for _s, f in subset]
        rmses = [f["rmse"] for _s, f in subset]
        mapes = [f["mape"] for _s, f in subset if f["mape"] is not None]
        smapes = [f["smape"] for _s, f in subset]
        coverage = [f["interval_coverage"] for _s, f in subset]

        baseline_block = {}
        for name in BASELINES:
            b_mae = [f["baselines"][name]["mae"] for _s, f in subset]
            b_rmse = [f["baselines"][name]["rmse"] for _s, f in subset]
            baseline_block[name] = {
                "mae": _mean_or_none(b_mae),
                "rmse": _mean_or_none(b_rmse),
                "skill_score_mae": _skill(_mean_or_none(maes), _mean_or_none(b_mae)),
                "beaten_in_folds": int(
                    sum(
                        1
                        for (_s, f) in subset
                        if f["mae"] < f["baselines"][name]["mae"]
                    )
                ),
            }

        return {
            "folds": len(subset),
            "mae": _mean_or_none(maes),
            "rmse": _mean_or_none(rmses),
            "mape": _mean_or_none(mapes),
            "mape_folds": len(mapes),
            "smape": _mean_or_none(smapes),
            "interval_coverage": _mean_or_none(coverage),
            "baselines": baseline_block,
        }

    by_method: dict[str, Any] = {}
    for method in sorted({f["method"] for _s, f in folds}):
        by_method[method] = block([(s, f) for s, f in folds if f["method"] == method])

    by_domain: dict[str, Any] = {}
    for domain in sorted({str(s["domain"]) for s, _f in folds}):
        by_domain[domain] = block(
            [(s, f) for s, f in folds if str(s["domain"]) == domain]
        )

    return {
        "counts": {
            "series_total": len(series_results),
            "series_scored": sum(1 for s in series_results if s["folds_scored"]),
            "series_skipped": sum(1 for s in series_results if not s["folds_scored"]),
            "folds_scored": len(folds),
        },
        "overall": block(folds),
        "by_method": by_method,
        "by_domain": by_domain,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_evaluation(
    dataset_id: int | None,
    horizon: int,
    folds: int,
) -> dict[str, Any]:
    """Backtest the forecasting job over the selected scope.

    Args:
        dataset_id: Dataset to scope to, or None for every dataset.
        horizon: Forecast horizon in days, matching the production job.
        folds: Maximum rolling-origin folds per series.

    Returns:
        A report containing the run configuration, the aggregate metrics, and
        the per-series detail needed to explain any individual result.

    """
    jobs_dir = _add_jobs_to_path()

    import forecasting  # noqa: PLC0415 - path is only valid after _add_jobs_to_path
    from ml_common import db_conn, read_daily_trend  # noqa: PLC0415

    logger.info(
        "Loaded production forecasting module",
        extra={"jobs_dir": str(jobs_dir), "statsmodels": forecasting._HAS_SM},
    )

    with db_conn() as conn:
        df = read_daily_trend(conn, dataset_id)

    if df.empty:
        logger.warning(
            "No daily_trend rows for scope — nothing to backtest",
            extra={"dataset_id": dataset_id},
        )
        return {
            "configuration": {
                "dataset_id": dataset_id,
                "horizon": horizon,
                "folds": folds,
            },
            "summary": summarise([]),
            "series": [],
        }

    series_results = []
    for keys, group in df.groupby(list(SERIES_KEYS), dropna=False):
        meta = dict(zip(SERIES_KEYS, keys, strict=False))
        series = forecasting._prepare_series(group)
        series_results.append(
            evaluate_series(meta, series, horizon, folds, forecasting._forecast_series)
        )

    summary = summarise(series_results)

    logger.info(
        "Forecast backtest complete",
        extra={
            "dataset_id": dataset_id,
            "series": summary["counts"]["series_total"],
            "folds": summary["counts"]["folds_scored"],
        },
    )

    return {
        "configuration": {
            "dataset_id": dataset_id,
            "horizon": horizon,
            "folds": folds,
            "min_train_points": MIN_TRAIN_POINTS,
            "statsmodels_available": forecasting._HAS_SM,
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "summary": summary,
        "series": series_results,
    }


def _fmt(value: float | None, suffix: str = "") -> str:
    """Format an optional number for the console report."""
    return "n/a" if value is None else f"{value:.4f}{suffix}"


def _print_report(report: dict[str, Any]) -> None:
    """Print a short human-readable summary of a report."""
    summary = report["summary"]
    counts = summary["counts"]
    overall = summary["overall"]
    config = report["configuration"]

    print("=" * 66)
    print(" Forecasting Evaluation — rolling-origin backtest")
    print("=" * 66)
    print(
        f" dataset={config['dataset_id'] or 'all'} horizon={config['horizon']}d "
        f"series={counts['series_total']} folds={counts['folds_scored']}"
    )
    if counts["series_skipped"]:
        print(f" {counts['series_skipped']} series too short to backtest")

    if not overall.get("folds"):
        print("\n No folds scored — nothing to report.")
        print("=" * 66)
        return

    print("\n Accuracy")
    print(" " + "-" * 62)
    print(f"   MAE                   {_fmt(overall['mae'])}")
    print(f"   RMSE                  {_fmt(overall['rmse'])}")
    print(
        f"   MAPE                  {_fmt(overall['mape'], '%')}"
        f"  ({overall['mape_folds']}/{overall['folds']} folds defined)"
    )
    print(f"   sMAPE                 {_fmt(overall['smape'], '%')}")
    print(f"   95% interval coverage {_fmt(overall['interval_coverage'])}")

    print("\n Against baselines  (skill > 0 means the model wins)")
    print(" " + "-" * 62)
    for name, stats in overall["baselines"].items():
        print(
            f"   {name:<18} MAE {_fmt(stats['mae']):>10}   "
            f"skill {_fmt(stats['skill_score_mae']):>8}   "
            f"won {stats['beaten_in_folds']}/{overall['folds']} folds"
        )

    print("\n By method")
    print(" " + "-" * 62)
    for method, stats in summary["by_method"].items():
        print(
            f"   {method:<18} n={stats['folds']:<4} MAE {_fmt(stats['mae']):>10}   "
            f"sMAPE {_fmt(stats['smape'], '%')}"
        )

    print("\n By domain")
    print(" " + "-" * 62)
    for domain, stats in summary["by_domain"].items():
        print(
            f"   {domain:<18} n={stats['folds']:<4} MAE {_fmt(stats['mae']):>10}   "
            f"sMAPE {_fmt(stats['smape'], '%')}"
        )

    print("\n" + "=" * 66)


def main() -> None:
    """Parse arguments, run the backtest, and write the report."""
    parser = argparse.ArgumentParser(
        description="Rolling-origin backtest of the production forecasting job."
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=None,
        help="Scope to one dataset; omit to backtest every dataset.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=7,
        help="Forecast horizon in days; keep this equal to the production job.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Maximum rolling-origin folds per series.",
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

    report = run_evaluation(args.dataset_id, args.horizon, args.folds)
    _print_report(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        logger.info("Report written", extra={"path": str(args.out)})


if __name__ == "__main__":
    main()
