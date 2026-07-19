"""Forecasting job -- the 'Future' column of the intelligence view.

Reads the daily trend series, fits a time-series model per domain metric, and
writes projected values with confidence bounds to ``ml.forecasts``. Series too
short to model are skipped rather than extrapolated, since a forecast from two
points would look authoritative while meaning nothing.
"""
# Forecasting job — the "Future" column. Reads analytics.daily_trend, fits a
# per domain-metric time-series model, and writes future values to ml.forecasts.

from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import pandas as pd

from ml_common import (
    announce_mode,
    configure_job_logging,
    db_conn,
    make_version,
    read_daily_trend,
    register_model_version,
    target_dataset_id,
    write_forecasts,
)

logger = logging.getLogger(__name__)

# statsmodels is optional: without it the job still runs, using the linear
# fallback for every series. The reason it is unavailable is logged once at
# import so a silently degraded run is never a mystery later.
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    _HAS_SM = True
except ImportError:
    logger.warning(
        "statsmodels is not installed — falling back to linear trend forecasts",
        exc_info=True,
    )
    _HAS_SM = False

warnings.filterwarnings("ignore")

# How many days ahead to forecast, and the minimum history each method needs.
HORIZON = int(os.getenv("OPS_FORECAST_HORIZON", "7"))
MIN_POINTS_SMOOTHING = 6
MIN_POINTS_LINEAR = 3


# Collapses a daily_trend group into an ordered, gap-filled daily series of avg_value.
def _prepare_series(group: pd.DataFrame) -> pd.Series:
    s = (
        group.sort_values("trend_date")
        .set_index("trend_date")["avg_value"]
        .astype(float)
    )
    s = s[~s.index.duplicated(keep="last")]
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx).interpolate(method="linear").ffill().bfill()
    return s


# Holt-Winters additive-trend forecast with residual-based bounds; primary method.
def _forecast_smoothing(series: pd.Series, horizon: int):
    model = ExponentialSmoothing(series, trend="add", seasonal=None).fit()
    fitted = model.fittedvalues
    resid_std = float(np.std(series.values - fitted.values))
    fc = model.forecast(horizon)
    margin = 1.96 * resid_std
    dates = pd.date_range(
        series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    return dates, fc.values, fc.values - margin, fc.values + margin, "holt_winters"


# Linear-trend (least-squares) forecast fallback for short series.
def _forecast_linear(series: pd.Series, horizon: int):
    y = series.values.astype(float)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    resid_std = float(np.std(y - (slope * x + intercept)))
    fx = np.arange(len(y), len(y) + horizon)
    fy = slope * fx + intercept
    margin = 1.96 * resid_std
    dates = pd.date_range(
        series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    return dates, fy, fy - margin, fy + margin, "linear_trend"


# Picks the best available method for a series given its length; None if too short.
def _forecast_series(series: pd.Series, horizon: int):
    if _HAS_SM and len(series) >= MIN_POINTS_SMOOTHING:
        try:
            return _forecast_smoothing(series, horizon)
        except Exception:
            # Holt-Winters fails on degenerate series (all-constant values,
            # non-finite input). That is expected often enough not to fail the
            # job, but it is recorded so a run that quietly used the weaker
            # method everywhere can still be diagnosed.
            logger.warning(
                "Holt-Winters failed on a %d-point series — using linear trend",
                len(series),
                extra={"series_length": len(series), "method": "holt_winters"},
                exc_info=True,
            )
    if len(series) >= MIN_POINTS_LINEAR:
        return _forecast_linear(series, horizon)
    return None


# Turns one series' forecast arrays into ml.forecasts row dicts.
def _rows_from_forecast(
    meta: dict, dates, values, lo, hi, model_name, version
) -> list[dict]:
    rows = []
    for d, v, lo_val, hi_val in zip(dates, values, lo, hi, strict=False):
        rows.append(
            {
                "dataset_id": meta["dataset_id"],
                "business_name": meta["business_name"],
                "industry": meta["industry"],
                "domain": meta["domain"],
                "metric_name": meta["metric_name"],
                "forecast_date": pd.Timestamp(d).date(),
                "forecast_value": float(v),
                "lower_bound": float(lo_val),
                "upper_bound": float(hi_val),
                "model_name": model_name,
                "model_version": version,
            }
        )
    return rows


# Orchestrates the whole job: read features, forecast every series, write + register.
def run() -> int:
    """Run the forecasting job over the selected scope.

    Fits a model per domain metric, writes the forecasts, and registers the run.

    Returns:
        The number of forecast rows written.
    """
    dataset_id = target_dataset_id()
    scope = announce_mode(dataset_id)
    version = make_version("forecasting")

    with db_conn() as conn:
        df = read_daily_trend(conn, dataset_id)

        if df.empty:
            logger.warning(
                "No daily_trend rows for scope — nothing to forecast",
                extra={"scope": scope, "version": version},
            )
            register_model_version(
                conn,
                "forecasting",
                "statistical",
                version,
                scope,
                params={"horizon": HORIZON},
                row_count=0,
            )
            return 0

        all_rows: list[dict] = []
        series_done = 0
        series_skipped = 0

        group_cols = [
            "dataset_id",
            "business_name",
            "industry",
            "domain",
            "metric_name",
        ]
        for keys, group in df.groupby(group_cols, dropna=False):
            meta = dict(zip(group_cols, keys, strict=False))
            series = _prepare_series(group)
            result = _forecast_series(series, HORIZON)
            if result is None:
                series_skipped += 1
                continue
            dates, values, lo, hi, model_name = result
            all_rows.extend(
                _rows_from_forecast(meta, dates, values, lo, hi, model_name, version)
            )
            series_done += 1

        written = write_forecasts(conn, dataset_id, all_rows)
        register_model_version(
            conn,
            "forecasting",
            "statistical",
            version,
            scope,
            params={"horizon": HORIZON, "statsmodels": _HAS_SM},
            metrics={
                "series_forecasted": series_done,
                "series_skipped": series_skipped,
            },
            row_count=written,
        )

    logger.info(
        "Forecasting complete: %d rows written from %d series (%d skipped), "
        "version=%s",
        written,
        series_done,
        series_skipped,
        version,
        extra={
            "version": version,
            "series_forecasted": series_done,
            "series_skipped": series_skipped,
            "rows_written": written,
        },
    )
    return written


if __name__ == "__main__":
    configure_job_logging()
    run()
