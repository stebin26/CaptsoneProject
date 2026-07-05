# Forecasting job — the "Future" column. Reads analytics.daily_trend, fits a
# per domain-metric time-series model, and writes future values to ml.forecasts.

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

from ml_common import (
    announce_mode,
    db_conn,
    make_version,
    read_daily_trend,
    register_model_version,
    target_dataset_id,
    write_forecasts,
)

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HAS_SM = True
except Exception:
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
    dates = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
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
    dates = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    return dates, fy, fy - margin, fy + margin, "linear_trend"


# Picks the best available method for a series given its length; None if too short.
def _forecast_series(series: pd.Series, horizon: int):
    if _HAS_SM and len(series) >= MIN_POINTS_SMOOTHING:
        try:
            return _forecast_smoothing(series, horizon)
        except Exception:
            pass
    if len(series) >= MIN_POINTS_LINEAR:
        return _forecast_linear(series, horizon)
    return None


# Turns one series' forecast arrays into ml.forecasts row dicts.
def _rows_from_forecast(meta: dict, dates, values, lo, hi, model_name, version) -> list[dict]:
    rows = []
    for d, v, l, h in zip(dates, values, lo, hi):
        rows.append(
            {
                "dataset_id": meta["dataset_id"],
                "business_name": meta["business_name"],
                "industry": meta["industry"],
                "domain": meta["domain"],
                "metric_name": meta["metric_name"],
                "forecast_date": pd.Timestamp(d).date(),
                "forecast_value": float(v),
                "lower_bound": float(l),
                "upper_bound": float(h),
                "model_name": model_name,
                "model_version": version,
            }
        )
    return rows


# Orchestrates the whole job: read features, forecast every series, write + register.
def run() -> int:
    dataset_id = target_dataset_id(sys.argv)
    scope = announce_mode(dataset_id)
    version = make_version("forecasting")

    with db_conn() as conn:
        df = read_daily_trend(conn, dataset_id)

        if df.empty:
            print("no daily_trend rows for scope — nothing to forecast")
            register_model_version(
                conn, "forecasting", "statistical", version, scope,
                params={"horizon": HORIZON}, row_count=0,
            )
            return 0

        all_rows: list[dict] = []
        series_done = 0
        series_skipped = 0

        group_cols = ["dataset_id", "business_name", "industry", "domain", "metric_name"]
        for keys, group in df.groupby(group_cols, dropna=False):
            meta = dict(zip(group_cols, keys))
            series = _prepare_series(group)
            result = _forecast_series(series, HORIZON)
            if result is None:
                series_skipped += 1
                continue
            dates, values, lo, hi, model_name = result
            all_rows.extend(_rows_from_forecast(meta, dates, values, lo, hi, model_name, version))
            series_done += 1

        written = write_forecasts(conn, dataset_id, all_rows)
        register_model_version(
            conn, "forecasting", "statistical", version, scope,
            params={"horizon": HORIZON, "statsmodels": _HAS_SM},
            metrics={"series_forecasted": series_done, "series_skipped": series_skipped},
            row_count=written,
        )

    print("=" * 40)
    print(f"[forecasting] version:          {version}")
    print(f"[forecasting] series forecast:  {series_done}")
    print(f"[forecasting] series skipped:   {series_skipped}")
    print(f"[forecasting] rows written:     {written}")
    print("=" * 40)
    return written


if __name__ == "__main__":
    run()