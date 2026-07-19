"""Anomaly detection job -- the 'Alerts' column of the intelligence view.

Flags unusual readings per entity and metric using an isolation forest, falling
back to a z-score when a group has too few points for the forest to be
meaningful, and writes the flagged readings to ``ml.anomalies`` with a severity
band.
"""
# Anomaly detection job — the "Alerts" column. Flags unusual readings per
# entity-metric using IsolationForest (with a z-score fallback) and writes to
# ml.anomalies.

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from ml_common import (
    announce_mode,
    bucket_level,
    configure_job_logging,
    db_conn,
    make_version,
    read_daily_trend,
    read_entity_features,
    register_model_version,
    target_dataset_id,
    write_anomalies,
)

logger = logging.getLogger(__name__)

# scikit-learn is optional: without it the job still runs, using the z-score
# fallback for every group. The reason it is unavailable is logged once at
# import so a silently degraded run is never a mystery later.
try:
    from sklearn.ensemble import IsolationForest

    _HAS_SKLEARN = True
except ImportError:
    logger.warning(
        "scikit-learn is not installed — falling back to z-score detection",
        exc_info=True,
    )
    _HAS_SKLEARN = False

warnings.filterwarnings("ignore")

# Minimum points needed per method, and z-score cutoff for the fallback path.
MIN_POINTS_IFOREST = 8
MIN_POINTS_ZSCORE = 4
ZSCORE_THRESHOLD = 2.5
IFOREST_CONTAMINATION = 0.1


# Maps an absolute z-score to a severity bucket for dashboard coloring.
def _zscore_severity(z: float) -> str:
    a = abs(z)
    if a >= 3.5:
        return "high"
    if a >= 3.0:
        return "medium"
    return "low"


# Maps an IsolationForest anomaly score (higher = more anomalous) to a severity bucket.
def _iforest_severity(score: float) -> str:
    return bucket_level(score * 100.0, low=40.0, high=70.0)


# ---------------------------------------------------------------------------
# Per-series detection over the daily_trend time series
# ---------------------------------------------------------------------------


# Builds an ordered daily avg_value series for one entity-metric group.
def _prepare_series(group: pd.DataFrame) -> pd.Series:
    s = (
        group.sort_values("trend_date")
        .set_index("trend_date")["avg_value"]
        .astype(float)
    )
    return s[~s.index.duplicated(keep="last")]


# IsolationForest over a single series' values; returns index positions flagged anomalous.
def _detect_iforest(values: np.ndarray):
    x = values.reshape(-1, 1)
    model = IsolationForest(
        contamination=IFOREST_CONTAMINATION,
        random_state=42,
        n_estimators=100,
    )
    preds = model.fit_predict(x)
    raw = -model.score_samples(x)  # higher = more anomalous
    flagged = np.where(preds == -1)[0]
    return flagged, raw


# Z-score detection over a single series; returns flagged positions and their z-values.
def _detect_zscore(values: np.ndarray):
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0:
        return np.array([], dtype=int), np.zeros_like(values), mean, std
    z = (values - mean) / std
    flagged = np.where(np.abs(z) >= ZSCORE_THRESHOLD)[0]
    return flagged, z, mean, std


# Runs the best available detector on one group and emits ml.anomalies row dicts.
def _rows_from_group(
    meta: dict, series: pd.Series, entity_id, version: str
) -> list[dict]:
    values = series.values.astype(float)
    dates = series.index
    rows: list[dict] = []

    if _HAS_SKLEARN and len(values) >= MIN_POINTS_IFOREST:
        try:
            flagged, raw = _detect_iforest(values)
        except Exception:
            # The forest can fail on degenerate input (constant or non-finite
            # values). Falling through to the z-score path below keeps the group
            # scored instead of failing the whole job for one bad series.
            logger.warning(
                "Isolation forest failed on a %d-point series for metric %s — "
                "falling back to z-score detection",
                len(values),
                meta.get("metric_name"),
                extra={
                    "series_length": len(values),
                    "metric_name": meta.get("metric_name"),
                    "domain": meta.get("domain"),
                },
                exc_info=True,
            )
        else:
            expected = float(np.median(values))
            for i in flagged:
                rows.append(
                    {
                        "dataset_id": meta["dataset_id"],
                        "business_name": meta["business_name"],
                        "industry": meta["industry"],
                        "domain": meta["domain"],
                        "entity_id": entity_id,
                        "metric_name": meta["metric_name"],
                        "anomaly_date": pd.Timestamp(dates[i]).date(),
                        "observed_value": float(values[i]),
                        "expected_value": expected,
                        "deviation": float(raw[i]),
                        "severity": _iforest_severity(float(raw[i])),
                        "method": "isolation_forest",
                        "model_version": version,
                    }
                )
            return rows

    if len(values) >= MIN_POINTS_ZSCORE:
        flagged, z, mean, _std = _detect_zscore(values)
        for i in flagged:
            rows.append(
                {
                    "dataset_id": meta["dataset_id"],
                    "business_name": meta["business_name"],
                    "industry": meta["industry"],
                    "domain": meta["domain"],
                    "entity_id": entity_id,
                    "metric_name": meta["metric_name"],
                    "anomaly_date": pd.Timestamp(dates[i]).date(),
                    "observed_value": float(values[i]),
                    "expected_value": mean,
                    "deviation": float(z[i]),
                    "severity": _zscore_severity(float(z[i])),
                    "method": "zscore",
                    "model_version": version,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Fallback detection over entity_features (when a group has no daily series)
# ---------------------------------------------------------------------------


# Flags entities whose last_value sits far from the metric's cross-entity mean.
def _rows_from_features(
    features: pd.DataFrame, seen_keys: set, version: str
) -> list[dict]:
    rows: list[dict] = []
    group_cols = ["dataset_id", "business_name", "industry", "domain", "metric_name"]

    for keys, grp in features.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, keys, strict=False))
        vals = grp["last_value"].astype(float).values
        if len(vals) < MIN_POINTS_ZSCORE:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if std == 0:
            continue
        for _, row in grp.iterrows():
            key = (
                meta["dataset_id"],
                meta["domain"],
                row["entity_id"],
                meta["metric_name"],
            )
            if key in seen_keys:
                continue
            z = (float(row["last_value"]) - mean) / std
            if abs(z) < ZSCORE_THRESHOLD:
                continue
            rows.append(
                {
                    "dataset_id": meta["dataset_id"],
                    "business_name": meta["business_name"],
                    "industry": meta["industry"],
                    "domain": meta["domain"],
                    "entity_id": row["entity_id"],
                    "metric_name": meta["metric_name"],
                    "anomaly_date": None,
                    "observed_value": float(row["last_value"]),
                    "expected_value": mean,
                    "deviation": float(z),
                    "severity": _zscore_severity(float(z)),
                    "method": "zscore",
                    "model_version": version,
                }
            )
    return rows


# Orchestrates the job: time-series detection first, feature-level fallback second, then write.
def run() -> int:
    """Run the anomaly detection job over the selected scope.

    Scores each entity-metric group, writes the flagged anomalies, and registers
    the run.

    Returns:
        The number of anomaly rows written.
    """
    dataset_id = target_dataset_id()
    scope = announce_mode(dataset_id)
    version = make_version("anomaly_detection")

    with db_conn() as conn:
        trend = read_daily_trend(conn, dataset_id)
        features = read_entity_features(conn, dataset_id)

        all_rows: list[dict] = []
        series_scanned = 0

        # entity_id lives in entity_features, not daily_trend, so join it in by the
        # shared keys to attribute each time series to its entity where possible.
        if not trend.empty:
            ef_keys = ["dataset_id", "domain", "metric_name"]
            entity_lookup = (
                features.groupby(ef_keys)["entity_id"].first().to_dict()
                if not features.empty
                else {}
            )
            group_cols = [
                "dataset_id",
                "business_name",
                "industry",
                "domain",
                "metric_name",
            ]
            for keys, group in trend.groupby(group_cols, dropna=False):
                meta = dict(zip(group_cols, keys, strict=False))
                series = _prepare_series(group)
                if series.empty:
                    continue
                entity_id = entity_lookup.get(
                    (meta["dataset_id"], meta["domain"], meta["metric_name"])
                )
                all_rows.extend(_rows_from_group(meta, series, entity_id, version))
                series_scanned += 1

        # Feature-level pass covers entity-metrics that had no usable daily series.
        seen = {
            (r["dataset_id"], r["domain"], r["entity_id"], r["metric_name"])
            for r in all_rows
        }
        if not features.empty:
            all_rows.extend(_rows_from_features(features, seen, version))

        written = write_anomalies(conn, dataset_id, all_rows)

        high = sum(1 for r in all_rows if r["severity"] == "high")
        register_model_version(
            conn,
            "anomaly_detection",
            "unsupervised",
            version,
            scope,
            params={
                "sklearn": _HAS_SKLEARN,
                "contamination": IFOREST_CONTAMINATION,
                "zscore_threshold": ZSCORE_THRESHOLD,
            },
            metrics={"series_scanned": series_scanned, "high_severity": high},
            row_count=written,
        )

    logger.info(
        "Anomaly detection complete: %d anomalies written (%d high severity) "
        "from %d series scanned, version=%s",
        written,
        high,
        series_scanned,
        version,
        extra={
            "version": version,
            "series_scanned": series_scanned,
            "anomalies_written": written,
            "high_severity": high,
        },
    )
    return written


if __name__ == "__main__":
    configure_job_logging()
    run()
