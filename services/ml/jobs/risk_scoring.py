"""Unsupervised risk scoring for the Assets and Maintenance domains.

These two domains have no failure labels, so supervised prediction is not
honest here. Instead a 0-100 relative degradation risk is derived per entity
from three observable components -- adverse trend, volatility, and anomaly
severity -- and written to ``ml.risk_scores`` with the contributing factors
recorded alongside, so any score can be explained rather than merely asserted.
"""
# Unsupervised risk scoring — Assets + Maintenance "Future" without labels.
# Derives a 0–100 degradation risk per entity from trend, variability, and
# anomaly severity, then writes to ml.risk_scores.

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
    read_entity_features,
    register_model_version,
    target_dataset_id,
    write_risk_scores,
)

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Domains this job scores (the two blocked on labeling, now handled unsupervised).
TARGET_DOMAINS = {"assets", "maintenance"}

# For maintenance-type metrics a rising trend is bad; for asset-output metrics a
# falling trend is bad. This decides the sign of the trend contribution per domain.
HIGHER_IS_WORSE = {"maintenance"}

# Weights for the three risk components; kept explicit so the score is auditable.
W_TREND = 0.45
W_VOLATILITY = 0.25
W_ANOMALY = 0.30


# Scales a 1-D array to 0–1 via min-max; returns zeros when the range is degenerate.
def _minmax(values: np.ndarray) -> np.ndarray:
    v = values.astype(float)
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


# Directional trend contribution: normalized slope oriented so "worse" is always high.
def _trend_component(slopes: np.ndarray, domain: str) -> np.ndarray:
    directed = slopes if domain in HIGHER_IS_WORSE else -slopes
    directed = np.clip(directed, 0.0, None)  # only adverse movement adds risk
    return _minmax(directed)


# Volatility contribution: higher std relative to peers means less stable, more risk.
def _volatility_component(std_values: np.ndarray, avg_values: np.ndarray) -> np.ndarray:
    denom = np.where(np.abs(avg_values) < 1e-9, 1.0, np.abs(avg_values))
    cv = np.abs(std_values) / denom  # coefficient of variation
    return _minmax(cv)


# Anomaly contribution: per-entity weighted anomaly count normalized across peers.
def _anomaly_component(entity_ids, anomaly_weight: dict) -> np.ndarray:
    raw = np.array([anomaly_weight.get(e, 0.0) for e in entity_ids], dtype=float)
    return _minmax(raw)


# Builds a per-entity anomaly weight (high=3, medium=2, low=1) from ml.anomalies.
def _load_anomaly_weights(conn, dataset_id: int | None) -> dict:
    sql = """
        SELECT dataset_id, domain, entity_id, severity, COUNT(*) AS n
        FROM ml.anomalies
        WHERE entity_id IS NOT NULL
    """
    params: tuple = ()
    if dataset_id is not None:
        sql += " AND dataset_id = %s"
        params = (dataset_id,)
    sql += " GROUP BY dataset_id, domain, entity_id, severity"

    try:
        df = pd.read_sql(sql, conn, params=params)
    except Exception:
        logger.exception(
            "Could not read ml.anomalies (dataset_id=%s) — has the anomaly "
            "detection job run before this one?",
            dataset_id,
            extra={"table": "ml.anomalies", "dataset_id": dataset_id},
        )
        raise

    weight_map = {"high": 3.0, "medium": 2.0, "low": 1.0}
    out: dict = {}
    for _, r in df.iterrows():
        key = (r["dataset_id"], str(r["domain"]).lower(), r["entity_id"])
        out[key] = out.get(key, 0.0) + weight_map.get(r["severity"], 1.0) * float(
            r["n"]
        )
    return out


# Explains a single entity's score by capturing the raw drivers behind it.
def _factors(slope: float, cv: float, anomaly_w: float) -> dict:
    return {
        "trend_slope": round(float(slope), 4),
        "coeff_variation": round(float(cv), 4),
        "anomaly_weight": round(float(anomaly_w), 2),
    }


# Scores one domain's entities and returns ml.risk_scores row dicts.
def _score_domain(
    domain: str,
    grp: pd.DataFrame,
    anomaly_weights: dict,
    version: str,
) -> list[dict]:
    # Collapse to one row per entity by averaging its metrics, so an entity gets a
    # single composite risk rather than one per metric.
    agg = (
        grp.groupby(
            ["dataset_id", "business_name", "industry", "entity_id"], dropna=False
        )
        .agg(
            trend_slope=("trend_slope", "mean"),
            std_value=("std_value", "mean"),
            avg_value=("avg_value", "mean"),
        )
        .reset_index()
    )
    if agg.empty:
        return []

    slopes = agg["trend_slope"].fillna(0.0).values
    stds = agg["std_value"].fillna(0.0).values
    avgs = agg["avg_value"].fillna(0.0).values
    entity_ids = agg["entity_id"].values

    trend_c = _trend_component(slopes, domain)
    vol_c = _volatility_component(stds, avgs)

    ds_id = agg["dataset_id"].iloc[0]
    weight_lookup = [anomaly_weights.get((ds_id, domain, e), 0.0) for e in entity_ids]
    anomaly_c = _anomaly_component(
        entity_ids, {e: w for e, w in zip(entity_ids, weight_lookup, strict=False)}
    )

    score = (W_TREND * trend_c + W_VOLATILITY * vol_c + W_ANOMALY * anomaly_c) * 100.0

    rows: list[dict] = []
    for i in range(len(agg)):
        s = float(score[i])
        rows.append(
            {
                "dataset_id": int(agg["dataset_id"].iloc[i]),
                "business_name": agg["business_name"].iloc[i],
                "industry": agg["industry"].iloc[i],
                "domain": domain,
                "entity_id": entity_ids[i],
                "risk_score": round(s, 2),
                "risk_level": bucket_level(s),
                "contributing_factors": _factors(slopes[i], vol_c[i], weight_lookup[i]),
                "model_name": "unsupervised_risk",
                "model_version": version,
            }
        )
    return rows


# Orchestrates the job: read features, score target domains, write + register.
def run() -> int:
    """Run the risk scoring job over the selected scope.

    Scores the target domains' entities, writes the results, and registers the run.

    Returns:
        The number of risk-score rows written.
    """
    dataset_id = target_dataset_id()
    scope = announce_mode(dataset_id)
    version = make_version("risk_scoring")

    with db_conn() as conn:
        features = read_entity_features(conn, dataset_id)

        if features.empty:
            logger.warning(
                "No entity_features rows for scope — nothing to score",
                extra={"scope": scope, "version": version},
            )
            register_model_version(
                conn,
                "risk_scoring",
                "unsupervised",
                version,
                scope,
                params={"target_domains": sorted(TARGET_DOMAINS)},
                row_count=0,
            )
            return 0

        features["domain"] = features["domain"].astype(str).str.lower()
        anomaly_weights = _load_anomaly_weights(conn, dataset_id)

        all_rows: list[dict] = []
        domains_scored = 0
        for domain in sorted(TARGET_DOMAINS):
            grp = features[features["domain"] == domain]
            if grp.empty:
                continue
            all_rows.extend(_score_domain(domain, grp, anomaly_weights, version))
            domains_scored += 1

        written = write_risk_scores(conn, dataset_id, all_rows)
        high = sum(1 for r in all_rows if r["risk_level"] == "high")

        register_model_version(
            conn,
            "risk_scoring",
            "unsupervised",
            version,
            scope,
            params={
                "target_domains": sorted(TARGET_DOMAINS),
                "weights": {
                    "trend": W_TREND,
                    "volatility": W_VOLATILITY,
                    "anomaly": W_ANOMALY,
                },
            },
            metrics={"domains_scored": domains_scored, "high_risk_entities": high},
            row_count=written,
        )

    logger.info(
        "Risk scoring complete: %d entities scored across %d domain(s), "
        "%d high risk, version=%s",
        written,
        domains_scored,
        high,
        version,
        extra={
            "version": version,
            "domains_scored": domains_scored,
            "entities_scored": written,
            "high_risk": high,
        },
    )
    return written


if __name__ == "__main__":
    configure_job_logging()
    run()
