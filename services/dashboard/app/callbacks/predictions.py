"""Callbacks for /predictions.

Three models run per dataset, domain-agnostically: they work on a telecom
tower's signal strength exactly as they work on a bottling line's output,
because they only ever see the universal schema.
"""

from __future__ import annotations

from typing import Any

from dash import Input, Output, State, callback, html

from app import feedback, ids
from app.api_client import (
    APIError,
    analytics_metrics,
    list_datasets,
    ml_anomalies,
    ml_forecasts,
    ml_overview,
    ml_risk_scores,
)
from app.charts import prediction_charts as pc
from app.components import ui
from app.constants import DOMAIN_ORDER
from app.utils import fmt, group


@callback(
    Output(ids.PRED_DATASET, "options"),
    Output(ids.PRED_DATASET, "value"),
    Input(ids.PRED_INIT, "n_intervals"),
    State(ids.ACCESS_TOKEN, "data"),
)
def populate_datasets(
    _init: int | None, token: str | None
) -> tuple[list[dict[str, Any]], Any]:
    """Fill the dataset selector and preselect the first entry.

    Args:
        _init: Interval tick that triggers the initial load.
        token: Caller's access token.

    Returns:
        The selector options and the initially selected dataset.
    """
    try:
        datasets = list_datasets(token=token)
    except APIError:
        return [], None

    options = [
        {"label": d["business_name"], "value": d["dataset_id"]} for d in datasets
    ]
    return options, (options[0]["value"] if options else None)


@callback(
    Output(ids.PRED_KPIS, "children"),
    Output(ids.PRED_DOMAIN_CHARTS, "children"),
    Output(ids.PRED_TABLE, "children"),
    Input(ids.PRED_DATASET, "value"),
    State(ids.ACCESS_TOKEN, "data"),
)
def load_predictions(dataset_id: int | None, token: str | None) -> tuple[Any, Any, Any]:
    """Load and render the forecasts, alerts, and risk scores for a dataset.

    Args:
        dataset_id: Selected dataset.
        token: Caller's access token.

    Returns:
        The rendered prediction panels, or a message when unavailable.
    """
    if dataset_id is None:
        return "", "", ""

    try:
        overview = ml_overview(dataset_id, token=token)
        forecasts = ml_forecasts(dataset_id, token=token)
        anomalies = ml_anomalies(dataset_id, limit=2000, token=token)
        risks = ml_risk_scores(dataset_id, token=token)
    except APIError as exc:
        return feedback.error(f"Could not load ML results: {exc}"), "", ""

    # Analytics metrics are supporting context, not the point of this page.
    # Their absence degrades the table but must not blank the whole page.
    try:
        metrics = analytics_metrics(dataset_id, token=token)
    except APIError:
        metrics = []

    if not forecasts and not anomalies and not risks:
        return (
            feedback.empty(
                "No predictions for this dataset yet. Run the ML orchestration DAG "
                "in Airflow \u2014 it is triggered manually, not on upload."
            ),
            "",
            "",
        )

    by_domain = {
        "forecasts": group(forecasts, "domain"),
        "anomalies": group(anomalies, "domain"),
        "risks": group(risks, "domain"),
    }
    active = (
        set(by_domain["forecasts"])
        | set(by_domain["anomalies"])
        | set(by_domain["risks"])
    )

    charts = []
    for domain in DOMAIN_ORDER:
        if domain not in active:
            continue
        figure = pc.build(
            pc.DOMAIN_CHART.get(domain, "forecast_line"),
            by_domain["forecasts"].get(domain, []),
            by_domain["anomalies"].get(domain, []),
            by_domain["risks"].get(domain, []),
            ui.tokens.domain_ink(domain) if hasattr(ui, "tokens") else _ink(domain),
        )
        if figure is None:
            continue
        charts.append(_domain_card(domain, figure))

    return (
        _kpis(overview, active),
        ui.grid(*charts, cols=2),
        _table(metrics, forecasts, anomalies, risks, active),
    )


# ============================================================
# Render helpers
# ============================================================


def _ink(domain: str) -> str:
    from app.design import tokens

    return tokens.domain_ink(domain)


def _domain_card(domain: str, figure: Any) -> html.Div:
    return ui.card(
        html.Div(
            [
                ui.domain_tile(domain),
                html.Div(
                    [
                        html.Div(domain.capitalize(), className="card-title"),
                        html.Div(
                            pc.CHART_SUBTITLE.get(pc.DOMAIN_CHART.get(domain, ""), ""),
                            className="kpi-note",
                        ),
                    ]
                ),
            ],
            style={
                "display": "flex",
                "gap": "0.75rem",
                "alignItems": "center",
                "marginBottom": "0.75rem",
            },
        ),
        _graph(figure),
    )


def _graph(figure: Any) -> Any:
    from dash import dcc

    return dcc.Graph(
        figure=figure,
        config={
            "displaylogo": False,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
            "responsive": True,
        },
    )


def _kpis(overview: dict[str, Any], active: set[str]) -> html.Div:
    return html.Div(
        ui.grid(
            ui.kpi(
                "Forecasts",
                f"{overview.get('forecast_count', 0):,}",
                f"across {len(active)} domains",
            ),
            ui.kpi(
                "Anomalies",
                f"{overview.get('anomaly_count', 0):,}",
                "readings outside expectation",
                tone="danger",
            ),
            ui.kpi(
                "High-risk entities",
                overview.get("high_risk_count", 0),
                "ranked, not predicted",
                tone="warn",
            ),
            ui.kpi("Models", 3, "forecast \u00b7 anomaly \u00b7 risk"),
            cols=4,
        ),
        style={"marginBottom": "1.5rem"},
    )


def _table(
    metrics: list[dict[str, Any]],
    forecasts: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    risks: list[dict[str, Any]],
    active: set[str],
) -> Any:
    records = _records(metrics, forecasts, anomalies, risks, active)
    if not records:
        return ""

    rows = []
    for r in records:
        rows.append(
            [
                ui.domain_chip(r["domain"]),
                r["metric"],
                fmt(r["current"]),
                fmt(r["forecast"]),
                _change(r["change"]),
                _anomaly_count(r["anomalies"]),
                _risk(r["risk"]),
            ]
        )

    return ui.section(
        "Every metric, side by side",
        ui.card(
            ui.table(
                [
                    "Domain",
                    "Metric",
                    "Current",
                    "Forecast (7d avg)",
                    "Change",
                    "Anomalies",
                    "Risk",
                ],
                rows,
                note=(
                    "Risk is a relative degradation ranking, not a probability of "
                    "failure. No failure labels exist in this data, so none are "
                    "invented. Fully-populated rows are shown first."
                ),
            )
        ),
    )


def _records(
    metrics: list[dict[str, Any]],
    forecasts: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    risks: list[dict[str, Any]],
    active: set[str],
) -> list[dict[str, Any]]:
    """Join four ML outputs into one row per (domain, metric).

    Rows where every column carries a real value sort to the top. A partial row
    is still shown -- an absent forecast is information, not something to hide.
    """
    current = {
        (str(m["domain"]).lower(), m["metric_name"]): m.get("avg_value")
        for m in metrics
    }

    forecast_avg: dict[tuple[str, str], float] = {}
    forecast_change: dict[tuple[str, str], float] = {}
    for domain, rows in group(forecasts, "domain").items():
        by_metric: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_metric.setdefault(row["metric_name"], []).append(row)

        for metric, series in by_metric.items():
            series = sorted(series, key=lambda r: r["forecast_date"])
            values = [
                r["forecast_value"]
                for r in series
                if r.get("forecast_value") is not None
            ]
            if values:
                forecast_avg[(domain, metric)] = sum(values) / len(values)
                forecast_change[(domain, metric)] = values[-1] - values[0]

    anomaly_count: dict[tuple[str, str], int] = {}
    for a in anomalies:
        key = (str(a["domain"]).lower(), a["metric_name"])
        anomaly_count[key] = anomaly_count.get(key, 0) + 1

    worst_risk: dict[str, dict[str, Any]] = {}
    for r in risks:
        domain = str(r["domain"]).lower()
        best = worst_risk.get(domain)
        if best is None or (r.get("risk_score") or 0) > (best.get("risk_score") or 0):
            worst_risk[domain] = r

    keys = {
        k
        for k in set(current) | set(forecast_avg) | set(anomaly_count)
        if k[0] in active
    }

    records = []
    for domain, metric in keys:
        cur = current.get((domain, metric))
        fcast = forecast_avg.get((domain, metric))
        change = forecast_change.get((domain, metric))
        risk = worst_risk.get(domain)
        records.append(
            {
                "domain": domain,
                "metric": metric,
                "current": cur,
                "forecast": fcast,
                "change": change,
                "anomalies": anomaly_count.get((domain, metric), 0),
                "risk": risk,
                "filled": all(v is not None for v in (cur, fcast, change))
                and risk is not None,
            }
        )

    records.sort(key=lambda r: (not r["filled"], r["domain"], r["metric"]))
    return records


def _change(change: float | None) -> Any:
    if change is None:
        return html.Span("\u2014", className="trend-flat")

    sign = "+" if change >= 0 else "\u2212"
    # A change is neither good nor bad without knowing the metric. Rising
    # downtime and rising output are both "up". Colour is direction only.
    tone = "trend-up" if change >= 0 else "trend-down"
    return html.Span(f"{sign}{abs(change):,.2f}", className=tone)


def _anomaly_count(count: int) -> Any:
    if count == 0:
        return html.Span("0", className="trend-flat")
    return html.Span(str(count), className="trend-up")


def _risk(risk: dict[str, Any] | None) -> Any:
    if not risk:
        return html.Span("\u2014", className="trend-flat")

    return html.Span(
        [
            html.Span(fmt(risk.get("risk_score"), 0)),
            " ",
            ui.badge(risk.get("risk_level")),
        ]
    )
