"""Callbacks for /analytics."""

from __future__ import annotations

from typing import Any

from dash import Input, Output, State, callback, html

from app import feedback, ids
from app.api_client import (
    APIError,
    analytics_features,
    analytics_metrics,
    analytics_trend,
    list_datasets,
)
from app.charts import domain_charts
from app.components import ui
from app.constants import DOMAIN_ORDER
from app.utils import fmt, group_by_domain


@callback(
    Output(ids.ANALYTICS_DATASET, "options"),
    Output(ids.ANALYTICS_DATASET, "value"),
    Input(ids.ANALYTICS_INIT, "n_intervals"),
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
    Output(ids.ANALYTICS_SUMMARY, "children"),
    Output(ids.ANALYTICS_METRICS_CHART, "children"),
    Output(ids.ANALYTICS_METRIC, "options"),
    Output(ids.ANALYTICS_METRIC, "value"),
    Input(ids.ANALYTICS_DATASET, "value"),
    State(ids.ACCESS_TOKEN, "data"),
)
def load_metrics(
    dataset_id: int | None,
    token: str | None,
) -> tuple[Any, Any, list[dict[str, Any]], Any]:
    """Load and render the selected dataset's domain metrics.

    Args:
        dataset_id: Selected dataset.
        token: Caller's access token.

    Returns:
        The rendered metrics, or a message when unavailable.
    """
    if dataset_id is None:
        return "", "", [], None

    try:
        metrics = analytics_metrics(dataset_id, token=token)
    except APIError as exc:
        return feedback.error(f"Could not load analytics: {exc}"), "", [], None

    if not metrics:
        return (
            feedback.empty(
                "No analytics for this dataset yet. Confirm a mapping, or run the "
                "analytics DAG in Airflow."
            ),
            "",
            [],
            None,
        )

    options = [
        {
            "label": f"{m['domain']} \u00b7 {m['metric_name']}",
            "value": f"{m['domain']}|{m['metric_name']}",
        }
        for m in metrics
    ]

    return (
        _summary(metrics),
        ui.chart(domain_charts.chart_metric_averages(metrics)),
        options,
        options[0]["value"] if options else None,
    )


@callback(
    Output(ids.ANALYTICS_TREND_CHART, "children"),
    Input(ids.ANALYTICS_DATASET, "value"),
    Input(ids.ANALYTICS_METRIC, "value"),
    State(ids.ACCESS_TOKEN, "data"),
)
def load_trend(
    dataset_id: int | None, metric_key: str | None, token: str | None
) -> Any:
    """Load and render the daily trend for the selected metric.

    Args:
        dataset_id: Selected dataset.
        metric_key: Selected domain and metric.
        token: Caller's access token.

    Returns:
        The rendered trend chart, or a message when unavailable.
    """
    if dataset_id is None or not metric_key:
        return ""

    domain, _, metric_name = metric_key.partition("|")

    try:
        points = analytics_trend(
            dataset_id, domain=domain, metric_name=metric_name, token=token
        )
    except APIError as exc:
        return feedback.error(f"Could not load the trend: {exc}")

    if not points:
        return feedback.empty(
            "No daily trend for this metric \u2014 the source data has no "
            "timestamps to plot against."
        )

    return ui.chart(domain_charts.chart_daily_trend(points, domain, metric_name))


@callback(
    Output(ids.ANALYTICS_FEATURES_TABLE, "children"),
    Input(ids.ANALYTICS_DATASET, "value"),
    State(ids.ACCESS_TOKEN, "data"),
)
def load_features(dataset_id: int | None, token: str | None) -> Any:
    """Load and render the selected dataset's entity features.

    Args:
        dataset_id: Selected dataset.
        token: Caller's access token.

    Returns:
        The rendered features, or a message when unavailable.
    """
    if dataset_id is None:
        return ""

    try:
        features = analytics_features(dataset_id, limit=50, token=token)
    except APIError as exc:
        return feedback.error(f"Could not load features: {exc}")

    if not features:
        return feedback.empty("No engineered features yet.")

    rows = [
        [
            ui.domain_chip(f["domain"]),
            f["entity_ref"],
            f["metric_name"],
            fmt(f.get("obs_count"), 0),
            fmt(f.get("avg_value")),
            fmt(f.get("std_value")),
            fmt(f.get("last_value")),
            html.Span(
                [
                    fmt(f.get("trend_slope"), 3),
                    " ",
                    # The slope's sign is the fact; whether it is good news is a
                    # question about the domain, and this table does not answer it.
                    ui.trend(f.get("trend_slope")),
                ]
            ),
        ]
        for f in features
    ]

    return ui.card(
        ui.table(
            ["Domain", "Entity", "Metric", "Obs", "Avg", "Std", "Last", "Trend"],
            rows,
        )
    )


@callback(
    Output(ids.ANALYTICS_DOMAIN_STATUS, "children"),
    Output(ids.ANALYTICS_DOMAIN_CHARTS, "children"),
    Input(ids.ANALYTICS_DATASET, "value"),
    State(ids.ACCESS_TOKEN, "data"),
)
def load_domain_dashboard(dataset_id: int | None, token: str | None) -> tuple[Any, Any]:
    """Load and render the per-domain chart panels for a dataset.

    Args:
        dataset_id: Selected dataset.
        token: Caller's access token.

    Returns:
        The rendered domain panels, or a message when unavailable.
    """
    if dataset_id is None:
        return "", ""

    try:
        features = analytics_features(dataset_id, limit=500, token=token)
    except APIError as exc:
        return feedback.error(f"Could not load domain data: {exc}"), ""

    if not features:
        return feedback.empty("No engineered features yet."), ""

    by_domain = group_by_domain(features)
    active = set(by_domain)

    charts = []
    for domain in DOMAIN_ORDER:
        if domain not in active:
            continue
        figure = domain_charts.build(dataset_id, domain, by_domain[domain])
        if figure is None:
            continue
        figure.update_layout(height=300, title=domain.capitalize())
        charts.append(ui.chart(figure))

    return _domain_status(active), ui.grid(*charts, cols=2)


# ============================================================
# Render helpers
# ============================================================


def _summary(metrics: list[dict[str, Any]]) -> Any:
    first = metrics[0]
    domains = sorted({m["domain"] for m in metrics})
    return ui.card(
        html.Span(first.get("business_name") or "Unknown", style={"fontWeight": 600}),
        html.Span(
            f"  \u00b7  {first.get('industry') or 'industry unspecified'}"
            f"  \u00b7  {len(metrics)} metrics across {len(domains)} domains",
            className="kpi-note",
        ),
    )


def _domain_status(active: set[str]) -> html.Div:
    """All eight domains, always. An absent one is shown, not hidden.

    A dataset covering four of eight is visibly half-blind, and that is the
    point: it is what makes a later "no data for that question" explainable
    rather than mysterious.
    """
    chips = [
        ui.domain_chip(domain, present=domain in active) for domain in DOMAIN_ORDER
    ]
    return html.Div(
        [
            html.P(
                f"{len(active)} of 8 domains carry data in this dataset.",
                className="page-subtitle",
            ),
            html.Div(
                chips,
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "0.5rem",
                    "marginBottom": "1.5rem",
                },
            ),
        ]
    )
