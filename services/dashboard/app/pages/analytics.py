from __future__ import annotations

from typing import Any

import dash
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from app.api_client import (
    APIError,
    analytics_features,
    analytics_metrics,
    analytics_trend,
    list_datasets,
)
from app import theme

dash.register_page(__name__, path="/analytics", name="Analytics")


# Fixed chart type per domain
DOMAIN_ORDER = [
    "assets",
    "operations",
    "quality",
    "maintenance",
    "inventory",
    "workforce",
    "finance",
    "customers",
]

DOMAIN_CHART = {
    "assets": "line",
    "operations": "column",
    "quality": "bubble",
    "maintenance": "bar_trend",
    "inventory": "treemap",
    "workforce": "hbar",
    "finance": "area",
    "customers": "donut",
}


def layout() -> html.Div:
    return html.Div(
        style=theme.PAGE_STYLE,
        children=[
            html.H1("Analytics", style=theme.HEADING_STYLE),
            html.P(
                "Spark-computed analytics across the universal domains. "
                "Pick a dataset to explore its metrics, trends, and engineered features.",
                style=theme.SUBHEADING_STYLE,
            ),
            html.Label("Dataset", style=theme.LABEL_STYLE),
            dcc.Dropdown(
                id="analytics-dataset",
                options=[],
                placeholder="Select a dataset",
                clearable=False,
                style={"marginBottom": "1.5rem"},
            ),
            dcc.Interval(id="analytics-init", interval=300, max_intervals=1),
            html.Div(id="analytics-summary"),
            html.Div(id="analytics-metrics-chart"),
            html.Div(
                style={"marginTop": "1.5rem"},
                children=[
                    html.Label("Trend — pick a metric", style=theme.LABEL_STYLE),
                    dcc.Dropdown(
                        id="analytics-metric",
                        options=[],
                        placeholder="Select a metric",
                        clearable=False,
                        style={"marginBottom": "1rem"},
                    ),
                    html.Div(id="analytics-trend-chart"),
                ],
            ),
            html.Hr(
                style={
                    "margin": "2rem 0",
                    "border": "none",
                    "borderTop": f"1px solid {theme.COLORS['border']}",
                }
            ),
            html.H3("Engineered features (ML-ready)", style={"fontSize": "1.2rem"}),
            html.P(
                "Per-entity features computed by Spark — the foundation for forecasting "
                "and predictive models.",
                style=theme.SUBHEADING_STYLE,
            ),
            html.Div(id="analytics-features-table"),

            # ---- Domain dashboard (below the features table) ----
            html.Hr(
                style={
                    "margin": "2rem 0",
                    "border": "none",
                    "borderTop": f"1px solid {theme.COLORS['border']}",
                }
            ),
            html.H3("Domain dashboard", style={"fontSize": "1.2rem"}),
            html.P(
                "Each active domain visualized with a chart suited to its data. "
                "Only domains present in this dataset are shown.",
                style=theme.SUBHEADING_STYLE,
            ),
            html.Div(id="analytics-domain-status"),
            html.Div(id="analytics-domain-charts"),
        ],
    )


# ============================================================
# Populate the dataset dropdown
# ============================================================

@callback(
    Output("analytics-dataset", "options"),
    Output("analytics-dataset", "value"),
    Input("analytics-init", "n_intervals"),
)
def populate_datasets(_init: int | None) -> tuple[list[dict[str, Any]], Any]:
    try:
        datasets = list_datasets()
    except APIError:
        return [], None

    options = [
        {
            "label": f"{d['business_name']} ({d['source_filename']})",
            "value": d["dataset_id"],
        }
        for d in datasets
    ]
    default = options[0]["value"] if options else None
    return options, default


# ============================================================
# Load metrics + populate metric dropdown when dataset changes
# ============================================================

@callback(
    Output("analytics-summary", "children"),
    Output("analytics-metrics-chart", "children"),
    Output("analytics-metric", "options"),
    Output("analytics-metric", "value"),
    Input("analytics-dataset", "value"),
)
def load_metrics(
    dataset_id: int | None,
) -> tuple[Any, Any, list[dict[str, Any]], Any]:
    if dataset_id is None:
        return "", "", [], None

    try:
        metrics = analytics_metrics(dataset_id)
    except APIError as exc:
        return _error(f"Could not load analytics: {exc}"), "", [], None

    if not metrics:
        msg = html.Div(
            "No analytics computed for this dataset yet. "
            "Run the analytics pipeline in Airflow first.",
            style={"color": theme.COLORS["text_muted"]},
        )
        return msg, "", [], None

    summary = _render_summary(metrics)
    chart = _render_metrics_chart(metrics)

    metric_options = [
        {
            "label": f"{m['domain']}.{m['metric_name']}",
            "value": f"{m['domain']}|{m['metric_name']}",
        }
        for m in metrics
    ]
    default_metric = metric_options[0]["value"] if metric_options else None

    return summary, chart, metric_options, default_metric


def _render_summary(metrics: list[dict[str, Any]]) -> html.Div:
    first = metrics[0]
    domains = sorted({m["domain"] for m in metrics})
    return html.Div(
        style=theme.CARD_STYLE,
        children=[
            html.Span(
                first.get("business_name") or "Unknown",
                style={"fontWeight": 600, "fontSize": "1.1rem"},
            ),
            html.Span(
                f"  ·  {first.get('industry') or 'unspecified'}  ·  "
                f"{len(metrics)} metrics across {len(domains)} domains",
                style={"color": theme.COLORS["text_muted"]},
            ),
        ],
    )


def _render_metrics_chart(metrics: list[dict[str, Any]]) -> Any:
    names = [f"{m['domain']}.{m['metric_name']}" for m in metrics]
    avgs = [m.get("avg_value") or 0 for m in metrics]
    colors = [theme.domain_color(m["domain"]) for m in metrics]

    fig = go.Figure()
    fig.add_bar(x=names, y=avgs, marker_color=colors)
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=140),
        plot_bgcolor="white",
        title="Average value per metric",
        xaxis=dict(tickangle=-45),
    )
    return html.Div(dcc.Graph(figure=fig), style=theme.CARD_STYLE)


# ============================================================
# Trend chart when metric changes
# ============================================================

@callback(
    Output("analytics-trend-chart", "children"),
    Input("analytics-dataset", "value"),
    Input("analytics-metric", "value"),
)
def load_trend(dataset_id: int | None, metric_key: str | None) -> Any:
    if dataset_id is None or not metric_key:
        return ""

    domain, _, metric_name = metric_key.partition("|")

    try:
        points = analytics_trend(dataset_id, domain=domain, metric_name=metric_name)
    except APIError as exc:
        return _error(f"Could not load trend: {exc}")

    if not points:
        return html.Div(
            "No daily trend available for this metric "
            "(the data may have no timestamps).",
            style={"color": theme.COLORS["text_muted"]},
        )

    days = [p["day"] for p in points]
    avgs = [p.get("avg_value") or 0 for p in points]

    fig = go.Figure()
    fig.add_scatter(
        x=days,
        y=avgs,
        mode="lines+markers",
        line=dict(color=theme.domain_color(domain), width=2),
    )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=40, b=40),
        plot_bgcolor="white",
        title=f"Daily trend — {domain}.{metric_name}",
    )
    return html.Div(dcc.Graph(figure=fig), style=theme.CARD_STYLE)


# ============================================================
# Features table when dataset changes
# ============================================================

@callback(
    Output("analytics-features-table", "children"),
    Input("analytics-dataset", "value"),
)
def load_features(dataset_id: int | None) -> Any:
    if dataset_id is None:
        return ""

    try:
        features = analytics_features(dataset_id, limit=50)
    except APIError as exc:
        return _error(f"Could not load features: {exc}")

    if not features:
        return html.Div(
            "No engineered features yet.",
            style={"color": theme.COLORS["text_muted"]},
        )

    header = html.Tr(
        [
            html.Th(h, style=_TH_STYLE)
            for h in ["Domain", "Entity", "Metric", "Obs", "Avg", "Std", "Last", "Trend"]
        ]
    )

    rows = []
    for f in features:
        rows.append(
            html.Tr(
                [
                    html.Td(
                        f["domain"],
                        style={**_TD_STYLE, "color": theme.domain_color(f["domain"]),
                               "fontWeight": 500},
                    ),
                    html.Td(f["entity_ref"], style=_TD_STYLE),
                    html.Td(f["metric_name"], style=_TD_STYLE),
                    html.Td(_fmt(f.get("obs_count")), style=_TD_STYLE),
                    html.Td(_fmt(f.get("avg_value")), style=_TD_STYLE),
                    html.Td(_fmt(f.get("std_value")), style=_TD_STYLE),
                    html.Td(_fmt(f.get("last_value")), style=_TD_STYLE),
                    html.Td(_fmt(f.get("trend_slope")), style=_TD_STYLE),
                ]
            )
        )

    table = html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.85rem"},
    )
    return html.Div(table, style=theme.CARD_STYLE)


# ============================================================
# Domain status row + per-domain charts (below the table)
# ============================================================

@callback(
    Output("analytics-domain-status", "children"),
    Output("analytics-domain-charts", "children"),
    Input("analytics-dataset", "value"),
)
def load_domain_dashboard(dataset_id: int | None) -> tuple[Any, Any]:
    if dataset_id is None:
        return "", ""

    try:
        features = analytics_features(dataset_id, limit=500)
    except APIError:
        return "", ""

    if not features:
        return "", ""

    by_domain: dict[str, list[dict[str, Any]]] = {}
    for f in features:
        by_domain.setdefault(f["domain"], []).append(f)

    active = set(by_domain.keys())

    status_row = _render_domain_status(active)

    charts = []
    for domain in DOMAIN_ORDER:
        if domain not in active:
            continue
        chart = _render_domain_chart(dataset_id, domain, by_domain[domain])
        if chart is not None:
            charts.append(chart)

    return status_row, html.Div(charts)


def _render_domain_status(active: set[str]) -> html.Div:
    pills = []
    for domain in DOMAIN_ORDER:
        is_on = domain in active
        pills.append(
            html.Span(
                [
                    domain.capitalize(),
                    html.Span(
                        " ✓" if is_on else " —",
                        style={"fontWeight": 600},
                    ),
                ],
                style={
                    "fontSize": "0.8rem",
                    "padding": "0.25rem 0.7rem",
                    "borderRadius": "6px",
                    "marginRight": "0.5rem",
                    "marginBottom": "0.5rem",
                    "display": "inline-block",
                    "backgroundColor": (
                        _domain_tint(domain) if is_on else "#F1EFE8"
                    ),
                    "color": (
                        theme.domain_color(domain) if is_on else "#888780"
                    ),
                },
            )
        )

    return html.Div(
        style={"margin": "1rem 0 1.5rem"},
        children=[
            html.P(
                f"{len(active)} of 8 domains have data. "
                "Charts below show only the active domains.",
                style={**theme.SUBHEADING_STYLE, "marginBottom": "0.75rem"},
            ),
            html.Div(pills),
        ],
    )


def _render_domain_chart(
    dataset_id: int,
    domain: str,
    rows: list[dict[str, Any]],
) -> Any:
    chart_type = DOMAIN_CHART.get(domain, "column")
    color = theme.domain_color(domain)

    if chart_type == "line":
        fig = _chart_line(dataset_id, domain, rows, color)
    elif chart_type == "area":
        fig = _chart_area(dataset_id, domain, rows, color)
    elif chart_type == "bubble":
        fig = _chart_bubble(rows, color)
    elif chart_type == "bar_trend":
        fig = _chart_bar_trend(rows)
    elif chart_type == "hbar":
        fig = _chart_hbar(rows, color)
    elif chart_type == "treemap":
        fig = _chart_treemap(rows, domain)
    elif chart_type == "donut":
        fig = _chart_donut(rows, domain)
    else:
        fig = _chart_column(rows, color)

    if fig is None:
        return None

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=60),
        plot_bgcolor="white",
        title=f"{domain.capitalize()}",
    )
    return html.Div(dcc.Graph(figure=fig), style=theme.CARD_STYLE)


def _avg_by_entity(rows: list[dict[str, Any]]) -> tuple[list[str], list[float]]:
    ents = [r["entity_ref"] for r in rows]
    avgs = [float(r.get("avg_value") or 0) for r in rows]
    return ents, avgs


def _chart_column(rows: list[dict[str, Any]], color: str) -> go.Figure:
    ents, avgs = _avg_by_entity(rows)
    fig = go.Figure()
    fig.add_bar(x=ents, y=avgs, marker_color=color)
    fig.update_xaxes(tickangle=-45)
    return fig


def _chart_hbar(rows: list[dict[str, Any]], color: str) -> go.Figure:
    ents, avgs = _avg_by_entity(rows)
    fig = go.Figure()
    fig.add_bar(x=avgs, y=ents, orientation="h", marker_color=color)
    return fig


def _chart_bubble(rows: list[dict[str, Any]], color: str) -> go.Figure:
    ents = [r["entity_ref"] for r in rows]
    avgs = [float(r.get("avg_value") or 0) for r in rows]
    stds = [float(r.get("std_value") or 0) for r in rows]
    sizes = [max(8, s * 2) for s in stds]
    fig = go.Figure()
    fig.add_scatter(
        x=list(range(len(ents))),
        y=avgs,
        mode="markers",
        marker=dict(size=sizes, color=color, opacity=0.5),
        text=ents,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(ents))),
        ticktext=ents,
        tickangle=-45,
    )
    return fig


def _chart_bar_trend(rows: list[dict[str, Any]]) -> go.Figure:
    ents = [r["entity_ref"] for r in rows]
    avgs = [float(r.get("avg_value") or 0) for r in rows]
    trends = [float(r.get("trend_slope") or 0) for r in rows]
    colors = ["#E24B4A" if t > 0 else "#BA7517" for t in trends]
    fig = go.Figure()
    fig.add_bar(x=ents, y=avgs, marker_color=colors)
    fig.update_xaxes(tickangle=-45)
    return fig


def _chart_treemap(rows: list[dict[str, Any]], domain: str) -> go.Figure:
    ents = [r["entity_ref"] for r in rows]
    avgs = [abs(float(r.get("avg_value") or 0)) for r in rows]
    fig = go.Figure(
        go.Treemap(
            labels=ents,
            parents=[""] * len(ents),
            values=avgs,
            marker_colorscale="Blues",
        )
    )
    return fig


def _chart_donut(rows: list[dict[str, Any]], domain: str) -> go.Figure:
    ents = [r["entity_ref"] for r in rows]
    avgs = [abs(float(r.get("avg_value") or 0)) for r in rows]
    fig = go.Figure(go.Pie(labels=ents, values=avgs, hole=0.55))
    return fig


def _chart_line(
    dataset_id: int,
    domain: str,
    rows: list[dict[str, Any]],
    color: str,
) -> go.Figure:
    metric = rows[0]["metric_name"]
    try:
        points = analytics_trend(dataset_id, domain=domain, metric_name=metric)
    except APIError:
        points = []
    if points:
        days = [p["day"] for p in points]
        vals = [float(p.get("avg_value") or 0) for p in points]
        fig = go.Figure()
        fig.add_scatter(x=days, y=vals, mode="lines+markers",
                        line=dict(color=color, width=2))
        return fig
    return _chart_column(rows, color)


def _chart_area(
    dataset_id: int,
    domain: str,
    rows: list[dict[str, Any]],
    color: str,
) -> go.Figure:
    metric = rows[0]["metric_name"]
    try:
        points = analytics_trend(dataset_id, domain=domain, metric_name=metric)
    except APIError:
        points = []
    if points:
        days = [p["day"] for p in points]
        vals = [float(p.get("avg_value") or 0) for p in points]
        fig = go.Figure()
        fig.add_scatter(x=days, y=vals, mode="lines", fill="tozeroy",
                        line=dict(color=color, width=2))
        return fig
    return _chart_column(rows, color)


def _domain_tint(domain: str) -> str:
    tints = {
        "assets": "#E6F1FB",
        "operations": "#E6F1FB",
        "quality": "#E1F5EE",
        "maintenance": "#FAEEDA",
        "inventory": "#EEEDFE",
        "workforce": "#EEEDFE",
        "finance": "#E1F5EE",
        "customers": "#FBEAF0",
    }
    return tints.get(domain, "#F1EFE8")


# ============================================================
# Small helpers
# ============================================================

_TH_STYLE = {
    "textAlign": "left",
    "padding": "0.5rem 0.6rem",
    "borderBottom": f"2px solid {theme.COLORS['border']}",
    "color": theme.COLORS["text_muted"],
    "fontWeight": 600,
}

_TD_STYLE = {
    "padding": "0.4rem 0.6rem",
    "borderBottom": f"1px solid {theme.COLORS['border']}",
}


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _error(message: str) -> html.Div:
    return html.Div(
        message,
        style={
            "color": theme.COLORS["danger"],
            "backgroundColor": "#fef2f2",
            "border": f"1px solid {theme.COLORS['danger']}",
            "borderRadius": "8px",
            "padding": "0.6rem 0.9rem",
            "fontSize": "0.9rem",
        },
    )