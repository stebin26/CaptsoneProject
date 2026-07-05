from __future__ import annotations

from typing import Any

import dash
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from app.api_client import (
    APIError,
    analytics_metrics,
    list_datasets,
    ml_anomalies,
    ml_forecasts,
    ml_overview,
    ml_risk_scores,
)
from app import theme

dash.register_page(__name__, path="/predictions", name="Predictions")


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

# Each domain maps to the chart best suited to its dominant ML output type.
DOMAIN_CHART = {
    "assets": "risk_hbar",
    "operations": "forecast_line",
    "quality": "anomaly_scatter",
    "maintenance": "risk_vbar",
    "inventory": "forecast_area",
    "workforce": "forecast_line",
    "finance": "forecast_area",
    "customers": "forecast_area",
}

_RISK_COLOR = {"high": "#dc2626", "medium": "#d97706", "low": "#16a34a"}
_SEV_COLOR = {"high": "#dc2626", "medium": "#d97706", "low": "#9ca3af"}


def layout() -> html.Div:
    return html.Div(
        style=theme.PAGE_STYLE,
        children=[
            html.H1("Predictions", style=theme.HEADING_STYLE),
            html.P(
                "Machine-learning intelligence per domain — current state, forecasted "
                "future, and alerts. Only domains present in this dataset are shown.",
                style=theme.SUBHEADING_STYLE,
            ),
            html.Label("Dataset", style=theme.LABEL_STYLE),
            dcc.Dropdown(
                id="pred-dataset",
                options=[],
                placeholder="Select a dataset",
                clearable=False,
                style={"marginBottom": "1.5rem"},
            ),
            dcc.Interval(id="pred-init", interval=300, max_intervals=1),
            html.Div(id="pred-domain-charts"),
            html.Div(id="pred-kpis"),
            html.Div(id="pred-table"),
        ],
    )


# ============================================================
# Dataset dropdown
# ============================================================

@callback(
    Output("pred-dataset", "options"),
    Output("pred-dataset", "value"),
    Input("pred-init", "n_intervals"),
)
def populate_datasets(_init: int | None) -> tuple[list[dict[str, Any]], Any]:
    try:
        datasets = list_datasets()
    except APIError:
        return [], None
    options = [
        {"label": f"{d['business_name']} ({d['source_filename']})", "value": d["dataset_id"]}
        for d in datasets
    ]
    return options, (options[0]["value"] if options else None)


# ============================================================
# Main load — charts, KPIs, table
# ============================================================

@callback(
    Output("pred-domain-charts", "children"),
    Output("pred-kpis", "children"),
    Output("pred-table", "children"),
    Input("pred-dataset", "value"),
)
def load_predictions(dataset_id: int | None) -> tuple[Any, Any, Any]:
    if dataset_id is None:
        return "", "", ""

    try:
        overview = ml_overview(dataset_id)
        forecasts = ml_forecasts(dataset_id)
        anomalies = ml_anomalies(dataset_id, limit=2000)
        risks = ml_risk_scores(dataset_id)
    except APIError as exc:
        return _error(f"Could not load ML results: {exc}"), "", ""

    try:
        metrics = analytics_metrics(dataset_id)
    except APIError:
        metrics = []

    if not forecasts and not anomalies and not risks:
        msg = html.Div(
            "No ML results for this dataset yet. Confirm an upload (the ML pipeline "
            "runs automatically) or trigger the ML orchestration DAG in Airflow.",
            style={"color": theme.COLORS["text_muted"]},
        )
        return msg, "", ""

    fc_by_domain = _group(forecasts, "domain")
    an_by_domain = _group(anomalies, "domain")
    rk_by_domain = _group(risks, "domain")
    active = set(fc_by_domain) | set(an_by_domain) | set(rk_by_domain)

    cards = []
    for i, domain in enumerate([d for d in DOMAIN_ORDER if d in active]):
        cards.append(
            _domain_card(
                i + 1,
                domain,
                fc_by_domain.get(domain, []),
                an_by_domain.get(domain, []),
                rk_by_domain.get(domain, []),
            )
        )
    charts_grid = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(2, 1fr)",
            "gap": "1rem",
            "marginBottom": "1.5rem",
        },
        children=cards,
    )

    kpis = _kpi_row(overview, active)
    table = _unified_table(metrics, forecasts, anomalies, risks, active)

    return charts_grid, kpis, table


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(str(r.get(key)).lower(), []).append(r)
    return out


# ============================================================
# Domain card (header + chart)
# ============================================================

def _domain_card(
    index: int,
    domain: str,
    forecasts: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    risks: list[dict[str, Any]],
) -> html.Div:
    color = theme.domain_color(domain)
    chart_type = DOMAIN_CHART.get(domain, "forecast_line")
    fig = _build_chart(chart_type, forecasts, anomalies, risks, color)
    subtitle = {
        "risk_hbar": "Risk Bar Chart",
        "risk_vbar": "Risk Bar Chart (by equipment)",
        "forecast_line": "Forecast Line + Confidence Band",
        "forecast_area": "Forecast Area Chart",
        "anomaly_scatter": "Anomaly Scatter (Observed vs Expected)",
    }.get(chart_type, "")

    header = html.Div(
        style={"display": "flex", "alignItems": "center", "marginBottom": "0.5rem"},
        children=[
            html.Span(
                str(index),
                style={
                    "backgroundColor": color, "color": "#fff", "borderRadius": "6px",
                    "width": "22px", "height": "22px", "display": "inline-flex",
                    "alignItems": "center", "justifyContent": "center",
                    "fontSize": "0.75rem", "fontWeight": 700, "marginRight": "0.5rem",
                },
            ),
            html.Div([
                html.Div(domain.capitalize(),
                         style={"fontWeight": 700, "fontSize": "1rem", "color": color}),
                html.Div(subtitle,
                         style={"fontSize": "0.72rem", "color": theme.COLORS["text_muted"]}),
            ]),
        ],
    )

    body = (
        dcc.Graph(
            figure=fig,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )
        if fig is not None
        else html.Div("No chartable data.",
                      style={"fontSize": "0.82rem", "color": theme.COLORS["text_muted"]})
    )
    return html.Div(style={**theme.CARD_STYLE, "marginBottom": 0}, children=[header, body])


# ============================================================
# Chart builders
# ============================================================

def _build_chart(chart_type, forecasts, anomalies, risks, color):
    if chart_type == "risk_hbar":
        fig = _chart_risk_bar(risks, horizontal=True)
        if fig is None:
            fig = _chart_forecast(forecasts, color, fill=False)
    elif chart_type == "risk_vbar":
        fig = _chart_risk_bar(risks, horizontal=False)
        if fig is None:
            fig = _chart_forecast(forecasts, color, fill=False)
    elif chart_type == "anomaly_scatter":
        fig = _chart_anomaly_scatter(anomalies)
        if fig is None:
            fig = _chart_forecast(forecasts, color, fill=False)
    elif chart_type == "forecast_area":
        fig = _chart_forecast(forecasts, color, fill=True)
        if fig is None:
            fig = _chart_risk_bar(risks, horizontal=True)
    else:  # forecast_line
        fig = _chart_forecast(forecasts, color, fill=False)
        if fig is None:
            fig = _chart_risk_bar(risks, horizontal=True)

    if fig is not None:
        fig.update_layout(
            height=260,
            margin=dict(l=30, r=20, t=20, b=40),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=True,
            legend=dict(orientation="h", y=-0.2, font=dict(size=9)),
        )
    return fig


def _chart_forecast(forecasts, color, fill: bool):
    if not forecasts:
        return None
    metric = sorted({f["metric_name"] for f in forecasts})[0]
    series = sorted([f for f in forecasts if f["metric_name"] == metric],
                    key=lambda f: f["forecast_date"])
    dates = [f["forecast_date"] for f in series]
    vals = [f.get("forecast_value") for f in series]
    lo = [f.get("lower_bound") for f in series]
    hi = [f.get("upper_bound") for f in series]

    fig = go.Figure()
    if all(x is not None for x in hi) and all(x is not None for x in lo):
        fig.add_scatter(x=dates + dates[::-1], y=hi + lo[::-1], fill="toself",
                        fillcolor=_rgba(color, 0.12), line=dict(width=0),
                        hoverinfo="skip", name="Confidence Band")
    line_color = "#1f2937" if fill else color   # area charts: black line, light fill
    fig.add_scatter(x=dates, y=vals, mode="lines+markers",
                    line=dict(color=line_color, width=2, dash="dot"),
                    marker=dict(color=line_color, size=5),
                    fill="tozeroy" if fill else None,
                    fillcolor=_rgba(color, 0.15) if fill else None,
                    name="Forecast")
    fig.update_layout(title=dict(text=metric, font=dict(size=11), x=0.02))
    return fig


def _chart_risk_bar(risks, horizontal: bool):
    if not risks:
        return None
    ordered = sorted(risks, key=lambda r: r.get("risk_score") or 0, reverse=True)[:12]
    ents = [r.get("entity_ref") or "—" for r in ordered]
    scores = [float(r.get("risk_score") or 0) for r in ordered]
    colors = [_RISK_COLOR.get(r.get("risk_level"), "#9ca3af") for r in ordered]

    fig = go.Figure()
    if horizontal:
        fig.add_bar(x=scores[::-1], y=ents[::-1], orientation="h",
                    marker_color=colors[::-1], name="Risk Score")
        fig.update_layout(xaxis_title="Risk Score")
    else:
        fig.add_bar(x=ents, y=scores, marker_color=colors, name="Risk Score")
        fig.update_layout(yaxis_title="Risk Score", xaxis=dict(tickangle=-45))
    fig.update_layout(title=dict(text="Risk score by equipment", font=dict(size=11), x=0.02),
                      showlegend=False)
    return fig


def _chart_anomaly_scatter(anomalies):
    if not anomalies:
        return None
    obs = [a.get("observed_value") for a in anomalies]
    exp = [a.get("expected_value") for a in anomalies]
    sev = [a.get("severity") for a in anomalies]
    if all(o is None for o in obs) or all(e is None for e in exp):
        return None
    fig = go.Figure()
    for level in ("low", "medium", "high"):
        xs = [exp[i] for i in range(len(sev)) if sev[i] == level and exp[i] is not None]
        ys = [obs[i] for i in range(len(sev)) if sev[i] == level and obs[i] is not None]
        if xs:
            fig.add_scatter(x=xs, y=ys, mode="markers",
                            marker=dict(size=10, color=_SEV_COLOR[level], opacity=0.75),
                            name=level.capitalize())
    fig.update_layout(title=dict(text="Observed vs expected", font=dict(size=11), x=0.02),
                      xaxis_title="Expected", yaxis_title="Observed")
    return fig


# ============================================================
# KPI cards
# ============================================================

def _kpi_row(overview: dict[str, Any], active: set[str]) -> html.Div:
    items = [
        ("Total Forecasts", overview.get("forecast_count", 0),
         f"Across {len(active)} domains", theme.COLORS["brand"]),
        ("Anomalies Detected", overview.get("anomaly_count", 0),
         "Flagged readings", theme.COLORS["skipped"]),
        ("High Risk Entities", overview.get("high_risk_count", 0),
         "Assets + Maintenance", theme.COLORS["danger"]),
        ("Models Run", 3, "Forecast · Anomaly · Risk", theme.COLORS["collected"]),
    ]
    cards = []
    for title, value, sub, color in items:
        cards.append(
            html.Div(
                style={**theme.CARD_STYLE, "marginBottom": 0, "borderTop": f"3px solid {color}"},
                children=[
                    html.Div(str(value),
                             style={"fontSize": "1.8rem", "fontWeight": 700, "color": color}),
                    html.Div(title,
                             style={"fontSize": "0.85rem", "fontWeight": 600,
                                    "color": theme.COLORS["text"]}),
                    html.Div(sub,
                             style={"fontSize": "0.75rem", "color": theme.COLORS["text_muted"]}),
                ],
            )
        )
    return html.Div(
        style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
               "gap": "1rem", "marginBottom": "1.5rem"},
        children=cards,
    )


# ============================================================
# Unified table
# ============================================================

def _unified_table(metrics, forecasts, anomalies, risks, active) -> html.Div:
    current_map: dict[tuple, float] = {}
    for m in metrics:
        current_map[(str(m["domain"]).lower(), m["metric_name"])] = m.get("avg_value")

    fc_group = _group(forecasts, "domain")
    fc_avg: dict[tuple, float] = {}
    fc_change: dict[tuple, float] = {}
    for domain, rows in fc_group.items():
        by_metric: dict[str, list[dict]] = {}
        for r in rows:
            by_metric.setdefault(r["metric_name"], []).append(r)
        for metric, series in by_metric.items():
            series = sorted(series, key=lambda r: r["forecast_date"])
            vals = [r.get("forecast_value") for r in series if r.get("forecast_value") is not None]
            if vals:
                fc_avg[(domain, metric)] = sum(vals) / len(vals)
                fc_change[(domain, metric)] = vals[-1] - vals[0]

    an_count: dict[tuple, int] = {}
    for a in anomalies:
        key = (str(a["domain"]).lower(), a["metric_name"])
        an_count[key] = an_count.get(key, 0) + 1

    rk_map: dict[tuple, dict] = {}
    for r in risks:
        key = (str(r["domain"]).lower(), r.get("entity_ref"))
        cur = rk_map.get(key)
        if cur is None or (r.get("risk_score") or 0) > (cur.get("risk_score") or 0):
            rk_map[key] = r

    keys = set(current_map) | set(fc_avg) | set(an_count)
    keys = {k for k in keys if k[0] in active}

    # Build row records first so we can sort fully-filled rows to the top.
    records = []
    for domain, metric in keys:
        current = current_map.get((domain, metric))
        fcast = fc_avg.get((domain, metric))
        change = fc_change.get((domain, metric))
        anoms = an_count.get((domain, metric), 0)
        domain_risk = _max_domain_risk(rk_map, domain)
        # A row is "complete" when every column has a real value (risk present too).
        filled = all(v is not None for v in (current, fcast, change)) and domain_risk is not None
        records.append({
            "domain": domain, "metric": metric, "current": current,
            "fcast": fcast, "change": change, "anoms": anoms,
            "risk": domain_risk, "filled": filled,
        })

    # Complete rows first, then partial; each group alphabetized by domain/metric.
    records.sort(key=lambda r: (not r["filled"], r["domain"], r["metric"]))

    if not records:
        return ""

    body_rows = []
    for rec in records:
        body_rows.append(
            html.Tr([
                html.Td(rec["domain"].capitalize(),
                        style={**_TD, "color": theme.domain_color(rec["domain"]),
                               "fontWeight": 600}),
                html.Td(rec["metric"], style=_TD),
                html.Td(_fmt(rec["current"]), style=_TD),
                html.Td(_fmt(rec["fcast"]), style=_TD),
                html.Td(_change_cell(rec["change"]), style={**_TD, "textAlign": "right"}),
                html.Td(_anom_cell(rec["anoms"]), style=_TD),
                html.Td(_risk_cell(rec["risk"]), style={**_TD, "borderRight": "none"}),
            ])
        )

    header = html.Tr([
        html.Th(h, style=(_TH if i < 6 else {**_TH, "borderRight": "none"}))
        for i, h in enumerate(
            ["Domain", "Metric", "Current", "Forecast (7d avg)", "Trend", "Anomalies", "Risk"]
        )
    ])
    table = html.Table(
        [html.Thead(header), html.Tbody(body_rows)],
        style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.85rem"},
    )
    return html.Div(
        children=[
            html.H3("Prediction data", style={"fontSize": "1.2rem", "marginBottom": "0.25rem"}),
            html.P("Current, forecast, trend, anomalies and risk per domain metric. "
                   "Fully-populated rows are shown first.",
                   style=theme.SUBHEADING_STYLE),
            html.Div(table, style=theme.CARD_STYLE),
        ]
    )


def _max_domain_risk(rk_map: dict[tuple, dict], domain: str) -> dict | None:
    best = None
    for (d, _ent), r in rk_map.items():
        if d != domain:
            continue
        if best is None or (r.get("risk_score") or 0) > (best.get("risk_score") or 0):
            best = r
    return best


# ============================================================
# Small helpers
# ============================================================

def _change_cell(change: float | None) -> Any:
    if change is None:
        return html.Span("—", style={"color": theme.COLORS["text_muted"]})
    sign = "+" if change >= 0 else "−"
    color = theme.COLORS["collected"] if change >= 0 else theme.COLORS["danger"]
    return html.Span(f"{sign}{abs(change):.2f}", style={"color": color, "fontWeight": 600})


def _anom_cell(count: int) -> Any:
    if count == 0:
        return html.Span("0", style={"color": theme.COLORS["text_muted"]})
    return html.Span(str(count),
                     style={"color": theme.COLORS["danger"], "fontWeight": 600})


def _risk_cell(risk: dict | None) -> Any:
    if not risk:
        return html.Span("—", style={"color": theme.COLORS["text_muted"]})
    level = risk.get("risk_level") or "low"
    score = _fmt(risk.get("risk_score"))
    return html.Span(f"{score} ({level})",
                     style={"color": _RISK_COLOR.get(level, "#6b7280"), "fontWeight": 600})


_TH = {
    "textAlign": "left",
    "padding": "0.5rem 0.6rem",
    "borderBottom": f"2px solid {theme.COLORS['border']}",
    "borderRight": f"1px solid {theme.COLORS['border']}",
    "color": theme.COLORS["text_muted"],
    "fontWeight": 600,
}
_TD = {
    "padding": "0.4rem 0.6rem",
    "borderBottom": f"1px solid {theme.COLORS['border']}",
    "borderRight": f"1px solid {theme.COLORS['border']}",
}


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(120,120,120,{alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


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