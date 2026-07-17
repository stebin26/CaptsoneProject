"""Callbacks for / (Executive Dashboard).

One interval fires once, one endpoint answers, one callback fans the result
out across every slot. The endpoint already did the cross-table assembly
server-side, so there is no orchestration here -- just rendering.
"""

from __future__ import annotations

from typing import Any

from dash import Input, Output, State, callback, html
from app import feedback, ids
from app.api_client import APIError, executive_summary, list_datasets
from app.charts import domain_charts
from app.components import ui
from app.constants import DOMAIN_ORDER
from app.design import tokens
from app.utils import fmt

# Risk band -> the CSS tone variable the KPI number is coloured with.
_BAND_TONE = {"high": "danger", "elevated": "warn", "low": "ok"}


@callback(
    Output(ids.EXEC_STORE, "options"),
    Output(ids.EXEC_STORE, "value"),
    Input(ids.EXEC_INIT, "n_intervals"),
    State(ids.ACCESS_TOKEN, "data"),
)
def populate_datasets(_init: int | None, token: str | None) -> tuple[list[dict[str, Any]], Any]:
    try:
        datasets = list_datasets(token=token)
    except APIError:
        return [], None

    options = [
        {"label": d["business_name"], "value": d["dataset_id"]} for d in datasets
    ]
    return options, (options[0]["value"] if options else None)


@callback(
    Output(ids.EXEC_ERROR, "children"),
    Output(ids.EXEC_KPI_RISK, "children"),
    Output(ids.EXEC_KPI_ALERTS, "children"),
    Output(ids.EXEC_KPI_INDUSTRY, "children"),
    Output(ids.EXEC_KPI_FRESHNESS, "children"),
    Output(ids.EXEC_DOMAIN_HEALTH, "children"),
    Output(ids.EXEC_TOP_RISKS, "children"),
    Output(ids.EXEC_ACTIVE_ALERTS, "children"),
    Output(ids.EXEC_FORECASTS, "children"),
    Output(ids.EXEC_INSIGHTS, "children"),
    Input(ids.EXEC_STORE, "value"),
    State(ids.ACCESS_TOKEN, "data"),
)
def load_summary(dataset_id: int | None, token: str | None):
    blank = ("", "", "", "", "", "", "", "", "")
    if dataset_id is None:
        return ("", *blank)

    try:
        s = executive_summary(dataset_id, token=token)
    except APIError as exc:
        return (feedback.error(f"Could not load the executive summary: {exc}"), *blank)

    return (
        "",
        _kpi_risk(s["risk_index"]),
        _kpi_alerts(s),
        _kpi_industry(s),
        _kpi_freshness(s),
        _domain_health(s["domain_health"]),
        _top_risks(s["top_risks"]),
        _active_alerts(s["active_alerts"]),
        _forecasts(s["forecasts"]),
        _insights(s["insights"]),
    )


# ============================================================
# KPI tiles
# ============================================================

def _kpi_risk(idx: dict[str, Any]) -> Any:
    tone = _BAND_TONE.get(idx["band"], "")
    peak = idx.get("peak_domain")
    note = f"{idx['label']} \u00b7 driven by {peak}" if peak else idx["label"]
    return ui.kpi("Operational Risk (relative)", idx["value"], note=note, tone=tone)


def _kpi_alerts(s: dict[str, Any]) -> Any:
    high = s["high_alert_count"]
    note = f"{high} high severity" if high else "none high severity"
    tone = "danger" if high else ""
    return ui.kpi("Open alerts", s["open_alert_count"], note=note, tone=tone)


def _kpi_industry(s: dict[str, Any]) -> Any:
    return ui.kpi(
        "Coverage",
        f"{s['active_domain_count']} / 8",
        note=(s.get("industry") or "industry unspecified"),
    )


def _kpi_freshness(s: dict[str, Any]) -> Any:
    start, end = s.get("date_start"), s.get("date_end")
    value = end or "\u2014"
    note = f"from {start}" if start else "no dated records"
    return ui.kpi("Latest data", value, note=note)


# ============================================================
# Domain health
# ============================================================

def _domain_health(rows: list[dict[str, Any]]) -> Any:
    by_domain = {r["domain"]: r for r in rows}
    chips = []
    for domain in DOMAIN_ORDER:
        r = by_domain.get(domain)
        present = bool(r and r["active"])
        suffix = ""
        if r and r["score"] is not None:
            suffix = f"{fmt(r['score'], 0)}"
        elif r and r["open_alerts"]:
            suffix = f"{r['open_alerts']} alerts"
        chips.append(ui.domain_chip(domain, present=present, suffix=suffix))

    return html.Div(
        chips,
        style={"display": "flex", "flexWrap": "wrap", "gap": "0.5rem"},
    )


# ============================================================
# Tables
# ============================================================

def _top_risks(rows: list[dict[str, Any]]) -> Any:
    if not rows:
        return feedback.empty("No risk scores yet. Run the ML orchestration DAG.")

    body = [
        [
            ui.domain_chip(r["domain"]),
            r["entity_ref"] or "\u2014",
            fmt(r["score"], 1),
            ui.badge(r["band"]),
            # Risk rising is bad news, so a positive slope shows red.
            ui.trend(r["trend_slope"], rising_is_bad=True),
        ]
        for r in rows
    ]
    return ui.card(ui.table(["Domain", "Entity", "Score", "Level", "Trend"], body))


def _active_alerts(rows: list[dict[str, Any]]) -> Any:
    if not rows:
        return feedback.empty("No anomalies flagged for this dataset.")

    body = [
        [
            ui.domain_chip(a["domain"]),
            a["entity_ref"] or "\u2014",
            a["metric_name"],
            ui.badge(a["severity"]),
            a["when"] or "\u2014",
            fmt(a["observed"]),
            fmt(a["expected"]),
        ]
        for a in rows
    ]
    return ui.card(
        ui.table(
            ["Domain", "Entity", "Metric", "Severity", "When", "Observed", "Expected"],
            body,
        )
    )


# ============================================================
# Forecasts + insights
# ============================================================

def _forecasts(rows: list[dict[str, Any]]) -> Any:
    if not rows:
        return feedback.empty("No forecasts yet. Run the ML orchestration DAG.")

    cards = []
    for f in rows:
        color = tokens.domain_ink(f["domain"])
        fig = domain_charts.chart_sparkline(
            f.get("history") or [],
            f.get("band_low") or [],
            f.get("band_high") or [],
            color=color,
        )
        pct = f.get("pct_change")
        rising_is_bad = f["domain"] in ("quality", "maintenance")
        header = html.Div(
            [
                ui.domain_chip(f["domain"]),
                html.Span(f["metric_name"], className="card-title",
                          style={"marginLeft": "0.5rem"}),
            ],
            style={"display": "flex", "alignItems": "center",
                   "marginBottom": "0.25rem"},
        )
        delta = html.Div(
            [
                html.Span(f"{fmt(f.get('last_value'))} \u2192 {fmt(f.get('next_value'))}  "),
                html.Span(
                    [f"{fmt(pct, 1)}%  " if pct is not None else "", ui.trend(pct, rising_is_bad)]
                ),
            ],
            className="kpi-note",
        )
        cards.append(ui.card(header, ui.chart(fig), delta))

    return ui.grid(*cards, cols=3)


def _insights(rows: list[dict[str, Any]]) -> Any:
    if not rows:
        return feedback.empty(
            "No cross-domain insights \u2014 this dataset carries a risk signal "
            "in only one domain, so there is nothing to correlate across."
        )

    cards = []
    for t in rows:
        cards.append(
            ui.card(
                html.Div(
                    [
                        ui.domain_chip(t["root"]),
                        html.Span(t["root_term"], style={"fontWeight": 600,
                                                         "marginLeft": "0.5rem"}),
                    ],
                    style={"marginBottom": "0.5rem"},
                ),
                html.P(t["narrative"], style={"margin": 0}),
            )
        )
    return html.Div(cards, className="grid grid-1")