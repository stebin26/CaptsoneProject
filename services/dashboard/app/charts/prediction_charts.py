"""Figure builders for /predictions. Brain, not face.

Each domain gets the chart best suited to its dominant ML output. Every builder
returns None when it has no data to draw -- the caller falls back to another
chart rather than rendering an empty axis. An absent chart is a legitimate
state; a blank one is a lie.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from app.design import tokens
from app.utils import rgba

# Which chart each domain gets. Fixed, industry-independent.
DOMAIN_CHART: dict[str, str] = {
    "assets": "risk_hbar",
    "operations": "forecast_line",
    "quality": "anomaly_scatter",
    "maintenance": "risk_vbar",
    "inventory": "forecast_area",
    "workforce": "forecast_line",
    "finance": "forecast_area",
    "customers": "forecast_area",
}

CHART_SUBTITLE: dict[str, str] = {
    "risk_hbar": "Risk ranking",
    "risk_vbar": "Risk ranking by entity",
    "forecast_line": "Forecast with confidence band",
    "forecast_area": "Forecast",
    "anomaly_scatter": "Observed against expected",
}

RISK_COLOR: dict[str, str] = {
    "high": tokens.DANGER,
    "medium": tokens.WARN,
    "low": tokens.OK,
}

SEVERITY_COLOR: dict[str, str] = {
    "high": tokens.DANGER,
    "medium": tokens.WARN,
    "low": tokens.INK_FAINT,
}

_FALLBACK_COLOR = tokens.INK_FAINT


def build(
    chart_type: str,
    forecasts: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    risks: list[dict[str, Any]],
    color: str,
) -> go.Figure | None:
    """Build the domain's chart, falling back when its preferred data is absent."""
    if chart_type == "risk_hbar":
        fig = chart_risk_bar(risks, horizontal=True)
        fig = fig if fig is not None else chart_forecast(forecasts, color, fill=False)
    elif chart_type == "risk_vbar":
        fig = chart_risk_bar(risks, horizontal=False)
        fig = fig if fig is not None else chart_forecast(forecasts, color, fill=False)
    elif chart_type == "anomaly_scatter":
        fig = chart_anomaly_scatter(anomalies)
        fig = fig if fig is not None else chart_forecast(forecasts, color, fill=False)
    elif chart_type == "forecast_area":
        fig = chart_forecast(forecasts, color, fill=True)
        fig = fig if fig is not None else chart_risk_bar(risks, horizontal=True)
    else:  # forecast_line
        fig = chart_forecast(forecasts, color, fill=False)
        fig = fig if fig is not None else chart_risk_bar(risks, horizontal=True)

    if fig is None:
        return None

    # Margins, background, font and legend all come from the Plotly template.
    # Only the height is per-chart.
    fig.update_layout(height=260)
    return fig


def chart_forecast(
    forecasts: list[dict[str, Any]],
    color: str,
    fill: bool,
) -> go.Figure | None:
    """Build a forecast chart with its confidence band.

    The band is drawn rather than only the central line, so the projection reads as
    an estimate with uncertainty instead of a promise.

    Args:
        forecasts: The forecast points with their bounds.
        color: Series colour.
        fill: Colour for the confidence band.

    Returns:
        The built figure, or nothing when there is nothing to plot.
    """
    if not forecasts:
        return None

    metric = sorted({f["metric_name"] for f in forecasts})[0]
    series = sorted(
        (f for f in forecasts if f["metric_name"] == metric),
        key=lambda f: f["forecast_date"],
    )

    dates = [f["forecast_date"] for f in series]
    values = [f.get("forecast_value") for f in series]
    lower = [f.get("lower_bound") for f in series]
    upper = [f.get("upper_bound") for f in series]

    fig = go.Figure()

    has_band = all(v is not None for v in upper) and all(v is not None for v in lower)
    if has_band:
        fig.add_scatter(
            x=dates + dates[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor=rgba(color, 0.12),
            line=dict(width=0),
            hoverinfo="skip",
            name="Confidence band",
        )

    line_color = tokens.INK if fill else color  # area charts: dark line, light fill
    fig.add_scatter(
        x=dates,
        y=values,
        mode="lines+markers",
        line=dict(color=line_color, width=2, dash="dot"),
        marker=dict(color=line_color, size=5),
        fill="tozeroy" if fill else None,
        fillcolor=rgba(color, 0.15) if fill else None,
        name="Forecast",
    )
    fig.update_layout(title=metric)
    return fig


def chart_risk_bar(
    risks: list[dict[str, Any]],
    horizontal: bool,
) -> go.Figure | None:
    """Build a bar chart of entity risk scores.

    Args:
        risks: The risk scores to plot.
        horizontal: Draw the bars horizontally.

    Returns:
        The built figure, or nothing when there is nothing to plot.
    """
    if not risks:
        return None

    ordered = sorted(risks, key=lambda r: r.get("risk_score") or 0, reverse=True)[:12]
    entities = [r.get("entity_ref") or "\u2014" for r in ordered]
    scores = [float(r.get("risk_score") or 0) for r in ordered]
    colors = [RISK_COLOR.get(r.get("risk_level"), _FALLBACK_COLOR) for r in ordered]

    fig = go.Figure()
    if horizontal:
        fig.add_bar(
            x=scores[::-1],
            y=entities[::-1],
            orientation="h",
            marker_color=colors[::-1],
            name="Risk score",
        )
        fig.update_layout(xaxis_title="Risk score")
    else:
        fig.add_bar(x=entities, y=scores, marker_color=colors, name="Risk score")
        fig.update_layout(yaxis_title="Risk score", xaxis=dict(tickangle=-45))

    fig.update_layout(title="Risk score by entity", showlegend=False)
    return fig


def chart_anomaly_scatter(
    anomalies: list[dict[str, Any]],
) -> go.Figure | None:
    """Build a scatter chart of anomalies over time, coloured by severity.

    Args:
        anomalies: The anomalies to plot.

    Returns:
        The built figure, or nothing when there is nothing to plot.
    """
    if not anomalies:
        return None

    observed = [a.get("observed_value") for a in anomalies]
    expected = [a.get("expected_value") for a in anomalies]
    severity = [a.get("severity") for a in anomalies]

    if all(o is None for o in observed) or all(e is None for e in expected):
        return None

    fig = go.Figure()
    for level in ("low", "medium", "high"):
        xs, ys = [], []
        for i, sev in enumerate(severity):
            if sev != level or expected[i] is None or observed[i] is None:
                continue
            xs.append(expected[i])
            ys.append(observed[i])
        if xs:
            fig.add_scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=10, color=SEVERITY_COLOR[level], opacity=0.75),
                name=level.capitalize(),
            )

    fig.update_layout(
        title="Observed against expected",
        xaxis_title="Expected",
        yaxis_title="Observed",
    )
    return fig
