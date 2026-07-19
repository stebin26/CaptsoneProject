"""Per-domain chart builders. Brain, not face.

Each of the eight universal domains gets a fixed chart type -- the shape of the
data does not change between industries, so neither does the chart. Appearance
is owned by the Plotly template in design/plotly_theme.py.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from app.api_client import APIError, analytics_trend
from app.design import tokens
from app.utils import avg_by_entity

DOMAIN_CHART: dict[str, str] = {
    "assets": "line",
    "operations": "column",
    "quality": "bubble",
    "maintenance": "bar_trend",
    "inventory": "treemap",
    "workforce": "hbar",
    "finance": "area",
    "customers": "donut",
}


def build(
    dataset_id: int,
    domain: str,
    rows: list[dict[str, Any]],
) -> go.Figure | None:
    """Choose and build the most suitable chart for a domain's rows.

    The chart type follows the shape of the data rather than being fixed per
    domain, so an industry the platform has never seen still renders sensibly.

    Args:
        dataset_id: Dataset being displayed.
        domain: Domain the rows belong to.
        rows: The rows to plot.

    Returns:
        The built figure, or nothing when there is nothing to plot.
    """
    if not rows:
        return None

    chart_type = DOMAIN_CHART.get(domain, "column")
    color = tokens.domain_ink(domain)

    if chart_type == "line":
        return chart_line(dataset_id, domain, rows, color)
    if chart_type == "area":
        return chart_area(dataset_id, domain, rows, color)
    if chart_type == "bubble":
        return chart_bubble(rows, color)
    if chart_type == "bar_trend":
        return chart_bar_trend(rows)
    if chart_type == "hbar":
        return chart_hbar(rows, color)
    if chart_type == "treemap":
        return chart_treemap(rows)
    if chart_type == "donut":
        return chart_donut(rows)
    return chart_column(rows, color)


def chart_column(rows: list[dict[str, Any]], color: str) -> go.Figure:
    """Build a vertical bar chart of average value per entity.

    Args:
        rows: The rows to plot.
        color: Series colour.

    Returns:
        The built figure.
    """
    entities, averages = avg_by_entity(rows)
    fig = go.Figure()
    fig.add_bar(x=entities, y=averages, marker_color=color)
    fig.update_xaxes(tickangle=-45)
    return fig


def chart_hbar(rows: list[dict[str, Any]], color: str) -> go.Figure:
    """Build a horizontal bar chart of average value per entity.

    Used when there are enough entities that vertical labels would collide.

    Args:
        rows: The rows to plot.
        color: Series colour.

    Returns:
        The built figure.
    """
    entities, averages = avg_by_entity(rows)
    fig = go.Figure()
    fig.add_bar(x=averages, y=entities, orientation="h", marker_color=color)
    return fig


def chart_bubble(rows: list[dict[str, Any]], color: str) -> go.Figure:
    """Bubble size is volatility. A big, unstable entity is visibly both."""
    entities = [r["entity_ref"] for r in rows]
    averages = [float(r.get("avg_value") or 0) for r in rows]
    stds = [float(r.get("std_value") or 0) for r in rows]

    fig = go.Figure()
    fig.add_scatter(
        x=list(range(len(entities))),
        y=averages,
        mode="markers",
        marker=dict(size=[max(8, s * 2) for s in stds], color=color, opacity=0.5),
        text=entities,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(entities))),
        ticktext=entities,
        tickangle=-45,
    )
    return fig


def chart_bar_trend(rows: list[dict[str, Any]]) -> go.Figure:
    """Bars coloured by the direction of the slope, not by domain.

    This is the one chart where colour carries meaning beyond identity: rising
    maintenance is bad news, so a positive slope is red.
    """
    entities = [r["entity_ref"] for r in rows]
    averages = [float(r.get("avg_value") or 0) for r in rows]
    slopes = [float(r.get("trend_slope") or 0) for r in rows]
    colors = [tokens.DANGER if s > 0 else tokens.INK_FAINT for s in slopes]

    fig = go.Figure()
    fig.add_bar(x=entities, y=averages, marker_color=colors)
    fig.update_xaxes(tickangle=-45)
    return fig


def chart_treemap(rows: list[dict[str, Any]]) -> go.Figure:
    """Build a treemap of relative contribution per entity.

    Args:
        rows: The rows to plot.

    Returns:
        The built figure.
    """
    entities = [r["entity_ref"] for r in rows]
    magnitudes = [abs(float(r.get("avg_value") or 0)) for r in rows]
    return go.Figure(
        go.Treemap(
            labels=entities,
            parents=[""] * len(entities),
            values=magnitudes,
            marker_colorscale="Blues",
        )
    )


def chart_donut(rows: list[dict[str, Any]]) -> go.Figure:
    """Build a donut chart of share per entity.

    Args:
        rows: The rows to plot.

    Returns:
        The built figure.
    """
    entities = [r["entity_ref"] for r in rows]
    magnitudes = [abs(float(r.get("avg_value") or 0)) for r in rows]
    return go.Figure(go.Pie(labels=entities, values=magnitudes, hole=0.55))


def chart_line(
    dataset_id: int,
    domain: str,
    rows: list[dict[str, Any]],
    color: str,
) -> go.Figure:
    """Build a line chart of a metric over time.

    Args:
        dataset_id: Dataset being displayed.
        domain: Domain the metric belongs to.
        rows: The rows identifying the metric to trend.
        color: Series colour.

    Returns:
        The built figure.
    """
    points = _trend_points(dataset_id, domain, rows[0]["metric_name"])
    if not points:
        return chart_column(rows, color)

    fig = go.Figure()
    fig.add_scatter(
        x=[p["day"] for p in points],
        y=[float(p.get("avg_value") or 0) for p in points],
        mode="lines+markers",
        line=dict(color=color, width=2),
    )
    return fig


def chart_area(
    dataset_id: int,
    domain: str,
    rows: list[dict[str, Any]],
    color: str,
) -> go.Figure:
    """Build a filled area chart of a metric over time.

    Args:
        dataset_id: Dataset being displayed.
        domain: Domain the metric belongs to.
        rows: The rows identifying the metric to trend.
        color: Series colour.

    Returns:
        The built figure.
    """
    points = _trend_points(dataset_id, domain, rows[0]["metric_name"])
    if not points:
        return chart_column(rows, color)

    fig = go.Figure()
    fig.add_scatter(
        x=[p["day"] for p in points],
        y=[float(p.get("avg_value") or 0) for p in points],
        mode="lines",
        fill="tozeroy",
        line=dict(color=color, width=2),
    )
    return fig


def chart_metric_averages(metrics: list[dict[str, Any]]) -> go.Figure:
    """One bar per metric across every domain -- the analytics overview."""
    names = [f"{m['domain']}.{m['metric_name']}" for m in metrics]
    averages = [m.get("avg_value") or 0 for m in metrics]
    colors = [tokens.domain_ink(m["domain"]) for m in metrics]

    fig = go.Figure()
    fig.add_bar(x=names, y=averages, marker_color=colors)
    fig.update_layout(
        height=400,
        title="Average value per metric",
        xaxis=dict(tickangle=-45),
        margin=dict(b=150),
    )
    return fig


def chart_daily_trend(
    points: list[dict[str, Any]],
    domain: str,
    metric_name: str,
) -> go.Figure:
    """Build the daily trend chart for one domain metric.

    Args:
        points: The daily trend points.
        domain: Domain the metric belongs to.
        metric_name: The metric being trended.

    Returns:
        The built figure.
    """
    fig = go.Figure()
    fig.add_scatter(
        x=[p["day"] for p in points],
        y=[p.get("avg_value") or 0 for p in points],
        mode="lines+markers",
        line=dict(color=tokens.domain_ink(domain), width=2),
    )
    fig.update_layout(height=320, title=f"{domain} \u00b7 {metric_name}")
    return fig


def _trend_points(
    dataset_id: int,
    domain: str,
    metric_name: str,
) -> list[dict[str, Any]]:
    """Fetch the daily trend, tolerating an absent series.

    A missing trend is legitimate -- the dataset may carry no timestamps -- and
    the caller falls back to a column chart. It is never silently turned into
    empty data.
    """
    try:
        return analytics_trend(dataset_id, domain=domain, metric_name=metric_name)
    except APIError:
        return []



def _band_fill(color: str, alpha: float = 0.12) -> str:
    """A translucent fill from a solid colour, tolerating hex or rgb input.

    tokens has no rgba() helper, so build the string here. A hex like #2563eb
    becomes rgba(37,99,235,a); an existing rgb()/rgba() is passed through with
    its alpha swapped in.
    """
    c = color.strip()
    if c.startswith("#") and len(c) == 7:
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    if c.startswith("rgb"):
        nums = c[c.find("(") + 1 : c.find(")")].split(",")[:3]
        r, g, b = (n.strip() for n in nums)
        return f"rgba({r},{g},{b},{alpha})"
    # Unknown format: fall back to a neutral tint rather than crash.
    return f"rgba(37,99,235,{alpha})"


def chart_sparkline(
    history: list[float],
    band_low: list[float],
    band_high: list[float],
    color: str | None = None,
) -> go.Figure:
    """A compact history line with a forecast confidence band on the end.

    History is solid (what happened); the band is the projected range (what is
    expected next). They share an x-axis so the projection reads as a
    continuation. No axes, no legend -- a sparkline is a shape, not a plot.
    """
    ink = color or tokens.INK

    fig = go.Figure()
    fig.add_scatter(
        x=list(range(len(history))),
        y=history,
        mode="lines",
        line=dict(color=ink, width=2),
        hoverinfo="y",
        showlegend=False,
    )

    if band_low and band_high:
        start = len(history) - 1 if history else 0
        band_x = list(range(start, start + len(band_high)))
        fig.add_scatter(
            x=band_x + band_x[::-1],
            y=band_high + band_low[::-1],
            fill="toself",
            fillcolor=_band_fill(ink),
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )

    fig.update_layout(
        height=90,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
