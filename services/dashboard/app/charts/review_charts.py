"""Figure builders for /review. Brain, not face."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from app.design import tokens


def coverage(rows: list[dict[str, Any]]) -> go.Figure:
    """Collected vs skipped, per domain.

    A dataset covering four of eight domains is visibly half-blind. Showing the
    skipped bar alongside the collected one is what makes that legible -- and
    what makes a later "no data for that question" explainable rather than
    mysterious.
    """
    domains = [c["domain"] for c in rows]
    collected = [c["features_collected"] for c in rows]
    skipped = [c["features_skipped"] for c in rows]

    fig = go.Figure()
    fig.add_bar(name="Collected", x=domains, y=collected, marker_color=tokens.OK)
    fig.add_bar(name="Skipped", x=domains, y=skipped, marker_color=tokens.WARN)
    fig.update_layout(
        barmode="stack",
        height=300,
        title="Feature coverage by domain",
    )
    return fig


def metric_totals(metrics: list[dict[str, Any]]) -> go.Figure:
    """Summed value per metric across the hub, coloured by domain."""
    names = [f"{m['domain']}.{m['metric_name']}" for m in metrics]
    totals = [m.get("metric_sum") or 0 for m in metrics]
    colors = [tokens.domain_ink(m["domain"]) for m in metrics]

    fig = go.Figure()
    fig.add_bar(x=names, y=totals, marker_color=colors)
    fig.update_layout(
        height=380,
        title="Metric totals across the hub",
        xaxis=dict(tickangle=-45),
        margin=dict(b=140),
    )
    return fig
