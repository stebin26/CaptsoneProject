"""Callbacks for /intelligence.

The knowledge graph is the map; the ML signals decide which roads are lit. An
edge only surfaces when both domains carry a corroborating signal in this
dataset -- the graph alone is never enough.

Strengths are named (Critical, Strong, Medium, Weak), not decimal. A plant
manager cannot act on 0.62. They can act on "Strong".
"""

from __future__ import annotations

from typing import Any

from dash import Input, Output, State, callback, html

from app import feedback, ids
from app.api_client import APIError, intelligence, list_datasets
from app.components import ui
from app.constants import DOMAIN_ORDER, domain_label

DIRECTION_ARROW: dict[str, str] = {
    "down": "\u2193",
    "up": "\u2191",
    "flat": "\u2192",
}

# Rising and falling are not good and bad. Which is which depends on the metric.
DIRECTION_TONE: dict[str, str] = {
    "down": "trend-up",  # something falling that matters -- coloured as a concern
    "up": "trend-down",
    "flat": "trend-flat",
}


@callback(
    Output(ids.BI_DATASET, "options"),
    Output(ids.BI_DATASET, "value"),
    Input(ids.BI_INIT, "n_intervals"),
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
    Output(ids.BI_SUMMARY, "children"),
    Output(ids.BI_INSIGHTS, "children"),
    Input(ids.BI_DATASET, "value"),
    State(ids.ACCESS_TOKEN, "data"),
)
def load_intelligence(dataset_id: int | None, token: str | None) -> tuple[Any, Any]:
    """Load and render the cross-domain insights for a dataset.

    Args:
        dataset_id: Selected dataset.
        token: Caller's access token.

    Returns:
        The rendered insights, or a message when none are available.
    """
    if dataset_id is None:
        return "", ""

    try:
        data = intelligence(dataset_id, token=token)
    except APIError as exc:
        return feedback.error(f"Could not load intelligence: {exc}"), ""

    insights = data.get("insights", [])
    active = data.get("active_domains", [])
    summary = _summary(data, active, insights)

    if not insights:
        return summary, ui.card(
            html.Div("No cross-domain insights yet", className="empty-title"),
            html.P(
                "An insight appears when two connected domains both carry a "
                "corroborating ML signal. Run the ML orchestration DAG so the "
                "signals exist, then come back.",
                className="empty-body",
            ),
        )

    return summary, html.Div([_insight(ins) for ins in insights])


# ============================================================
# Render helpers
# ============================================================


def _summary(
    data: dict[str, Any],
    active: list[str],
    insights: list[dict[str, Any]],
) -> html.Div:
    """All eight domains, always. The dimmed ones say what this dataset cannot see."""
    active_set = {d.lower() for d in active}

    return ui.card(
        html.Div(
            [
                html.Span(
                    data.get("business_name") or "Unknown",
                    style={"fontWeight": 600},
                ),
                html.Span(
                    f"  \u00b7  {data.get('industry') or 'industry unspecified'}"
                    f"  \u00b7  {len(insights)} insight(s) across "
                    f"{len(active)} of 8 domains",
                    className="kpi-note",
                ),
            ],
            style={"marginBottom": "0.75rem"},
        ),
        html.Div(
            [
                ui.domain_chip(domain, present=domain in active_set)
                for domain in DOMAIN_ORDER
            ],
            style={"display": "flex", "flexWrap": "wrap", "gap": "0.5rem"},
        ),
    )


def _insight(ins: dict[str, Any]) -> html.Div:
    root = (ins.get("root") or "").lower()
    is_loop = ins.get("is_loop", False)

    return html.Div(
        [
            _insight_head(ins, root, is_loop),
            html.P(ins.get("narrative", ""), className="insight-text"),
            _impacted(ins.get("impacted", [])),
            _recommendation(ins.get("recommendation", "")),
        ],
        className=f"insight d-{root or 'none'}",
        style={"marginTop": "0.75rem"},
    )


def _insight_head(
    ins: dict[str, Any],
    root: str,
    is_loop: bool,
) -> html.Div:
    children: list[Any] = [
        html.Span(
            (ins.get("root_term") or domain_label(root)),
            style={"fontWeight": 600},
        )
    ]

    # A feedback loop has no single direction. Showing an arrow on one would
    # misrepresent it, so it gets a badge instead.
    if is_loop:
        partner = ins.get("loop_partner")
        label = "feedback loop"
        if partner:
            label += f" \u00b7 {domain_label(partner)}"
        children.append(html.Span(label, className="badge badge-medium"))
    else:
        direction = ins.get("direction", "flat")
        children.append(
            html.Span(
                DIRECTION_ARROW.get(direction, "\u2192"),
                className=DIRECTION_TONE.get(direction, "trend-flat"),
            )
        )

    children.append(
        html.Span(
            f"priority {ins.get('score', 0):.1f}",
            className="kpi-note",
        )
    )

    return html.Div(
        children,
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "0.6rem",
            "flexWrap": "wrap",
            "marginBottom": "0.5rem",
        },
    )


def _impacted(impacted: list[dict[str, Any]]) -> html.Div:
    """Named strengths, never decimals.

    "Why is Maintenance to Operations Critical?" has a business answer.
    "Why is it 0.83?" does not.
    """
    chips = []
    for imp in impacted:
        domain = (imp.get("domain") or "").lower()
        term = imp.get("term") or domain_label(domain)
        strength = (imp.get("strength") or "").replace("_", " ")
        chips.append(
            ui.domain_chip(domain, suffix=strength)
            if domain
            else html.Span(f"{term} \u00b7 {strength}", className="badge badge-neutral")
        )

    return html.Div(
        chips,
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "0.4rem",
            "margin": "0.75rem 0",
        },
    )


def _recommendation(text: str) -> html.Div:
    if not text:
        return html.Div()

    return html.Div(
        [
            html.Div("What to do", className="kpi-label"),
            html.Div(text),
        ],
        style={
            "background": "var(--accent-tint)",
            "borderRadius": "var(--r-md)",
            "padding": "0.6rem 0.9rem",
        },
    )
