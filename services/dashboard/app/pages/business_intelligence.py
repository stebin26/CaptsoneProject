from __future__ import annotations

from typing import Any

import dash
from dash import Input, Output, callback, dcc, html

from app.api_client import APIError, intelligence, list_datasets
from app import theme

dash.register_page(__name__, path="/intelligence", name="Intelligence")


_STRENGTH_COLOR = {
    "critical": "#dc2626",
    "strong": "#d97706",
    "medium": "#0d9488",
    "weak": "#6b7280",
    "very_weak": "#9ca3af",
}

_DIRECTION_ICON = {"down": "▼", "up": "▲", "flat": "◆"}
_DIRECTION_COLOR = {"down": "#dc2626", "up": "#16a34a", "flat": "#d97706"}

_LOOP_COLOR = "#7c3aed"  # distinct purple for reinforcing-loop insights


def layout() -> html.Div:
    return html.Div(
        style=theme.PAGE_STYLE,
        children=[
            html.H1("Business Intelligence", style=theme.HEADING_STYLE),
            html.P(
                "Cross-domain insights. The engine reasons over how the active business "
                "domains influence one another, using the ML signals from this dataset, "
                "and explains the story in your own business terms.",
                style=theme.SUBHEADING_STYLE,
            ),
            html.Label("Dataset", style=theme.LABEL_STYLE),
            dcc.Dropdown(
                id="bi-dataset",
                options=[],
                placeholder="Select a dataset",
                clearable=False,
                style={"marginBottom": "1.5rem"},
            ),
            dcc.Interval(id="bi-init", interval=300, max_intervals=1),
            html.Div(id="bi-summary"),
            html.Div(id="bi-insights"),
        ],
    )


# ============================================================
# Dataset dropdown
# ============================================================

@callback(
    Output("bi-dataset", "options"),
    Output("bi-dataset", "value"),
    Input("bi-init", "n_intervals"),
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
# Load insights
# ============================================================

@callback(
    Output("bi-summary", "children"),
    Output("bi-insights", "children"),
    Input("bi-dataset", "value"),
)
def load_intelligence(dataset_id: int | None) -> tuple[Any, Any]:
    if dataset_id is None:
        return "", ""

    try:
        data = intelligence(dataset_id)
    except APIError as exc:
        return _error(f"Could not load intelligence: {exc}"), ""

    insights = data.get("insights", [])
    active = data.get("active_domains", [])

    # Active-domain chips make it transparent which domains produced these insights.
    domain_chips = [
        html.Span(
            d.capitalize(),
            style={
                "fontSize": "0.78rem",
                "padding": "0.2rem 0.65rem",
                "borderRadius": "999px",
                "marginRight": "0.4rem",
                "marginTop": "0.5rem",
                "display": "inline-block",
                "backgroundColor": theme.COLORS["brand_light"],
                "color": theme.domain_color(d),
                "border": f"1px solid {theme.domain_color(d)}",
                "fontWeight": 600,
            },
        )
        for d in active
    ]

    summary = html.Div(
        style=theme.CARD_STYLE,
        children=[
            html.Div([
                html.Span(data.get("business_name") or "Unknown",
                          style={"fontWeight": 600, "fontSize": "1.1rem"}),
                html.Span(
                    f"  ·  {data.get('industry') or 'unspecified'}  ·  "
                    f"{len(active)} active domains  ·  "
                    f"{len(insights)} cross-domain insight(s)",
                    style={"color": theme.COLORS["text_muted"]},
                ),
            ]),
            html.Div(
                [html.Span("Active domains: ",
                           style={"fontSize": "0.75rem", "fontWeight": 600,
                                  "color": theme.COLORS["text_muted"],
                                  "marginRight": "0.3rem"}),
                 *domain_chips],
                style={"marginTop": "0.5rem"},
            ),
        ],
    )

    if not insights:
        empty = html.Div(
            "No cross-domain insights yet. This appears when at least two connected "
            "domains show corroborating ML signals. Confirm an upload or run the ML "
            "orchestration DAG so the signals are available.",
            style={"color": theme.COLORS["text_muted"], **theme.CARD_STYLE},
        )
        return summary, empty

    cards = [_insight_card(i + 1, ins) for i, ins in enumerate(insights)]
    return summary, html.Div(cards)


# ============================================================
# Insight card
# ============================================================

def _insight_card(index: int, ins: dict[str, Any]) -> html.Div:
    root = ins.get("root", "")
    is_loop = ins.get("is_loop", False)
    color = _LOOP_COLOR if is_loop else theme.domain_color(root)
    direction = ins.get("direction", "flat")
    dir_icon = _DIRECTION_ICON.get(direction, "◆")
    dir_color = _DIRECTION_COLOR.get(direction, theme.COLORS["text_muted"])

    header_children = [
        html.Span(
            str(index),
            style={
                "backgroundColor": color, "color": "#fff", "borderRadius": "6px",
                "width": "24px", "height": "24px", "display": "inline-flex",
                "alignItems": "center", "justifyContent": "center",
                "fontSize": "0.8rem", "fontWeight": 700, "marginRight": "0.6rem",
            },
        ),
        html.Span(
            (ins.get("root_term") or root).capitalize(),
            style={"fontWeight": 700, "fontSize": "1.05rem", "color": color},
        ),
    ]

    # Feedback-loop insights get a distinct badge instead of a direction arrow.
    if is_loop:
        partner = (ins.get("loop_partner") or "").capitalize()
        header_children.append(
            html.Span(
                f"  ⟲ Feedback Loop{f' · {partner}' if partner else ''}",
                style={
                    "fontSize": "0.72rem", "fontWeight": 700,
                    "color": _LOOP_COLOR, "backgroundColor": "#f3e8ff",
                    "padding": "0.15rem 0.55rem", "borderRadius": "999px",
                    "marginLeft": "0.5rem",
                },
            )
        )
    else:
        header_children.append(
            html.Span(
                f"  {dir_icon}",
                style={"color": dir_color, "fontWeight": 700, "marginLeft": "0.4rem"},
            )
        )

    header_children.append(
        html.Span(
            f"  · priority {ins.get('score', 0):.1f}",
            style={"color": theme.COLORS["text_muted"], "fontSize": "0.8rem",
                   "marginLeft": "0.5rem"},
        )
    )

    header = html.Div(
        style={"display": "flex", "alignItems": "center", "marginBottom": "0.6rem",
               "flexWrap": "wrap"},
        children=header_children,
    )

    narrative = html.P(
        ins.get("narrative", ""),
        style={"fontSize": "0.95rem", "lineHeight": "1.5",
               "color": theme.COLORS["text"], "marginBottom": "0.75rem"},
    )

    # Impacted-domain chips, colored by relationship strength.
    chips = []
    for imp in ins.get("impacted", []):
        s_color = _STRENGTH_COLOR.get(imp.get("strength"), "#6b7280")
        chips.append(
            html.Span(
                [
                    (imp.get("term") or imp.get("domain")).capitalize(),
                    html.Span(f"  {imp.get('strength', '')}",
                              style={"fontWeight": 600, "fontSize": "0.7rem"}),
                ],
                style={
                    "fontSize": "0.8rem",
                    "padding": "0.25rem 0.7rem",
                    "borderRadius": "999px",
                    "marginRight": "0.5rem",
                    "marginBottom": "0.5rem",
                    "display": "inline-block",
                    "border": f"1px solid {s_color}",
                    "color": s_color,
                },
            )
        )

    chip_row = html.Div(chips, style={"marginBottom": "0.75rem"})

    recommendation = html.Div(
        [
            html.Span("Recommendation  ",
                      style={"fontSize": "0.7rem", "fontWeight": 700,
                             "textTransform": "uppercase", "letterSpacing": "0.04em",
                             "color": theme.COLORS["text_muted"]}),
            html.Span(ins.get("recommendation", ""),
                      style={"fontSize": "0.9rem", "color": theme.COLORS["text"]}),
        ],
        style={
            "backgroundColor": theme.COLORS["brand_light"],
            "borderRadius": "8px",
            "padding": "0.6rem 0.9rem",
        },
    )

    return html.Div(
        style={**theme.CARD_STYLE, "borderLeft": f"4px solid {color}"},
        children=[header, narrative, chip_row, recommendation],
    )


# ============================================================
# Helpers
# ============================================================

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