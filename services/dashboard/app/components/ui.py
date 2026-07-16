"""Shared UI components. FACE.

Styling lives in app/assets/*.css. These return className, never a style dict.
If a page needs a card, a chip or a table, it comes from here -- so changing
how a card looks is one edit, not nine.
"""

from __future__ import annotations

from typing import Any

from dash import dcc, html

from app.constants import domain_label
from app.utils import fmt


def page(*children: Any) -> html.Div:
    """A page body. The topbar already carries the title -- do not repeat it."""
    return html.Div(list(children), className="page")


def lede(text: str) -> html.P:
    """The one sentence that says what this page proves. Directly under the topbar."""
    return html.P(text, className="page-subtitle")


def section(title: str, *children: Any, note: str | None = None) -> html.Div:
    head: list[Any] = [html.H2(title, className="section-title")]
    if note:
        head.append(html.P(note, className="page-subtitle"))
    return html.Div([*head, *children], style={"marginTop": "2rem"})


def card(*children: Any, className: str = "") -> html.Div:
    return html.Div(list(children), className=f"card {className}".strip())


def grid(*children: Any, cols: int = 2) -> html.Div:
    return html.Div(list(children), className=f"grid grid-{cols}")


def rule() -> html.Hr:
    return html.Hr(className="rule")


# ---------- KPI ----------

def kpi(label: str, value: Any, note: str = "", tone: str = "") -> html.Div:
    """One number, one label, one line of context. Tone colours the number only."""
    return html.Div(
        [
            html.Div(label, className="kpi-label"),
            html.Div(
                str(value),
                className="kpi-value",
                style={"color": f"var(--{tone})"} if tone else None,
            ),
            html.Div(note, className="kpi-note") if note else None,
        ],
        className="kpi",
    )


# ---------- Domain ----------

def domain_chip(domain: str, present: bool = True, suffix: str = "") -> html.Span:
    """A domain chip. Absent domains are shown, not hidden.

    An empty slot is information: it is what makes a later "no data for that
    question" explainable rather than mysterious.
    """
    classes = f"domain-chip d-{domain.lower()}"
    if not present:
        classes += " is-absent"

    label = domain_label(domain)
    if suffix:
        label = f"{label} \u00b7 {suffix}"

    return html.Span(label, className=classes)


def domain_tile(domain: str) -> html.Span:
    return html.Span(
        domain_label(domain)[0],
        className=f"domain-tile d-{domain.lower()}",
    )


# ---------- Badges ----------

def badge(level: str | None) -> html.Span:
    """A severity or risk level. Unknown levels are neutral, never red."""
    value = (level or "").lower()
    known = {"critical", "high", "medium", "low"}
    tone = value if value in known else "neutral"
    return html.Span(value or "\u2014", className=f"badge badge-{tone}")


def trend(value: float | None, rising_is_bad: bool = False) -> html.Span:
    """An arrow with a colour.

    Direction is not the same as good or bad -- rising downtime is bad, rising
    output is good. The caller says which, because only the caller knows.
    """
    if value is None:
        return html.Span("\u2014", className="trend-flat")

    rising = value > 0
    if abs(value) < 1e-9:
        return html.Span("\u2192", className="trend-flat")

    bad = rising if rising_is_bad else not rising
    arrow = "\u2191" if rising else "\u2193"
    return html.Span(arrow, className="trend-up" if bad else "trend-down")


# ---------- Table ----------

def table(headers: list[str], rows: list[list[Any]], note: str | None = None) -> Any:
    """A table that scrolls inside its card rather than widening the page."""
    element = html.Div(
        html.Table(
            [
                html.Thead(html.Tr([html.Th(h) for h in headers])),
                html.Tbody([html.Tr([html.Td(c) for c in row]) for row in rows]),
            ],
            className="table",
        ),
        className="table-scroll",
    )
    if not note:
        return element
    return html.Div([element, html.P(note, className="table-note")])

# ---------- Form ----------

def field(label: str, control: Any) -> html.Div:
    return html.Div(
        [html.Label(label, className="label"), control],
        style={"marginBottom": "1rem"},
    )


def text_input(component_id: str, placeholder: str = "", value: str = "") -> dcc.Input:
    return dcc.Input(
        id=component_id,
        type="text",
        value=value,
        placeholder=placeholder,
        className="input",
    )


def button(
    label: str,
    component_id: str,
    variant: str = "primary",
    small: bool = False,
) -> html.Button:
    classes = f"btn btn-{variant}"
    if small:
        classes += " btn-sm"
    return html.Button(label, id=component_id, n_clicks=0, className=classes)


# ---------- Chart ----------

def chart(figure: Any, title: str | None = None) -> html.Div:
    """A chart in a card. The card owns the chrome; the figure owns the data."""
    children: list[Any] = []
    if title:
        children.append(html.Div(title, className="card-title"))
    children.append(dcc.Graph(figure=figure, config=_GRAPH_CONFIG))
    return html.Div(children, className="card")


_GRAPH_CONFIG = {
    "displaylogo": False,
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "responsive": True,
}


# Re-exported so callbacks can format numbers without a second import.
__all__ = [
    "page", "lede", "section", "card", "grid", "rule",
    "kpi", "domain_chip", "domain_tile", "badge", "trend",
    "table", "field", "text_input", "button", "chart", "fmt",
]