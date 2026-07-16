"""Small shared building blocks."""

from __future__ import annotations

from typing import Any

from dash import html


def icon(name: str, size: str = "") -> html.Span:
    """An icon. Colour is inherited from the parent via currentColor."""
    classes = " ".join(filter(None, ["icon", name, size]))
    return html.Span(className=classes)


def empty_state(
    title: str,
    body: str,
    icon_name: str = "i-hub",
    action: Any = None,
) -> html.Div:
    """An empty screen is an invitation to act, not an apology.

    It says what will be here, and what unlocks it.
    """
    children: list[Any] = [
        icon(icon_name),
        html.Div(title, className="empty-title"),
        html.P(body, className="empty-body"),
    ]
    if action is not None:
        children.append(action)

    return html.Div(children, className="empty-state")