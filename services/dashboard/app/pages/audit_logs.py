"""/audit-logs -- routed, reachable, honest about not being built yet."""

from __future__ import annotations

import dash
from dash import html

from app.components.primitives import empty_state

dash.register_page(__name__, path="/audit-logs", name="Audit Logs")


def layout() -> html.Div:
    """Build the Audit Logs page.

    Currently a placeholder; the page exists so the navigation and permission model
    stay complete.

    Returns:
        The page layout.
    """
    return empty_state(
        title="Audit logs",
        body=(
            "Who did what, and when. Every upload, mapping change and agent "
            "query, attributable to a person. This unlocks with authentication."
        ),
        icon_name="i-audit",
    )
