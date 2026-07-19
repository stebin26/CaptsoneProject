"""/users -- routed, reachable, honest about not being built yet."""

from __future__ import annotations

import dash
from dash import html

from app.components.primitives import empty_state

dash.register_page(__name__, path="/users", name="Users")


def layout() -> html.Div:
    """Build the User page.

    Currently a placeholder; the page exists so the navigation and permission model
    stay complete.

    Returns:
        The page layout.
    """
    return empty_state(
        title="Users",
        body=(
            "People with access to this platform, and what each of them can do. "
            "This unlocks with authentication and role-based access control."
        ),
        icon_name="i-users",
    )
