"""/settings -- routed, reachable, honest about not being built yet."""

from __future__ import annotations

import dash
from dash import html

from app.components.primitives import empty_state

dash.register_page(__name__, path="/settings", name="Settings")


def layout() -> html.Div:
    return empty_state(
        title="Settings",
        body=(
            "Platform configuration lives here. Nothing to configure yet \u2014 "
            "the pipeline runs on defaults, and every knob that matters is in "
            "the mapping step."
        ),
        icon_name="i-settings",
    )