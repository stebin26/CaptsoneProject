"""/confirm -- layout only. Callbacks live in app/callbacks/mapping_confirm.py."""

from __future__ import annotations

import dash
from dash import html

from app import ids
from app.components import ui
from app.components.nav import BY_HREF

dash.register_page(__name__, path="/confirm", name="Mapping")


def layout() -> html.Div:
    """Build the Mapping Confirmation page.

    The human decision point of onboarding: the suggested column mapping is
    reviewed and confirmed once before any data is loaded.

    Returns:
        The page layout.
    """
    return ui.page(
        ui.lede(BY_HREF["/confirm"].subtitle),
        html.Div(id=ids.CONFIRM_HEADER),
        html.Div(id=ids.CONFIRM_COLUMNS),
        html.Div(
            ui.button("Confirm and load into hub", ids.CONFIRM_SUBMIT),
            style={"marginTop": "1.5rem"},
        ),
        html.Div(id=ids.CONFIRM_STATUS, style={"marginTop": "1rem"}),
    )
