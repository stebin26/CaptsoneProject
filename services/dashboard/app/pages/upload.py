"""/ -- layout only.

The topbar owns the title. This page owns its content and nothing else.
No <h1>, no style dict, no theme import.
"""

from __future__ import annotations

import dash
from dash import dcc, html

from app import ids
from app.components import ui
from app.components.nav import BY_HREF

dash.register_page(__name__, path="/upload", name="Upload")


def layout() -> html.Div:
    """Build the Upload page.

    Where a business uploads a CSV and onboarding begins.

    Returns:
        The page layout.
    """
    return ui.page(
        ui.lede(BY_HREF["/upload"].subtitle),
        ui.card(
            ui.field(
                "Business name",
                ui.text_input(ids.UPLOAD_BUSINESS_NAME, "e.g. NorthStar Telecom"),
            ),
            ui.field(
                "Industry (optional)",
                ui.text_input(
                    ids.UPLOAD_INDUSTRY, "e.g. telecom, manufacturing, aerospace"
                ),
            ),
            ui.field(
                "Data file (CSV)",
                dcc.Upload(
                    id=ids.UPLOAD_DATA,
                    multiple=False,
                    accept=".csv",
                    className="dropzone",
                    children=html.Div(
                        ["Drag and drop, or ", html.B("select a CSV file")]
                    ),
                ),
            ),
            html.Div(id=ids.UPLOAD_FILENAME, style={"marginBottom": "1rem"}),
            ui.button("Profile and suggest mappings", ids.UPLOAD_SUBMIT),
            html.Div(id=ids.UPLOAD_STATUS, style={"marginTop": "1rem"}),
        ),
    )
