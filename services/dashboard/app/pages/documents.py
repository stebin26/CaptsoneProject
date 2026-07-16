"""/documents -- layout only. Callbacks live in app/callbacks/documents.py."""

from __future__ import annotations

import dash
from dash import dcc, html

from app import ids
from app.components import ui
from app.components.nav import BY_HREF

dash.register_page(__name__, path="/documents", name="Documents")


def layout() -> html.Div:
    return ui.page(
        html.H1("What the numbers cannot tell you", className="page-title"),
        ui.lede(BY_HREF["/documents"].subtitle),

        ui.field(
            "Dataset",
            dcc.Dropdown(
                id=ids.DOC_DATASET,
                options=[],
                placeholder="Select a dataset",
                clearable=False,
            ),
        ),
        dcc.Interval(id=ids.DOC_INIT, interval=300, max_intervals=1),

        dcc.Upload(
            id=ids.DOC_UPLOAD,
            multiple=True,
            className="dropzone",
            children=html.Div(
                [
                    html.Div(["Drag and drop, or ", html.B("browse")]),
                    html.Div(
                        "PDF, DOCX or TXT \u00b7 several at once",
                        className="kpi-note",
                    ),
                ]
            ),
        ),
        html.Div(id=ids.DOC_UPLOAD_STATUS, style={"margin": "1rem 0"}),

        # Polls only while something is still being indexed.
        dcc.Interval(id=ids.DOC_POLL, interval=3000, disabled=True),
        html.Div(id=ids.DOC_LIST),

        ui.rule(),
        ui.section(
            "Ask your documents",
            html.Div(
                [
                    dcc.Input(
                        id=ids.DOC_QUESTION,
                        type="text",
                        placeholder=(
                            "e.g. What is the maintenance interval for the pumps?"
                        ),
                        className="input",
                        debounce=True,
                        style={"flex": "1"},
                    ),
                    html.Button(
                        "Ask",
                        id=ids.DOC_ASK,
                        n_clicks=0,
                        className="btn btn-primary",
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "0.5rem",
                    "marginBottom": "1rem",
                },
            ),
            dcc.Loading(html.Div(id=ids.DOC_ANSWER), type="dot"),
            note=(
                "Answers come only from the documents uploaded for this dataset. "
                "If nothing matches, it says so rather than inventing a manual."
            ),
        ),
    )