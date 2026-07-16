"""/copilot -- layout only. Callbacks live in app/callbacks/copilot.py."""

from __future__ import annotations

import uuid

import dash
from dash import dcc, html

from app import ids
from app.components import ui
from app.components.nav import BY_HREF
from app.constants import SUGGESTED_PROMPTS

dash.register_page(__name__, path="/copilot", name="AI Copilot")


def layout() -> html.Div:
    return ui.page(
        html.H1("Ask, and see the working", className="page-title"),
        ui.lede(BY_HREF["/copilot"].subtitle),

        html.Div(id=ids.COPILOT_STATUS),
        dcc.Interval(id=ids.COPILOT_INIT, interval=300, max_intervals=1),

        ui.field(
            "Scope",
            dcc.Dropdown(
                id=ids.COPILOT_DATASET,
                options=[],
                placeholder="All datasets \u2014 the agent will choose",
                clearable=True,
            ),
        ),

        html.Div(id=ids.COPILOT_TRANSCRIPT, className="transcript"),

        html.Div(
            [
                html.Span(
                    prompt,
                    id=ids.copilot_suggestion(i),
                    n_clicks=0,
                    className="prompt-chip",
                )
                for i, prompt in enumerate(SUGGESTED_PROMPTS)
            ],
            id=ids.COPILOT_SUGGESTIONS,
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "0.5rem",
                "margin": "0.75rem 0",
            },
        ),

        html.Div(
            [
                dcc.Textarea(
                    id=ids.COPILOT_INPUT,
                    placeholder="Ask about your operations\u2026",
                    className="input",
                    style={"flex": "1", "minHeight": "56px", "resize": "vertical"},
                ),
                html.Button(
                    "Ask",
                    id=ids.COPILOT_SEND,
                    n_clicks=0,
                    className="btn btn-primary",
                    style={"height": "56px"},
                ),
            ],
            style={"display": "flex", "gap": "0.5rem", "alignItems": "flex-end"},
        ),
        html.P(
            "The agent runs on a local model and reasons step by step, so an "
            "answer can take a minute or two. It is working, not stuck.",
            className="kpi-note",
            style={"marginTop": "0.5rem"},
        ),

        # The session id lets the agent resolve "that" against the previous turn.
        dcc.Store(id=ids.COPILOT_HISTORY, data=[]),
        dcc.Store(id=ids.COPILOT_SESSION, data=str(uuid.uuid4())),
        dcc.Store(id=ids.COPILOT_PENDING, data=None),
    )