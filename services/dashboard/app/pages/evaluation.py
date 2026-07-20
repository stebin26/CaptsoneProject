"""/evaluation -- layout only.

Callbacks live in app/callbacks/evaluation.py. This page renders the committed
model-evaluation reports (RAG and ML) read-only; it takes no dataset selector,
because the reports are produced by scripts run against a fixed evaluation
corpus rather than per uploaded dataset.
"""

from __future__ import annotations

import dash
from app import ids
from app.components import ui
from app.components.nav import BY_HREF
from dash import dcc, html

dash.register_page(__name__, path="/evaluation", name="Evaluation")


def layout() -> html.Div:
    """Build the Evaluation page.

    Shows measured performance of the RAG assistant and the ML models. The two
    result containers are filled by the page's callback on load.

    Returns:
        The page layout.

    """
    return ui.page(
        html.H1("How well the models actually work", className="page-title"),
        ui.lede(BY_HREF["/evaluation"].subtitle),
        dcc.Interval(id=ids.EVAL_INIT, interval=300, max_intervals=1),
        html.Div(id=ids.EVAL_RAG_SECTION),
        html.Div(id=ids.EVAL_ML_SECTION),
    )