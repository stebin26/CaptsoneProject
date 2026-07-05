from __future__ import annotations

import os

import dash
from dash import Dash, dcc, html, page_container

from app import theme

app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title="Operations Intelligence Platform",
)

server = app.server  # exposed for gunicorn / WSGI


def _nav_bar() -> html.Div:
    link_style = {
        "color": theme.COLORS["brand"],
        "textDecoration": "none",
        "fontWeight": 500,
        "marginRight": "1.5rem",
        "fontSize": "0.95rem",
    }
    return html.Div(
        style={
            "borderBottom": f"1px solid {theme.COLORS['border']}",
            "backgroundColor": theme.COLORS["surface"],
            "padding": "1rem 1.5rem",
        },
        children=[
            html.Div(
                style={"maxWidth": "1100px", "margin": "0 auto",
                       "display": "flex", "alignItems": "center"},
                children=[
                    html.Span(
                        "Ops Intelligence",
                        style={"fontWeight": 700, "fontSize": "1.05rem",
                               "color": theme.COLORS["text"], "marginRight": "2rem"},
                    ),
                    dcc.Link("Upload", href="/", style=link_style),
                    dcc.Link("Confirm", href="/confirm", style=link_style),
                    dcc.Link("Review", href="/review", style=link_style),
                    dcc.Link("Datasets", href="/datasets", style=link_style),
                    dcc.Link("Analytics", href="/analytics", style=link_style),
                    dcc.Link("Predictions", href="/predictions", style=link_style),
                    dcc.Link("Intelligence", href="/intelligence", style=link_style),
                    
                    dcc.Link("Documents", href="/documents", style=link_style),
                ],
            )
        ],
    )


app.layout = html.Div(
    style={"backgroundColor": theme.COLORS["bg"], "minHeight": "100vh"},
    children=[
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="onboarding-store", storage_type="session"),
        dcc.Store(id="review-refresh", storage_type="memory"),
        _nav_bar(),
        page_container,
    ],
)


if __name__ == "__main__":
    host = os.environ.get("OPS_DASH_HOST", "0.0.0.0")
    port = int(os.environ.get("OPS_DASH_PORT", "8050"))
    debug = os.environ.get("OPS_ENVIRONMENT", "development") == "development"
    app.run(host=host, port=port, debug=debug)