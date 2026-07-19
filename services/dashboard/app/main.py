"""Dash app entry point.

The shell owns the chrome and the global stores. Pages own their content and
nothing else -- they never build navigation, never fetch the dataset list for
scoping, and never render their own topbar.
"""

from __future__ import annotations

import os

from dash import Dash, html, page_container

from app.components import shell

# Registers the Plotly template and makes it the default, which restyles every
# chart in the app without touching a single figure builder.
from app.design import plotly_theme  # noqa: F401

app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title="Industrial Data Hub",
    update_title=None,
    external_stylesheets=[
        # Inter, with a system fallback declared in 00-tokens.css. If the
        # network is unavailable during a demo the type degrades gracefully
        # rather than the page breaking.
        "https://fonts.googleapis.com/css2"
        "?family=Inter:wght@400;500;600;700&display=swap",
    ],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)

server = app.server  # exposed for gunicorn / WSGI

app.layout = html.Div(
    [
        *shell.stores(),
        shell.render(page_container),
    ]
)

if __name__ == "__main__":
    host = os.environ.get("OPS_DASH_HOST", "0.0.0.0")
    port = int(os.environ.get("OPS_DASH_PORT", "8050"))
    debug = os.environ.get("OPS_ENVIRONMENT", "development") == "development"
    app.run(host=host, port=port, debug=debug)


# ============================================================
# Callback registration  --  MUST stay at the bottom of this file
# ============================================================
# Importing this package is what registers every @callback in the app. It must
# come AFTER Dash() is constructed and after app.layout is assigned: the
# callback modules touch Dash's global registry, and if they are imported
# before the app exists the import chain breaks before `server` is defined --
# gunicorn then fails to find app.main:server and exits 3 with no log output.
#
# Never move this to the top. Never remove it: without it the app renders
# perfectly and does nothing.

from app import callbacks  # noqa: E402,F401
