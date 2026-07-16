"""A Plotly template that restyles every chart in the app.

Importing this module registers the template and makes it the default. Every
figure built anywhere in the app picks it up -- no figure code changes, no
per-chart styling, no risk of drift between two charts on the same page.

Figure builders remain BRAIN: they encode data. Appearance is owned here.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from app.constants import DOMAIN_ORDER
from app.design import tokens

TEMPLATE_NAME = "ops"

# Series colours follow the eight domains, in canonical order, so an
# unlabelled multi-series chart still reads in the app's own vocabulary.
_COLORWAY = [tokens.domain_ink(d) for d in DOMAIN_ORDER]

_AXIS = dict(
    showgrid=True,
    gridcolor=tokens.LINE,
    gridwidth=1,
    zeroline=False,
    linecolor=tokens.LINE,
    ticks="outside",
    tickcolor=tokens.LINE,
    ticklen=4,
    tickfont=dict(size=11, color=tokens.INK_MUTED),
    title=dict(font=dict(size=12, color=tokens.INK_MUTED)),
    automargin=True,
)

pio.templates[TEMPLATE_NAME] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=tokens.FONT, size=12, color=tokens.INK),
        colorway=_COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)",  # the card owns the background
        plot_bgcolor="rgba(0,0,0,0)",

        # Bottom margin leaves room for the legend, which sits below the plot
        # rather than above it -- above, it collides with the title.
        margin=dict(l=48, r=16, t=36, b=64),

        title=dict(
            font=dict(size=13, color=tokens.INK),
            x=0,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        xaxis=_AXIS,
        yaxis=_AXIS,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0,
            font=dict(size=11, color=tokens.INK_MUTED),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor=tokens.INK,
            bordercolor=tokens.INK,
            font=dict(family=tokens.FONT, size=12, color="#FFFFFF"),
        ),
        hovermode="closest",
        separators=".,",  # thousands separator, so 1,240 not 1240
        colorscale=dict(sequential=[[0, tokens.ACCENT_TINT], [1, tokens.ACCENT]]),
    )
)

pio.templates.default = TEMPLATE_NAME