"""Design system.

The CSS variables in app/assets/00-tokens.css are the single source of truth
for every colour, space and radius in the app. This package exists only for
the places Python genuinely needs a value: Plotly cannot read CSS variables,
so the chart palette has to be mirrored here. That is the one and only reason
any colour is written twice.
"""
