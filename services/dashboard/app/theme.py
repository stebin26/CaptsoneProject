from __future__ import annotations

# ============================================================
# Color palette
# ============================================================

COLORS = {
    "brand": "#4f46e5",
    "brand_dark": "#4338ca",
    "brand_light": "#eef2ff",
    "collected": "#16a34a",
    "skipped": "#d97706",
    "danger": "#dc2626",
    "text": "#1f2937",
    "text_muted": "#6b7280",
    "bg": "#f9fafb",
    "surface": "#ffffff",
    "border": "#e5e7eb",
}

# Per-domain colors for charts (matches the 8 universal domains)
DOMAIN_COLORS = {
    "assets": "#4f46e5",
    "operations": "#0ea5e9",
    "quality": "#dc2626",
    "maintenance": "#d97706",
    "inventory": "#16a34a",
    "workforce": "#9333ea",
    "finance": "#0d9488",
    "customers": "#db2777",
}

DEFAULT_DOMAIN_COLOR = "#6b7280"


def domain_color(domain: str) -> str:
    return DOMAIN_COLORS.get(domain.lower(), DEFAULT_DOMAIN_COLOR)


# ============================================================
# Reusable inline styles
# ============================================================

PAGE_STYLE = {
    "maxWidth": "1100px",
    "margin": "0 auto",
    "padding": "2rem 1.5rem",
    "fontFamily": "Inter, system-ui, sans-serif",
    "color": COLORS["text"],
}

CARD_STYLE = {
    "backgroundColor": COLORS["surface"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "12px",
    "padding": "1.25rem 1.5rem",
    "marginBottom": "1rem",
    "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
}

PRIMARY_BUTTON_STYLE = {
    "backgroundColor": COLORS["brand"],
    "color": "#ffffff",
    "border": "none",
    "borderRadius": "8px",
    "padding": "0.6rem 1.2rem",
    "fontSize": "0.95rem",
    "fontWeight": 500,
    "cursor": "pointer",
}

SECONDARY_BUTTON_STYLE = {
    "backgroundColor": COLORS["surface"],
    "color": COLORS["brand"],
    "border": f"1px solid {COLORS['brand']}",
    "borderRadius": "8px",
    "padding": "0.5rem 1rem",
    "fontSize": "0.9rem",
    "fontWeight": 500,
    "cursor": "pointer",
}

ADD_BUTTON_STYLE = {
    "backgroundColor": COLORS["collected"],
    "color": "#ffffff",
    "border": "none",
    "borderRadius": "6px",
    "padding": "0.35rem 0.8rem",
    "fontSize": "0.85rem",
    "fontWeight": 500,
    "cursor": "pointer",
}

HEADING_STYLE = {
    "fontSize": "1.6rem",
    "fontWeight": 600,
    "marginBottom": "0.25rem",
    "color": COLORS["text"],
}

SUBHEADING_STYLE = {
    "fontSize": "0.95rem",
    "color": COLORS["text_muted"],
    "marginBottom": "1.5rem",
}

LABEL_STYLE = {
    "fontSize": "0.85rem",
    "fontWeight": 500,
    "color": COLORS["text_muted"],
    "marginBottom": "0.3rem",
    "display": "block",
}

INPUT_STYLE = {
    "width": "100%",
    "padding": "0.6rem 0.8rem",
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "8px",
    "fontSize": "0.95rem",
    "marginBottom": "1rem",
}


# ============================================================
# Status badge helper
# ============================================================

def status_badge_style(status: str) -> dict[str, str]:
    palette = {
        "collected": COLORS["collected"],
        "added_later": COLORS["collected"],
        "skipped": COLORS["skipped"],
    }
    color = palette.get(status, COLORS["text_muted"])
    return {
        "display": "inline-block",
        "padding": "0.15rem 0.6rem",
        "borderRadius": "999px",
        "fontSize": "0.75rem",
        "fontWeight": 500,
        "color": "#ffffff",
        "backgroundColor": color,
    }