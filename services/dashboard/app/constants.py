"""Domain vocabulary shared across pages, charts and callbacks."""

from __future__ import annotations

# The eight universal domains, in canonical display order.
DOMAIN_ORDER: list[str] = [
    "assets",
    "operations",
    "quality",
    "maintenance",
    "inventory",
    "workforce",
    "finance",
    "customers",
]

DOMAIN_LABELS: dict[str, str] = {
    "assets": "Assets",
    "operations": "Operations",
    "quality": "Quality",
    "maintenance": "Maintenance",
    "inventory": "Inventory",
    "workforce": "Workforce",
    "finance": "Finance",
    "customers": "Customers",
}


def domain_label(domain: str) -> str:
    return DOMAIN_LABELS.get(domain.lower(), domain.replace("_", " ").title())

# Seed prompts for the copilot. Each one exercises a different tool path:
# hub, ML, causal analytics, and cross-domain intelligence.
SUGGESTED_PROMPTS: list[str] = [
    "What does this dataset contain?",
    "What are the main risks and problems?",
    "Why might production be dropping?",
    "How do the business areas affect each other?",
]