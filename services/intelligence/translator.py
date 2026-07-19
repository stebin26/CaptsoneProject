"""Insight translator (Level 2).

Converts the inference engine's generic, domain-level output into business
language using the dataset's own mapped column and business terms. Templates
produce the grounded default wording; an optional single Ollama call polishes
phrasing only. The model never invents facts -- if it is off or fails, the
template wording stands on its own.
"""
# Insight translator (Level 2). Converts generic domain-level inference output into
# human-readable business insights using the dataset's own mapped column/business
# terms. Templates produce the grounded default wording; an optional single Ollama
# call polishes phrasing only — it never invents facts.

from __future__ import annotations

import json
from typing import Any

from ops_common.config import settings
from ops_common.logging import get_logger

logger = get_logger(__name__)

# Fallback nouns per domain when the dataset has no better business term to show.
_DOMAIN_NOUN = {
    "assets": "assets",
    "operations": "operations",
    "quality": "quality",
    "maintenance": "maintenance",
    "inventory": "inventory",
    "workforce": "workforce",
    "finance": "financial performance",
    "customers": "customer activity",
}

# Verb phrasing for a domain's own movement, chosen by forecast direction.
_DIRECTION_PHRASE = {
    "down": "is trending downward",
    "up": "is trending upward",
    "flat": "shows unusual movement",
}

# Severity wording driven by the strongest edge strength in the impact set.
_STRENGTH_WORD = {
    "critical": "strongly",
    "strong": "significantly",
    "medium": "moderately",
    "weak": "slightly",
    "very_weak": "marginally",
}


# ---------------------------------------------------------------------------
# Business-term resolution
# ---------------------------------------------------------------------------


def build_term_map(
    metrics: list[dict[str, Any]] | None = None,
    mapping: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Resolve a display term per domain from the dataset's real columns."""
    term: dict[str, str] = {}

    if mapping:
        for domain, label in mapping.items():
            if label:
                term[str(domain).lower()] = str(label)

    if metrics:
        by_domain: dict[str, list[str]] = {}
        for m in metrics:
            d = str(m.get("domain", "")).lower()
            name = m.get("metric_name")
            if d and name:
                by_domain.setdefault(d, []).append(str(name))
        for d, names in by_domain.items():
            if d not in term:
                term[d] = _humanize(names[0])

    for d, noun in _DOMAIN_NOUN.items():
        term.setdefault(d, noun)

    return term


def _humanize(column: str) -> str:
    return column.replace("_", " ").strip().lower()


def _term(term_map: dict[str, str], domain: str) -> str:
    return term_map.get(domain.lower(), _DOMAIN_NOUN.get(domain.lower(), domain))


# ---------------------------------------------------------------------------
# Insight → template sentence (grounded default)
# ---------------------------------------------------------------------------


def translate_insight(
    insight: dict[str, Any],
    term_map: dict[str, str],
) -> dict[str, Any]:
    """Turn one engine insight into a grounded, business-worded narrative."""
    root = insight["root"]
    direction = insight.get("root_direction", "flat")
    impacts = insight.get("impacts", [])

    root_term = _term(term_map, root)
    root_metric = insight.get("root_metric")
    root_metric_phrase = f" ({_humanize(root_metric)})" if root_metric else ""

    for p in impacts:
        p["term"] = _term(term_map, p["target"])

    # Feedback loop: a reinforcing two-way relationship reads as one stronger story.
    if insight.get("is_loop"):
        partner_term = _term(term_map, insight.get("loop_partner", ""))
        fwd = insight.get("loop_forward_label") or "affects it"
        rev = insight.get("loop_reverse_label") or "feeds back"
        narrative = (
            f"{root_term.capitalize()} and {partner_term} are in a reinforcing loop — "
            f"{root_term} {fwd}, and in turn {partner_term} {rev}. Left unchecked, "
            f"this cycle compounds."
        )
        recommendation = (
            f"Break the loop at {root_term}: acting there eases the cycle with "
            f"{partner_term} rather than treating each side in isolation."
        )
        impacted = [
            {
                "domain": p["target"],
                "term": p["term"],
                "strength": p["strength"],
                "effect": p["effect"],
                "label": p["label"],
            }
            for p in impacts
        ]
        return {
            "root": root,
            "root_term": root_term,
            "direction": direction,
            "score": insight.get("score", 0.0),
            "is_loop": True,
            "loop_partner": insight.get("loop_partner"),
            "narrative": narrative,
            "recommendation": recommendation,
            "impacted": impacted,
        }

    opener = f"{root_term.capitalize()}{root_metric_phrase} {_DIRECTION_PHRASE.get(direction, 'shows unusual movement')}"

    primary = [p for p in impacts if p["strength"] in ("critical", "strong")]
    secondary = [p for p in impacts if p["strength"] not in ("critical", "strong")]

    clauses: list[str] = []
    for p in primary:
        adv = _STRENGTH_WORD.get(p["strength"], "")
        clauses.append(f"{adv} affecting {p['term']} ({p['label']})")
    for p in secondary:
        clauses.append(f"with a lesser effect on {p['term']}")

    if clauses:
        body = ", ".join(clauses[:-1])
        narrative = (
            f"{opener}, {clauses[0]}."
            if len(clauses) == 1
            else f"{opener}, {body}, and {clauses[-1]}."
        )
    else:
        narrative = f"{opener}."

    recommendation = _recommend(root_term, primary, secondary, direction)

    impacted = [
        {
            "domain": p["target"],
            "term": p["term"],
            "strength": p["strength"],
            "effect": p["effect"],
            "label": p["label"],
        }
        for p in impacts
    ]

    return {
        "root": root,
        "root_term": root_term,
        "direction": direction,
        "score": insight.get("score", 0.0),
        "is_loop": False,
        "loop_partner": None,
        "narrative": narrative,
        "recommendation": recommendation,
        "impacted": impacted,
    }


def _recommend(
    root_term: str,
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    direction: str,
) -> str:
    if primary:
        top = primary[0]
        return f"Prioritize {root_term} to protect {top['term']} — {top['label']}."
    if secondary:
        return f"Monitor {root_term}; downstream effects are currently limited."
    return f"Review {root_term} — its movement is not yet propagating to other areas."


# ---------------------------------------------------------------------------
# LLM polish — single call, phrasing only, strictly grounded on the facts
# ---------------------------------------------------------------------------

_POLISH_SYSTEM = (
    "You rewrite business-intelligence findings into clear, natural English. "
    "You are given structured FACTS for each finding. Reword only — never add, "
    "remove, or change any fact, domain, number, or relationship. Keep every "
    "business term exactly as given. Return strictly valid JSON, no prose."
)


def _polish_prompt(translated: list[dict[str, Any]]) -> str:
    # Send only the fields the LLM may reword; it must echo them back polished.
    payload = [
        {
            "id": i,
            "narrative": t["narrative"],
            "recommendation": t["recommendation"],
        }
        for i, t in enumerate(translated)
    ]
    return (
        "Reword each finding's 'narrative' and 'recommendation' to sound natural "
        "and professional, preserving all facts and terms exactly.\n\n"
        f"FACTS:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
        'Return JSON only: {"items": [{"id": <int>, "narrative": "...", '
        '"recommendation": "..."}]}'
    )


def _call_ollama_polish(prompt: str) -> str | None:
    import requests

    try:
        resp = requests.post(
            f"{settings.ollama_url.rstrip('/')}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": [
                    {"role": "system", "content": _POLISH_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2, "num_predict": 500},
            },
            timeout=(5, 120),
        )
        resp.raise_for_status()
        return (resp.json().get("message", {}).get("content") or "").strip() or None
    except Exception:  # noqa: BLE001
        logger.exception("Intelligence LLM polish failed")
        return None


def _apply_polish(translated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not translated:
        return translated

    provider = settings.llm_provider.lower()
    if provider != "ollama":
        # Only the local provider is wired for polish; others fall back to template.
        return translated

    raw = _call_ollama_polish(_polish_prompt(translated))
    if not raw:
        return translated

    try:
        data = json.loads(raw)
        items = data.get("items", [])
        by_id = {int(it["id"]): it for it in items if "id" in it}
    except Exception:  # noqa: BLE001
        logger.warning("Could not parse LLM polish output; using templates")
        return translated

    for i, t in enumerate(translated):
        it = by_id.get(i)
        if not it:
            continue
        # Reword only; keep all structured fields (root, impacted, score) intact.
        if it.get("narrative"):
            t["narrative"] = str(it["narrative"]).strip()
        if it.get("recommendation"):
            t["recommendation"] = str(it["recommendation"]).strip()
    return translated


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def translate_all(
    insights: list[dict[str, Any]],
    metrics: list[dict[str, Any]] | None = None,
    mapping: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Resolve terms, build grounded template narratives, then optionally polish."""
    term_map = build_term_map(metrics=metrics, mapping=mapping)
    translated = [translate_insight(ins, term_map) for ins in insights]

    if settings.llm_enabled and settings.intelligence_llm_polish:
        translated = _apply_polish(translated)

    return translated
