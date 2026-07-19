"""Mapping suggester -- proposing which domain each column belongs to.

This is the automation behind onboarding a new business without writing code:
the column profiles are described to an LLM, which classifies each column into
one of the eight universal domains. Because a cloud model must never be a hard
dependency, a keyword fallback using the domain registry's aliases produces a
usable mapping with the LLM disabled or unreachable -- the platform still runs
fully offline, just with lower-confidence suggestions the user reviews anyway.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ops_common.config import settings
from ops_common.domain.models import Domain
from ops_common.domain.registry import (
    match_domain_by_keyword,
    registry_as_prompt_context,
)
from ops_common.logging import get_logger

from app.profiling.profiler import ColumnProfile, DatasetProfile

logger = get_logger(__name__)

_VALID_DOMAINS = set(Domain.values())


@dataclass
class ColumnSuggestion:
    """A proposed mapping for one column, with its confidence and origin.

    ``source`` records whether the suggestion came from the model, the keyword
    fallback, or nothing, so the interface can show how it was arrived at.
    """
    column_name: str
    suggested_domain: str | None
    suggested_metric: str | None
    role: str  # "metric", "entity", or "skip"
    confidence: float
    source: str  # "llm", "keyword", or "none"

    def to_dict(self) -> dict[str, Any]:
        """Return this suggestion as a plain dictionary."""
        return {
            "column_name": self.column_name,
            "suggested_domain": self.suggested_domain,
            "suggested_metric": self.suggested_metric,
            "role": self.role,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass
class SuggestionResult:
    """The suggested mapping for every column of a dataset."""
    suggestions: list[ColumnSuggestion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return this suggestion as a plain dictionary."""
        return {"suggestions": [s.to_dict() for s in self.suggestions]}


_SYSTEM_PROMPT = (
    "You map raw data columns onto a fixed set of universal business domains. "
    "You never invent domains. You respond with JSON only, no prose, no markdown."
)


def _build_user_prompt(profile: DatasetProfile) -> str:
    domain_context = registry_as_prompt_context()
    cols = []
    for c in profile.columns:
        cols.append(
            {
                "column_name": c.column_name,
                "data_type": c.data_type,
                "distinct_count": c.distinct_count,
                "null_count": c.null_count,
                "is_numeric": c.is_numeric,
                "is_datetime": c.is_datetime,
                "is_identifier": c.is_identifier,
                "sample_values": c.sample_values,
            }
        )

    return (
        "Universal domains available:\n"
        f"{domain_context}\n\n"
        "For each column below, decide:\n"
        '- role: "entity" if it identifies a thing (ids, names), '
        '"metric" if it is a measurable value, "skip" if it carries no analytic value.\n'
        "- suggested_domain: one of "
        f"[{', '.join(sorted(_VALID_DOMAINS))}] or null if role is skip.\n"
        "- suggested_metric: a short snake_case metric name derived from the column.\n"
        "- confidence: 0.0 to 1.0.\n\n"
        "Columns:\n"
        f"{json.dumps(cols, ensure_ascii=False)}\n\n"
        'Respond with JSON exactly: {"suggestions": [{"column_name": ..., '
        '"role": ..., "suggested_domain": ..., "suggested_metric": ..., '
        '"confidence": ...}]}'
    )


def _keyword_suggestion(col: ColumnProfile) -> ColumnSuggestion:
    domain = match_domain_by_keyword(col.column_name)

    if col.is_identifier:
        return ColumnSuggestion(
            column_name=col.column_name,
            suggested_domain=domain.value if domain else None,
            suggested_metric=None,
            role="entity",
            confidence=0.5 if domain else 0.3,
            source="keyword",
        )

    if domain and col.is_numeric:
        return ColumnSuggestion(
            column_name=col.column_name,
            suggested_domain=domain.value,
            suggested_metric=_metric_name(col.column_name),
            role="metric",
            confidence=0.6,
            source="keyword",
        )

    if domain:
        return ColumnSuggestion(
            column_name=col.column_name,
            suggested_domain=domain.value,
            suggested_metric=_metric_name(col.column_name),
            role="metric",
            confidence=0.4,
            source="keyword",
        )

    # Numeric column with no keyword-matched domain: still a metric, just needs a
    # domain chosen at confirm. Skipping it silently drops usable data (this was
    # the cause of real metrics like signal_strength being dropped).
    if col.is_numeric:
        return ColumnSuggestion(
            column_name=col.column_name,
            suggested_domain=None,
            suggested_metric=_metric_name(col.column_name),
            role="metric",
            confidence=0.3,
            source="none",
        )

    return ColumnSuggestion(
        column_name=col.column_name,
        suggested_domain=None,
        suggested_metric=None,
        role="skip",
        confidence=0.2,
        source="none",
    )


def _metric_name(column_name: str) -> str:
    cleaned = column_name.strip().lower().replace(" ", "_").replace("-", "_")
    return "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")


def _keyword_fallback(profile: DatasetProfile) -> SuggestionResult:
    return SuggestionResult(suggestions=[_keyword_suggestion(c) for c in profile.columns])


def _call_llm(prompt: str) -> str | None:
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed, using keyword fallback")
        return None

    if not settings.anthropic_api_key:
        logger.warning("No Anthropic API key set, using keyword fallback")
        return None

    try:
        client = Anthropic(api_key=settings.anthropic_api_key)
        resp = client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts).strip()
    except Exception:  # noqa: BLE001
        logger.exception("LLM call failed, using keyword fallback")
        return None


def _parse_llm_response(raw: str, profile: DatasetProfile) -> SuggestionResult | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Could not parse LLM JSON response")
        return None

    # The model is instructed to return an object, but a list or a scalar is a
    # realistic malformed reply. Anything unexpected falls back to keywords
    # rather than raising, because a suggestion is advisory either way.
    if not isinstance(data, dict):
        logger.warning(
            "LLM returned %s instead of an object, using keyword fallback",
            type(data).__name__,
            extra={"response_type": type(data).__name__},
        )
        return None

    items = data.get("suggestions")
    if not isinstance(items, list):
        logger.warning("LLM response had no suggestions list, using keyword fallback")
        return None

    by_name = {c.column_name: c for c in profile.columns}
    suggestions: list[ColumnSuggestion] = []

    for item in items:
        if not isinstance(item, dict):
            logger.warning(
                "Ignoring malformed suggestion entry of type %s",
                type(item).__name__,
                extra={"entry_type": type(item).__name__},
            )
            continue
        name = item.get("column_name")
        if name not in by_name:
            continue
        domain = item.get("suggested_domain")
        if domain is not None and domain not in _VALID_DOMAINS:
            domain = None
        role = item.get("role", "skip")
        if role not in ("metric", "entity", "skip"):
            role = "skip"
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        suggestions.append(
            ColumnSuggestion(
                column_name=name,
                suggested_domain=domain,
                suggested_metric=item.get("suggested_metric"),
                role=role,
                confidence=max(0.0, min(1.0, confidence)),
                source="llm",
            )
        )

    suggested_names = {s.column_name for s in suggestions}
    for col in profile.columns:
        if col.column_name not in suggested_names:
            suggestions.append(_keyword_suggestion(col))

    return SuggestionResult(suggestions=suggestions)


def suggest_mappings(profile: DatasetProfile) -> SuggestionResult:
    """Suggest a domain and metric for every profiled column.

    Uses the LLM when enabled, falling back to keyword matching against the domain
    registry's aliases otherwise or on failure.

    Args:
        profile: The dataset profile to classify.

    Returns:
        One suggestion per column.
    """
    if not settings.llm_enabled:
        logger.info("LLM disabled, using keyword fallback")
        return _keyword_fallback(profile)

    prompt = _build_user_prompt(profile)
    raw = _call_llm(prompt)
    if raw is None:
        return _keyword_fallback(profile)

    parsed = _parse_llm_response(raw, profile)
    if parsed is None:
        return _keyword_fallback(profile)

    logger.info(
        "Generated LLM mapping suggestions",
        extra={"columns": len(parsed.suggestions), "file": profile.source_filename},
    )
    return parsed
