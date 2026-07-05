# Cross-domain inference engine (Level 2). Loads the master knowledge graph,
# restricts to the active-domain subgraph, lights edges only where both endpoints
# carry a corroborating ML signal, and produces ranked root-cause + impact paths.
# Mirror pairs (A→B and B→A both lit) are merged into one feedback-loop insight.

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

# Named strength → numeric, for ranking impacts (higher = stronger dependency).
_STRENGTH_VALUE = {"critical": 5, "strong": 4, "medium": 3, "weak": 2, "very_weak": 1}

# How far the engine walks outward from a triggered domain. Beyond 2 hops the
# business story stops being believable, so this is a deliberate cap, not a limit.
MAX_HOPS = 2

# A domain must clear this signal strength (0–1) to count as "lit" / corroborating.
SIGNAL_THRESHOLD = 0.20

_GRAPH_PATH = os.getenv("OPS_RELATIONSHIPS_PATH", "/app/data-hub/relationships.json")


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    strength: str
    effect: str
    label: str

    @property
    def weight(self) -> int:
        return _STRENGTH_VALUE.get(self.strength, 1)


@lru_cache(maxsize=1)
def load_graph(path: str = _GRAPH_PATH) -> tuple[Edge, ...]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    edges = tuple(
        Edge(
            source=str(e["source"]).lower(),
            target=str(e["target"]).lower(),
            strength=str(e.get("strength", "weak")).lower(),
            effect=str(e.get("effect", "negative")).lower(),
            label=str(e.get("label", "")),
        )
        for e in data.get("edges", [])
    )
    return edges


def _adjacency(edges: tuple[Edge, ...]) -> dict[str, list[Edge]]:
    adj: dict[str, list[Edge]] = {}
    for e in edges:
        adj.setdefault(e.source, []).append(e)
    return adj


# ---------------------------------------------------------------------------
# ML signal model
# ---------------------------------------------------------------------------

@dataclass
class DomainSignal:
    """Normalized 0–1 signal per domain, derived from Level 1 ML outputs.

    strength   — overall how much this domain is 'acting up' (drives triggers)
    direction  — 'down' | 'up' | 'flat' from the forecast trend
    risk       — max risk score seen (assets/maintenance)
    anomalies  — flagged anomaly count
    top_metric — the metric most responsible, for phrasing
    """
    domain: str
    strength: float = 0.0
    direction: str = "flat"
    risk: float = 0.0
    anomalies: int = 0
    high_anomalies: int = 0
    top_metric: str | None = None

    @property
    def is_lit(self) -> bool:
        return self.strength >= SIGNAL_THRESHOLD


def build_signals(
    forecasts: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    risks: list[dict[str, Any]],
) -> dict[str, DomainSignal]:
    """Collapse the three Level 1 outputs into one signal per active domain."""
    signals: dict[str, DomainSignal] = {}

    def _ensure(domain: str) -> DomainSignal:
        d = domain.lower()
        if d not in signals:
            signals[d] = DomainSignal(domain=d)
        return signals[d]

    # Forecast → direction + magnitude of expected move (per domain, worst metric).
    fc_by_dm: dict[tuple, list[dict]] = {}
    for f in forecasts:
        key = (str(f["domain"]).lower(), f["metric_name"])
        fc_by_dm.setdefault(key, []).append(f)

    fc_move: dict[str, float] = {}
    for (domain, metric), series in fc_by_dm.items():
        series = sorted(series, key=lambda r: r["forecast_date"])
        vals = [r.get("forecast_value") for r in series if r.get("forecast_value") is not None]
        if len(vals) < 2:
            continue
        first, last = vals[0], vals[-1]
        denom = abs(first) if abs(first) > 1e-9 else 1.0
        pct = (last - first) / denom  # signed fractional change over horizon
        sig = _ensure(domain)
        if abs(pct) > abs(fc_move.get(domain, 0.0)):
            fc_move[domain] = pct
            sig.direction = "down" if pct < -1e-6 else ("up" if pct > 1e-6 else "flat")
            sig.top_metric = metric

    # Anomalies → count + high-severity emphasis.
    for a in anomalies:
        sig = _ensure(str(a["domain"]).lower())
        sig.anomalies += 1
        if a.get("severity") == "high":
            sig.high_anomalies += 1
        if sig.top_metric is None:
            sig.top_metric = a.get("metric_name")

    # Risk → max normalized score.
    for r in risks:
        sig = _ensure(str(r["domain"]).lower())
        score = float(r.get("risk_score") or 0) / 100.0
        sig.risk = max(sig.risk, score)

    # Composite strength: blend forecast move, risk, and anomaly pressure into 0–1.
    for domain, sig in signals.items():
        move_c = min(1.0, abs(fc_move.get(domain, 0.0)))
        anom_c = min(1.0, sig.anomalies / 5.0)
        high_c = min(1.0, sig.high_anomalies / 2.0)
        sig.strength = round(
            min(1.0, 0.40 * sig.risk + 0.30 * move_c + 0.20 * anom_c + 0.10 * high_c),
            4,
        )
    return signals


# ---------------------------------------------------------------------------
# Traversal + insight construction
# ---------------------------------------------------------------------------

@dataclass
class ImpactPath:
    """One lit edge from a root domain toward an impacted domain."""
    target: str
    strength: str
    effect: str
    label: str
    weight: int
    hop: int
    target_signal: float


@dataclass
class Insight:
    root: str
    root_signal: float
    root_direction: str
    root_metric: str | None
    impacts: list[ImpactPath] = field(default_factory=list)
    score: float = 0.0
    # Feedback-loop fields (populated only when two domains reinforce each other).
    is_loop: bool = False
    loop_partner: str | None = None
    loop_forward_label: str | None = None
    loop_reverse_label: str | None = None


def _score_insight(root: DomainSignal, impacts: list[ImpactPath]) -> float:
    # A root's importance = its own signal plus corroborated, strength-weighted impact.
    impact_sum = sum(p.weight * p.target_signal for p in impacts)
    return round(root.strength * 5.0 + impact_sum, 4)


def infer(
    signals: dict[str, DomainSignal],
    active_domains: set[str] | None = None,
    max_hops: int = MAX_HOPS,
) -> list[Insight]:
    """Traverse the active subgraph from each lit domain and build ranked insights."""
    edges = load_graph()
    adj = _adjacency(edges)

    active = {d.lower() for d in (active_domains or set(signals.keys()))}
    lit = {d for d, s in signals.items() if s.is_lit and d in active}

    insights: list[Insight] = []
    for root in lit:
        root_sig = signals[root]
        visited: set[str] = {root}
        impacts: list[ImpactPath] = []

        # Breadth-limited walk: only step into active domains that are themselves lit.
        frontier: list[tuple[str, int]] = [(root, 0)]
        while frontier:
            node, hop = frontier.pop(0)
            if hop >= max_hops:
                continue
            for edge in adj.get(node, []):
                tgt = edge.target
                if tgt not in active or tgt in visited:
                    continue
                tgt_sig = signals.get(tgt)
                if tgt_sig is None or not tgt_sig.is_lit:
                    continue
                visited.add(tgt)
                impacts.append(
                    ImpactPath(
                        target=tgt,
                        strength=edge.strength,
                        effect=edge.effect,
                        label=edge.label,
                        weight=edge.weight,
                        hop=hop + 1,
                        target_signal=tgt_sig.strength,
                    )
                )
                frontier.append((tgt, hop + 1))

        if not impacts:
            continue  # a root with no corroborated downstream effect isn't a story

        # Rank impacts by dependency strength × corroborating signal.
        impacts.sort(key=lambda p: (p.weight * p.target_signal, p.weight), reverse=True)

        insight = Insight(
            root=root,
            root_signal=root_sig.strength,
            root_direction=root_sig.direction,
            root_metric=root_sig.top_metric,
            impacts=impacts,
        )
        insight.score = _score_insight(root_sig, impacts)
        insights.append(insight)

    # Collapse mirror pairs (A→B and B→A) into single feedback-loop insights.
    insights = _merge_feedback_loops(insights)

    insights.sort(key=lambda i: i.score, reverse=True)
    return insights


# ---------------------------------------------------------------------------
# Feedback-loop merging
# ---------------------------------------------------------------------------

def _direct_impact_to(insight: Insight, target: str) -> ImpactPath | None:
    # A first-hop impact from this insight's root to the given target, if any.
    for p in insight.impacts:
        if p.target == target and p.hop == 1:
            return p
    return None


def _merge_feedback_loops(insights: list[Insight]) -> list[Insight]:
    """Detect A→B + B→A mirror pairs and merge each into one loop insight.

    A reinforcing loop is a stronger finding than two one-way effects, so the pair
    becomes a single insight: the higher-scoring root leads, its partner's reverse
    edge is captured as the loop's return path. Non-loop insights pass through
    unchanged.
    """
    by_root = {ins.root: ins for ins in insights}
    consumed: set[str] = set()
    merged: list[Insight] = []

    for ins in insights:
        if ins.root in consumed:
            continue

        loop_ins = None
        for p in ins.impacts:
            if p.hop != 1:
                continue
            partner = by_root.get(p.target)
            if partner is None or partner.root in consumed:
                continue
            # Does the partner point back at us on a first hop? That's the loop.
            reverse = _direct_impact_to(partner, ins.root)
            if reverse is None:
                continue
            # Lead with the higher-scoring side for a natural root cause.
            primary, secondary = (ins, partner) if ins.score >= partner.score else (partner, ins)
            fwd = _direct_impact_to(primary, secondary.root)
            rev = _direct_impact_to(secondary, primary.root)

            loop_ins = Insight(
                root=primary.root,
                root_signal=primary.root_signal,
                root_direction=primary.root_direction,
                root_metric=primary.root_metric,
                impacts=primary.impacts,
                score=max(primary.score, secondary.score),
                is_loop=True,
                loop_partner=secondary.root,
                loop_forward_label=fwd.label if fwd else None,
                loop_reverse_label=rev.label if rev else None,
            )
            consumed.add(ins.root)
            consumed.add(partner.root)
            break

        merged.append(loop_ins if loop_ins is not None else ins)

    return merged


# ---------------------------------------------------------------------------
# Serialization (consumed by the API / translator)
# ---------------------------------------------------------------------------

def insight_to_dict(ins: Insight) -> dict[str, Any]:
    return {
        "root": ins.root,
        "root_signal": ins.root_signal,
        "root_direction": ins.root_direction,
        "root_metric": ins.root_metric,
        "score": ins.score,
        "is_loop": ins.is_loop,
        "loop_partner": ins.loop_partner,
        "loop_forward_label": ins.loop_forward_label,
        "loop_reverse_label": ins.loop_reverse_label,
        "impacts": [
            {
                "target": p.target,
                "strength": p.strength,
                "effect": p.effect,
                "label": p.label,
                "weight": p.weight,
                "hop": p.hop,
                "target_signal": p.target_signal,
            }
            for p in ins.impacts
        ],
    }


def run_inference(
    forecasts: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    risks: list[dict[str, Any]],
    active_domains: set[str] | None = None,
) -> list[dict[str, Any]]:
    """End-to-end entry point: raw Level 1 outputs → ranked insight dicts."""
    signals = build_signals(forecasts, anomalies, risks)
    if active_domains is None:
        active_domains = set(signals.keys())
    insights = infer(signals, active_domains=active_domains)
    return [insight_to_dict(i) for i in insights]