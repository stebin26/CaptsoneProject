"""The Operational Risk Index -- a single relative number for the whole operation.

This is deliberately a pure function with no database and no framework imports,
so it can be unit-tested with a plain list of numbers and so the formula lives
in exactly one place. The endpoint calls it; a future drill-through calls the
same function; the two can never disagree.

What it is NOT
--------------
It is not a probability of failure. The per-domain scores it averages are
themselves relative degradation rankings, not calibrated probabilities, so the
composite inherits that. The UI labels it "Operational Risk Index (relative)"
for exactly this reason -- the same honesty the rest of the platform applies to
"degradation risk, not failure probability".

The formula
-----------
A blend of the mean and the maximum across active domains:

    index = mean_weight * mean(scores) + max_weight * max(scores)

Pure mean lets one healthy domain mask a burning one -- a critical maintenance
failure drowns in good finance numbers. Pure max lets a single domain dictate
the entire score. The blend keeps the baseline (mean) while staying alert to
the worst case (max).

The 70/30 default is a starting point, not a law. It is configurable because
different organisations have different risk appetites -- a hospital weights the
worst case harder than a warehouse does. That configurability is the honest
answer to "why 70/30 and not 60/40": because the right split depends on who is
asking, so it is a setting, not a constant baked into the code.
"""

from __future__ import annotations

from dataclasses import dataclass

# Defaults. Override per deployment rather than editing these.
DEFAULT_MEAN_WEIGHT = 0.7
DEFAULT_MAX_WEIGHT = 0.3

# Band thresholds on the 0-100 scale.
BAND_ELEVATED = 34
BAND_HIGH = 67


@dataclass(frozen=True)
class RiskIndex:
    """The computed risk index and the components behind it.

    Carries the rounded value and band for display, plus the mean, peak, and peak
    domain so a drill-through can show why the number is what it is.
    """
    value: int  # 0-100, rounded
    band: str  # "low" | "elevated" | "high"
    label: str  # human label for the band
    domain_count: int  # how many active domains fed the number
    mean: float  # the components, exposed for the drill-through
    peak: float
    peak_domain: str | None


def band_for(value: float) -> tuple[str, str]:
    """Map an index value to its band name and display label.

    Args:
        value: The index value on the 0-100 scale.

    Returns:
        A ``(band, label)`` pair.
    """
    if value >= BAND_HIGH:
        return "high", "High"
    if value >= BAND_ELEVATED:
        return "elevated", "Elevated"
    return "low", "Low"


def compute(
    domain_scores: dict[str, float],
    mean_weight: float = DEFAULT_MEAN_WEIGHT,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> RiskIndex:
    """Blend per-domain risk scores into one 0-100 index.

    domain_scores maps an active domain to its representative risk score
    (0-100). Domains with no score are simply absent from the dict -- an
    absent domain does not pull the index down, because "we have no signal"
    is not the same as "this domain is healthy".
    """
    scores = [s for s in domain_scores.values() if s is not None]

    if not scores:
        return RiskIndex(
            value=0,
            band="low",
            label="No signal",
            domain_count=0,
            mean=0.0,
            peak=0.0,
            peak_domain=None,
        )

    total = mean_weight + max_weight
    mw = mean_weight / total  # normalise, so any pair of weights works
    xw = max_weight / total

    mean_score = sum(scores) / len(scores)
    peak_score = max(scores)
    peak_domain = max(domain_scores, key=lambda d: domain_scores[d])

    raw = mw * mean_score + xw * peak_score
    value = int(round(max(0.0, min(100.0, raw))))
    band, label = band_for(value)

    return RiskIndex(
        value=value,
        band=band,
        label=label,
        domain_count=len(scores),
        mean=round(mean_score, 1),
        peak=round(peak_score, 1),
        peak_domain=peak_domain,
    )
