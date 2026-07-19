"""Synthetic data generators for the supported demo industries.

Produces realistic, correlated operational data for manufacturing, telecom,
aerospace, and education. Correlation matters: values carry seasonality and
trend rather than being independent noise, so the downstream analytics, ML, and
intelligence layers have real structure to find. Every generator is seeded, so
the same configuration always produces the same data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from ops_common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IndustrySpec:
    """Describes one supported industry and how to generate data for it."""
    key: str
    entity_prefix: str
    entity_count: int
    builder: str
    description: str = ""


@dataclass
class GeneratorConfig:
    """Generation parameters: horizon, seed, and start date."""
    days: int = 90
    seed: int = 42
    start_date: datetime = field(default_factory=lambda: datetime(2025, 1, 1))


# ============================================================
# Shared helpers
# ============================================================


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _date_range(config: GeneratorConfig) -> list[datetime]:
    return [config.start_date + timedelta(days=d) for d in range(config.days)]


def _timestamp(date: datetime, rng: np.random.Generator) -> str:
    offset = timedelta(
        hours=int(rng.integers(6, 20)),
        minutes=int(rng.integers(0, 60)),
        seconds=int(rng.integers(0, 60)),
    )
    return (date + offset).strftime("%Y-%m-%d %H:%M:%S")


def _seasonal(day_index: int, period: int = 7, amplitude: float = 0.15) -> float:
    return 1.0 + amplitude * np.sin(2 * np.pi * day_index / period)


def _clip_positive(value: float) -> float:
    return max(0.0, float(value))


# ============================================================
# Manufacturing
# ============================================================


def build_manufacturing(
    config: GeneratorConfig, entity_count: int, prefix: str
) -> pd.DataFrame:
    """Generate a manufacturing dataset (machines, output, defects, downtime).

    Args:
        config: Generation parameters.
        entity_count: Number of distinct machines to simulate.
        prefix: Identifier prefix for generated entities.

    Returns:
        One row per entity per day.
    """
    rng = _rng(config.seed)
    dates = _date_range(config)
    rows = []

    base_output = rng.uniform(800, 1200, size=entity_count)
    base_quality = rng.uniform(0.01, 0.05, size=entity_count)

    for d_idx, date in enumerate(dates):
        season = _seasonal(d_idx)
        for m in range(entity_count):
            machine_id = f"{prefix}-{m + 1:03d}"
            units = _clip_positive(base_output[m] * season * rng.normal(1.0, 0.08))
            downtime = _clip_positive(
                rng.exponential(20) * (1.2 if season < 1 else 0.8)
            )
            defect_rate = min(
                0.25, _clip_positive(base_quality[m] + rng.normal(0, 0.01))
            )
            defects = _clip_positive(units * defect_rate)
            energy = _clip_positive(units * rng.uniform(0.8, 1.2))
            cost = _clip_positive(units * rng.uniform(4, 7) + downtime * 50)

            rows.append(
                {
                    "machine_id": machine_id,
                    "date": _timestamp(date, rng),
                    "units_produced": round(units, 1),
                    "downtime_minutes": round(downtime, 1),
                    "defect_count": round(defects, 1),
                    "energy_kwh": round(energy, 1),
                    "operating_cost": round(cost, 2),
                    "operator_count": int(rng.integers(2, 6)),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Telecom
# ============================================================


def build_telecom(
    config: GeneratorConfig, entity_count: int, prefix: str
) -> pd.DataFrame:
    """Generate a telecom dataset (towers, traffic, outages, subscribers).

    Args:
        config: Generation parameters.
        entity_count: Number of distinct towers to simulate.
        prefix: Identifier prefix for generated entities.

    Returns:
        One row per entity per day.
    """
    rng = _rng(config.seed + 1)
    dates = _date_range(config)
    rows = []

    base_subs = rng.uniform(5000, 20000, size=entity_count)

    for d_idx, date in enumerate(dates):
        season = _seasonal(d_idx, period=30, amplitude=0.1)
        for t in range(entity_count):
            tower_id = f"{prefix}-{t + 1:03d}"
            subscribers = _clip_positive(base_subs[t] * season * rng.normal(1.0, 0.03))
            data_gb = _clip_positive(subscribers * rng.uniform(0.5, 1.5))
            dropped_calls = _clip_positive(subscribers * rng.uniform(0.001, 0.01))
            sla_breaches = int(_clip_positive(rng.poisson(2)))
            outage_minutes = _clip_positive(rng.exponential(10))
            revenue = _clip_positive(subscribers * rng.uniform(8, 15))

            rows.append(
                {
                    "tower_id": tower_id,
                    "date": _timestamp(date, rng),
                    "active_subscribers": int(subscribers),
                    "data_usage_gb": round(data_gb, 1),
                    "dropped_calls": round(dropped_calls, 1),
                    "sla_breaches": sla_breaches,
                    "outage_minutes": round(outage_minutes, 1),
                    "bill_amount": round(revenue, 2),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Aerospace
# ============================================================


def build_aerospace(
    config: GeneratorConfig, entity_count: int, prefix: str
) -> pd.DataFrame:
    """Generate an aerospace dataset (aircraft, flight hours, maintenance).

    Args:
        config: Generation parameters.
        entity_count: Number of distinct aircraft to simulate.
        prefix: Identifier prefix for generated entities.

    Returns:
        One row per entity per day.
    """
    rng = _rng(config.seed + 2)
    dates = _date_range(config)
    rows = []

    for _d_idx, date in enumerate(dates):
        for a in range(entity_count):
            aircraft_id = f"{prefix}-{a + 1:03d}"
            flight_hours = _clip_positive(rng.uniform(4, 14))
            sorties = int(_clip_positive(rng.integers(1, 5)))
            inspection_failures = int(_clip_positive(rng.poisson(0.5)))
            service_cycles = int(_clip_positive(rng.integers(0, 3)))
            spare_parts_used = int(_clip_positive(rng.poisson(3)))
            maintenance_cost = _clip_positive(
                flight_hours * rng.uniform(200, 400) + service_cycles * 1500
            )
            crew_size = int(rng.integers(2, 8))

            rows.append(
                {
                    "aircraft_id": aircraft_id,
                    "date": _timestamp(date, rng),
                    "flight_hours": round(flight_hours, 1),
                    "sorties": sorties,
                    "inspection_failures": inspection_failures,
                    "service_cycles": service_cycles,
                    "spare_parts_used": spare_parts_used,
                    "maintenance_cost": round(maintenance_cost, 2),
                    "crew_size": crew_size,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Education
# ============================================================


def build_education(
    config: GeneratorConfig, entity_count: int, prefix: str
) -> pd.DataFrame:
    """Generate an education dataset (courses, enrolment, fees, outcomes).

    Sampled at a coarser interval than the other industries, since academic data is
    not naturally daily.

    Args:
        config: Generation parameters.
        entity_count: Number of distinct courses to simulate.
        prefix: Identifier prefix for generated entities.

    Returns:
        One row per entity per sampled period.
    """
    rng = _rng(config.seed + 3)
    dates = _date_range(config)[:: max(1, config.days // 30)]
    rows = []

    base_enroll = rng.uniform(20, 60, size=entity_count)

    for date in dates:
        for c in range(entity_count):
            classroom_id = f"{prefix}-{c + 1:03d}"
            enrolled = int(_clip_positive(base_enroll[c]))
            attendance = int(_clip_positive(enrolled * rng.uniform(0.7, 0.98)))
            avg_grade = round(_clip_positive(rng.normal(72, 10)), 1)
            failures = int(_clip_positive(enrolled * rng.uniform(0.02, 0.15)))
            fee_paid = _clip_positive(enrolled * rng.uniform(200, 500))
            staff_count = int(rng.integers(1, 4))

            rows.append(
                {
                    "classroom_id": classroom_id,
                    "date": _timestamp(date, rng),
                    "students_enrolled": enrolled,
                    "attendance_count": attendance,
                    "average_grade": avg_grade,
                    "failure_count": failures,
                    "fee_paid": round(fee_paid, 2),
                    "staff_count": staff_count,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Registry of industries
# ============================================================

_BUILDERS = {
    "manufacturing": build_manufacturing,
    "telecom": build_telecom,
    "aerospace": build_aerospace,
    "education": build_education,
}

INDUSTRY_SPECS: dict[str, IndustrySpec] = {
    "manufacturing": IndustrySpec(
        "manufacturing",
        "MCH",
        8,
        "build_manufacturing",
        "Factory machines with output, downtime, defects.",
    ),
    "telecom": IndustrySpec(
        "telecom",
        "TWR",
        10,
        "build_telecom",
        "Cell towers with subscribers, usage, SLA breaches.",
    ),
    "aerospace": IndustrySpec(
        "aerospace",
        "ACFT",
        6,
        "build_aerospace",
        "Aircraft with flight hours, inspections, maintenance.",
    ),
    "education": IndustrySpec(
        "education",
        "CLS",
        12,
        "build_education",
        "Classrooms with enrollment, attendance, grades, fees.",
    ),
}


def generate(industry: str, config: GeneratorConfig | None = None) -> pd.DataFrame:
    """Generate a dataset for one industry.

    Args:
        industry: Industry key to generate.
        config: Generation parameters; defaults are used when omitted.

    Returns:
        The generated dataset.

    Raises:
        ValueError: If the industry is not one of the supported keys.
    """
    industry = industry.lower()
    if industry not in _BUILDERS:
        raise ValueError(
            f"Unknown industry {industry!r}. Available: {sorted(_BUILDERS)}"
        )
    config = config or GeneratorConfig()
    spec = INDUSTRY_SPECS[industry]
    builder = _BUILDERS[industry]
    df = builder(config, spec.entity_count, spec.entity_prefix)
    logger.info(
        "Generated industry data",
        extra={"industry": industry, "rows": len(df), "columns": len(df.columns)},
    )
    return df


def generate_to_csv(
    industry: str,
    out_dir: str | Path,
    config: GeneratorConfig | None = None,
) -> Path:
    """Generate an industry dataset and write it to a CSV file.

    Args:
        industry: Industry key to generate.
        out_dir: Directory to write into; created if missing.
        config: Generation parameters; defaults are used when omitted.

    Returns:
        Path to the written CSV.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate(industry, config)
    out_path = out_dir / f"{industry}_sample.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote sample CSV", extra={"industry": industry, "path": str(out_path)})
    return out_path
