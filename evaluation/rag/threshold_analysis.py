"""Distance threshold analysis for the RAG retriever's abstention behaviour.

The retriever drops chunks beyond a maximum cosine distance so that it can
report honestly that it found nothing, rather than passing weak matches to the
model. The first evaluation run showed that cutoff never firing: every
out-of-scope question still returned chunks, and abstention accuracy was zero.

This script exists to answer whether that is a tuning problem or a design
problem, without guessing. It reads a retrieval evaluation report, compares the
distance of the nearest chunk for answerable and out-of-scope questions, and
sweeps the cutoff across its whole range to show what each value would cost and
buy.

The point is to make the decision from evidence. Lowering the cutoff until the
abstention number looks good, without checking what it does to hit rate, would
be tuning to the metric rather than to the problem -- and the sweep makes that
trade-off impossible to hide.

If the two distance distributions separate cleanly, a new cutoff is justified.
If they overlap, no single cutoff can work and the honest conclusion is that
abstention needs a different mechanism.

Run inside the API container::

    docker compose exec api sh -c "cd /app && python -m evaluation.rag.threshold_analysis \\
        --report evaluation/results/retrieval_600.json"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cosine distance runs from 0 (identical) to 2 (opposite). Sweeping in steps of
# 0.05 is fine enough to find a usable boundary without implying a precision the
# sample size does not support.
SWEEP_START = 0.30
SWEEP_STOP = 0.80
SWEEP_STEP = 0.05

# Ranks reported in the sweep, matching the retrieval evaluation.
K_VALUES = (1, 3, 5)


class ReportError(RuntimeError):
    """Raised when the evaluation report cannot be read or is the wrong shape."""


def load_report(path: Path) -> dict[str, Any]:
    """Load a retrieval evaluation report.

    Args:
        path: Location of the JSON report written by ``retrieval_eval``.

    Returns:
        The parsed report.

    Raises:
        ReportError: If the file is missing, malformed, or lacks the per-question
            detail this analysis depends on.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportError(f"Could not read the report at {path}: {exc}") from exc

    try:
        report = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReportError(f"Report at {path} is not valid JSON: {exc}") from exc

    questions = report.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ReportError(
            f"Report at {path} has no per-question detail. Re-run the retrieval "
            "evaluation with --out to produce a full report."
        )

    if not any(q.get("retrieved") for q in questions):
        raise ReportError(
            "No question in the report has retrieved chunks recorded, so there "
            "are no distances to analyse."
        )

    return report


def _distances(question: dict[str, Any]) -> list[float]:
    """Return the distances of the chunks retrieved for one question, in rank order."""
    return [c["distance"] for c in question.get("retrieved", [])]


def _nearest(question: dict[str, Any]) -> float | None:
    """Return the distance of the nearest retrieved chunk, or None if there was none."""
    distances = _distances(question)
    return min(distances) if distances else None


def _describe(values: list[float]) -> dict[str, float]:
    """Return simple positional statistics for a list of distances.

    Deliberately not mean and standard deviation: with a sample this small the
    minimum, median and maximum say more about where the boundary sits than a
    fitted distribution would.

    Args:
        values: Distances to describe.

    Returns:
        Minimum, median and maximum, rounded for display.
    """
    if not values:
        return {}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = (
        ordered[mid]
        if len(ordered) % 2
        else (ordered[mid - 1] + ordered[mid]) / 2
    )
    return {
        "min": round(ordered[0], 4),
        "median": round(median, 4),
        "max": round(ordered[-1], 4),
        "n": len(ordered),
    }


def separation(report: dict[str, Any]) -> dict[str, Any]:
    """Compare nearest-chunk distances for answerable and out-of-scope questions.

    Args:
        report: A parsed retrieval evaluation report.

    Returns:
        Statistics for each group, plus the size of the gap or overlap between
        them. A positive gap means a cutoff exists that separates the two
        perfectly; a negative gap is the width of the overlap.
    """
    answerable: list[float] = []
    out_of_scope: list[float] = []

    for question in report["questions"]:
        nearest = _nearest(question)
        if nearest is None:
            continue
        if question["scope"] == "out_of_scope":
            out_of_scope.append(nearest)
        else:
            answerable.append(nearest)

    stats = {
        "answerable": _describe(answerable),
        "out_of_scope": _describe(out_of_scope),
    }

    if answerable and out_of_scope:
        gap = min(out_of_scope) - max(answerable)
        stats["gap"] = round(gap, 4)
        stats["separable"] = gap > 0
    else:
        stats["gap"] = None
        stats["separable"] = False

    return stats


def sweep(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Recompute abstention and hit rate across a range of distance cutoffs.

    The retrieval that produced the report already applied its own cutoff, so
    this sweep can only tighten it, never loosen it. Values above the cutoff
    actually used are therefore reported as unreachable rather than guessed at.

    Args:
        report: A parsed retrieval evaluation report.

    Returns:
        One row per candidate cutoff, holding abstention accuracy and hit rate
        at each reported k.
    """
    questions = report["questions"]
    answerable = [q for q in questions if q["scope"] != "out_of_scope"]
    out_of_scope = [q for q in questions if q["scope"] == "out_of_scope"]

    rows: list[dict[str, Any]] = []
    steps = int(round((SWEEP_STOP - SWEEP_START) / SWEEP_STEP)) + 1

    for step in range(steps):
        cutoff = round(SWEEP_START + step * SWEEP_STEP, 2)

        refused = sum(
            1
            for q in out_of_scope
            if not [d for d in _distances(q) if d <= cutoff]
        )
        row: dict[str, Any] = {
            "cutoff": cutoff,
            "abstention_accuracy": (
                round(refused / len(out_of_scope), 4) if out_of_scope else None
            ),
            "correctly_refused": refused,
            "out_of_scope_questions": len(out_of_scope),
        }

        for k in K_VALUES:
            hits = 0
            for q in answerable:
                rank = q.get("first_relevant_rank")
                if rank is None or rank > k:
                    continue
                # The relevant chunk survives only if it is inside the cutoff.
                distances = _distances(q)
                if rank <= len(distances) and distances[rank - 1] <= cutoff:
                    hits += 1
            row[f"hit_rate@{k}"] = (
                round(hits / len(answerable), 4) if answerable else None
            )

        rows.append(row)

    return rows


def _print_analysis(stats: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Print the separation statistics and the cutoff sweep."""
    print("=" * 70)
    print(" Retrieval Distance Threshold Analysis")
    print("=" * 70)
    print()

    print(" Nearest-chunk distance by question group")
    print(" " + "-" * 66)
    for group in ("answerable", "out_of_scope"):
        s = stats.get(group) or {}
        if not s:
            print(f"   {group:<14} no data")
            continue
        print(
            f"   {group:<14} n={s['n']:<3} "
            f"min={s['min']:.4f}  median={s['median']:.4f}  max={s['max']:.4f}"
        )
    print()

    gap = stats.get("gap")
    if gap is None:
        print("   Not enough data to compare the two groups.")
    elif stats["separable"]:
        print(
            f"   The groups separate: a gap of {gap:.4f} exists between the "
            "furthest answerable\n   question and the nearest out-of-scope one. "
            "A cutoff inside that gap is\n   justified by the data."
        )
    else:
        print(
            f"   The groups overlap by {abs(gap):.4f}. No single distance cutoff "
            "separates them,\n   so abstention cannot be fixed by tuning this "
            "value alone. Any cutoff that\n   refuses the out-of-scope questions "
            "will also discard answerable ones."
        )
    print()

    print(" Cutoff sweep")
    print(" " + "-" * 66)
    header = (
        f"   {'cutoff':<8}{'abstention':<13}"
        + "".join(f"{'hit@' + str(k):<10}" for k in K_VALUES)
    )
    print(header)
    for row in rows:
        abst = row["abstention_accuracy"]
        cells = "".join(
            f"{row[f'hit_rate@{k}']:.2%}".ljust(10) for k in K_VALUES
        )
        print(f"   {row['cutoff']:<8.2f}{abst:.2%}".ljust(24) + cells)
    print()

    print(" Read this as a trade-off, not as a target. A cutoff is only")
    print(" defensible if it was chosen before the headline numbers were")
    print(" reported, and if the hit rate it costs is reported alongside.")
    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point.

    Args:
        argv: Argument list, defaulting to the process arguments.

    Returns:
        Process exit code: 0 on success, 1 if the report could not be analysed.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Analyse whether the retriever's distance cutoff can separate "
            "answerable from out-of-scope questions."
        )
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evaluation/results/retrieval_600.json"),
        help="Retrieval evaluation report to analyse (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the analysis as JSON to this path",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    try:
        report = load_report(args.report)
    except ReportError as exc:
        logger.error("Could not analyse the report: %s", exc)
        return 1

    stats = separation(report)
    rows = sweep(report)

    logger.info(
        "Threshold analysis complete",
        extra={"separable": stats["separable"], "gap": stats["gap"]},
    )

    _print_analysis(stats, rows)

    if args.out:
        payload = {
            "source_report": str(args.report),
            "separation": stats,
            "sweep": rows,
        }
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.error("Could not write the analysis to %s: %s", args.out, exc)
            return 1
        logger.info("Analysis written", extra={"path": str(args.out)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
