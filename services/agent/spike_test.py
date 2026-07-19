# services/agent/spike_test.py
"""SPIKE TEST — throwaway. Delete after the tool-calling decision is made.

Purpose: answer ONE question before we build the rest of Phase 4 —
    "Can llama3.2:3b reliably call the analytics tools through our graph?"

It runs a fixed battery of questions whose CORRECT first tool is known in
advance, then reports how often the model:
  - called a tool at all (vs. hallucinating an answer with no data)
  - called the RIGHT first tool
  - produced malformed output that our parser had to rescue

This isolates ONE variable — the model's tool-calling — so nothing else is in
play: no planner, no memory, no auth, no API, no dashboard. Just terminal.

PASS BAR (decide before running, so we don't move the goalposts):
  - correct-first-tool >= 8/10
  - zero hard crashes (our safety nets should catch all model misbehavior)
If it passes -> build the full agent on this foundation.
If it fails  -> weigh a larger tool-tuned model (7-8B) before going further.

Run from repo root inside the api container (where ops_common + api_app import):
    python -m services.agent.spike_test
    python -m services.agent.spike_test --dataset 542
    python -m services.agent.spike_test --list           # show test cases only
    python -m services.agent.spike_test --only 3         # run a single case
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

from .graph import run_once
from .llm import get_llm
from .tools.analytics_tool import (
    ANALYTICS_TOOL_FUNCTIONS,
    ANALYTICS_TOOL_SCHEMAS,
)

# ============================================================
# Test battery — each case knows its expected FIRST tool
# ============================================================
# We only grade the first tool because that is the routing decision under test.
# 'accept' lists tools that are also defensible as a first move, so we don't
# punish reasonable alternatives (e.g. going straight to features).


@dataclass
class Case:
    """One probe question with its ideal and acceptable first tools."""
    question: str
    expect: str  # the ideal first tool
    accept: tuple[str, ...]  # also-acceptable first tools


CASES: list[Case] = [
    Case(
        "What does this dataset contain?",
        expect="analytics_overview",
        accept=(),
    ),
    Case(
        "Give me an overall health check of the operation.",
        expect="analytics_overview",
        accept=(),
    ),
    Case(
        "Why is production decreasing?",
        expect="analytics_trend",
        accept=("analytics_overview",),
    ),
    Case(
        "Is downtime rising over time?",
        expect="analytics_trend",
        accept=("analytics_overview",),
    ),
    Case(
        "Which machine is degrading the most?",
        expect="analytics_features",
        accept=("analytics_overview",),
    ),
    Case(
        "Which entity stands out as the worst performer?",
        expect="analytics_features",
        accept=("analytics_overview",),
    ),
    Case(
        "What domains are present in this data?",
        expect="analytics_overview",
        accept=(),
    ),
    Case(
        "Show me the trend of quality metrics.",
        expect="analytics_trend",
        accept=("analytics_overview",),
    ),
    Case(
        "What is the average value of each metric?",
        expect="analytics_overview",
        accept=(),
    ),
    Case(
        "Rank the assets by how strongly they are trending.",
        expect="analytics_features",
        accept=("analytics_overview",),
    ),
]


# ============================================================
# Grading
# ============================================================


@dataclass
class Outcome:
    """The result of running one probe: which tool ran and whether it was right."""
    case: Case
    first_tool: str | None
    correct: bool
    acceptable: bool
    called_any_tool: bool
    steps: int
    answer: str
    tools_sequence: list[str]


def _first_tool(evidence: list[dict]) -> str | None:
    for e in evidence:
        if e.get("tool"):
            return e["tool"]
    return None


def _grade(case: Case, result: dict) -> Outcome:
    evidence = result.get("evidence", [])
    sequence = [e.get("tool", "?") for e in evidence]
    first = _first_tool(evidence)
    correct = first == case.expect
    acceptable = correct or (first in case.accept)
    return Outcome(
        case=case,
        first_tool=first,
        correct=correct,
        acceptable=acceptable,
        called_any_tool=first is not None,
        steps=result.get("steps", 0),
        answer=result.get("answer", ""),
        tools_sequence=sequence,
    )


# ============================================================
# Runner
# ============================================================


def _run_case(case: Case, dataset_id: int) -> Outcome:
    result = run_once(
        question=case.question,
        tool_schemas=ANALYTICS_TOOL_SCHEMAS,
        tool_registry=ANALYTICS_TOOL_FUNCTIONS,
        dataset_hint=dataset_id,
    )
    return _grade(case, result)


def _print_case(idx: int, outcome: Outcome) -> None:
    mark = "PASS" if outcome.acceptable else "FAIL"
    exact = (
        "exact" if outcome.correct else ("accepted" if outcome.acceptable else "wrong")
    )
    print(f"\n[{idx:02d}] {mark} ({exact})")
    print(f"     Q: {outcome.case.question}")
    print(f"     expected first tool : {outcome.case.expect}")
    print(f"     actual  first tool  : {outcome.first_tool}")
    print(
        f"     tool sequence       : {' -> '.join(outcome.tools_sequence) or '(none)'}"
    )
    print(f"     steps               : {outcome.steps}")
    print(f"     answer              : {_truncate(outcome.answer, 160)}")


def _truncate(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _summary(outcomes: list[Outcome]) -> bool:
    total = len(outcomes)
    exact = sum(o.correct for o in outcomes)
    accepted = sum(o.acceptable for o in outcomes)
    called = sum(o.called_any_tool for o in outcomes)

    print("\n" + "=" * 60)
    print("SPIKE SUMMARY")
    print("=" * 60)
    print(f"  cases                     : {total}")
    print(f"  called a tool at all      : {called}/{total}")
    print(f"  correct first tool (exact): {exact}/{total}")
    print(f"  acceptable first tool     : {accepted}/{total}")
    print("-" * 60)

    # The decision gate. Exact >= 8/10 is the pass bar we set up front.
    passed = exact >= 8 and called == total
    verdict = (
        "PASS — build the full agent on this."
        if passed
        else "FAIL — weigh a larger tool-tuned model before proceeding."
    )
    print(f"  VERDICT: {verdict}")
    print("=" * 60)
    return passed


# ============================================================
# Preflight — fail fast with a clear message if the model isn't reachable
# ============================================================


def _preflight() -> bool:
    client = get_llm()
    health = client.health_check()
    if not health["reachable"]:
        print("Ollama is NOT reachable.")
        print(f"  host: {client.config.host}")
        print(f"  error: {health.get('error')}")
        print("  Start Ollama on the host and confirm OPS_OLLAMA_HOST.")
        return False
    if not health["model_present"]:
        print(f"Model '{client.config.model}' is not pulled in Ollama.")
        print(f"  available: {', '.join(health.get('models', [])) or '(none)'}")
        print(f"  run: ollama pull {client.config.model}")
        return False
    print(f"Ollama reachable · model '{client.config.model}' present.")
    return True


# ============================================================
# CLI
# ============================================================


def main() -> int:
    """Run the spike battery and report tool-calling accuracy.

    Returns:
        A process exit code: 0 on success, non-zero on failure.
    """
    parser = argparse.ArgumentParser(
        description="Phase 4 agent tool-calling spike test."
    )
    parser.add_argument(
        "--dataset", type=int, default=542, help="dataset_id to probe (default 542)."
    )
    parser.add_argument(
        "--only", type=int, default=None, help="run a single case by 1-based index."
    )
    parser.add_argument("--list", action="store_true", help="list cases and exit.")
    args = parser.parse_args()

    if args.list:
        for i, c in enumerate(CASES, 1):
            print(f"{i:02d}. [{c.expect}] {c.question}")
        return 0

    if not _preflight():
        return 2

    selected = [CASES[args.only - 1]] if args.only else CASES
    start_index = args.only or 1

    print(f"\nRunning {len(selected)} case(s) against dataset {args.dataset} ...")
    outcomes: list[Outcome] = []
    t0 = time.time()
    for offset, case in enumerate(selected):
        idx = start_index + offset
        try:
            outcome = _run_case(case, args.dataset)
        except Exception as exc:  # noqa: BLE001
            # A hard crash here means a safety net is missing — that itself is a
            # spike finding, so surface it loudly rather than swallowing it.
            print(f"\n[{idx:02d}] HARD CRASH — {type(exc).__name__}: {exc}")
            print("     A crash means our error handling has a gap to fix.")
            outcomes.append(
                Outcome(case, None, False, False, False, 0, f"CRASH: {exc}", [])
            )
            continue
        _print_case(idx, outcome)
        outcomes.append(outcome)

    elapsed = time.time() - t0
    print(
        f"\nCompleted in {elapsed:.1f}s ({elapsed / max(len(selected), 1):.1f}s per case)."
    )

    if args.only:
        # Single-case mode is for eyeballing, not a verdict.
        return 0

    passed = _summary(outcomes)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
