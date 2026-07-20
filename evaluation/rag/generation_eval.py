"""Generation evaluation for the RAG document assistant.

Scores the generation half of the pipeline: given whatever retrieval returned,
did the system produce a correct, grounded answer, and did it refuse when it
should have.

No LLM judge is used, deliberately. The only model available locally is the same
llama3.2:3b that produces the answers, so asking it to grade its own output
would produce a number that cannot be defended and that a reviewer would
rightly discount. Every metric here is computed from string and numeric
comparison instead, which makes the whole evaluation reproducible: running it
twice on the same answers gives the same score.

Metrics
-------
Refusal accuracy
    Of the out-of-scope questions, how many did the system decline to answer.
    Retrieval-level abstention was measured separately and found to be zero, so
    this measures the system's real end-to-end behaviour, which comes from the
    generation prompt rather than from the distance cutoff.

False refusal rate
    Of the answerable questions, how many were refused anyway. Refusal accuracy
    alone is trivially maximised by refusing everything, so the two are always
    reported together.

Answer correctness
    Whether every answer keyword from the ground truth appears in the answer.
    Requiring all of them rather than any keeps a half-answer from scoring as a
    whole one.

Numeric groundedness
    Whether every number stated in the answer also appears in the retrieved
    context. This is the faithfulness proxy, and it is the right one for this
    corpus: the documents are almost entirely thresholds and codes, so the
    realistic hallucination is a wrong figure -- 0.45 in place of 0.045 -- not
    an invented narrative.

Citation rate
    Whether the answer carries a source marker, which the system prompt asks
    for.

Run inside the API container. This makes one model call per question and takes
appreciably longer than the retrieval evaluation::

    docker compose exec api sh -c "cd /app && python -m evaluation.rag.generation_eval \\
        --out evaluation/results/generation_600.json"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DATASET_ID = 5

# Candidate locations for the RAG service package. Its modules import one
# another by bare name, so the package directory itself goes on sys.path.
_RAG_PATH_CANDIDATES = (
    Path("/app/services/rag"),
    Path(__file__).resolve().parents[2] / "services" / "rag",
)

# Phrases that mark an answer as a refusal. The first two are the system's own
# fixed messages, imported at runtime; these cover the model declining in its
# own words while retrieval still returned something.
_REFUSAL_PATTERNS = (
    "not covered",
    "does not contain",
    "do not contain",
    "no information",
    "not found in",
    "cannot find",
    "could not find",
    "not mentioned",
    "not specified in",
    "unable to answer",
    "not available in",
    "the documents do not",
    "insufficient information",
)

# Matches integers and decimals, including those written with a leading dot.
# Percent signs, units and currency are stripped by the tokenizer around it.
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# Source markers the prompt asks the model to emit.
_CITATION_RE = re.compile(r"\[\s*source\s*\d+\s*\]", re.IGNORECASE)

# Numbers too common to carry meaning as groundedness evidence. Small integers
# appear in ordinary prose ("one of the two reasons") and counting them as
# ungrounded would penalise correct answers for writing English.
_TRIVIAL_NUMBERS = {"0", "1", "2", "3", "4", "5"}


class EvaluationSetupError(RuntimeError):
    """Raised when the evaluation cannot run at all.

    Kept distinct from a low score: a setup failure means there is no result to
    report, and reporting one anyway would be worse than reporting nothing.
    """


def _add_rag_to_path() -> Path:
    """Put the RAG service package on ``sys.path``.

    Returns:
        The directory that was added.

    Raises:
        EvaluationSetupError: If no candidate directory holds the QA chain.
    """
    for candidate in _RAG_PATH_CANDIDATES:
        if (candidate / "qa_chain.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            logger.debug("RAG package located", extra={"path": str(candidate)})
            return candidate

    tried = ", ".join(str(p) for p in _RAG_PATH_CANDIDATES)
    raise EvaluationSetupError(
        f"Could not locate the RAG service package. Tried: {tried}. "
        "Run this inside the API container, or from the repository root."
    )


def _system_refusal_messages() -> tuple[str, ...]:
    """Return the system's own fixed refusal messages, lowercased.

    These are read from the QA chain rather than copied, so that changing the
    wording there cannot silently break refusal detection here.

    Returns:
        The refusal message strings, or an empty tuple if they are not exposed
        under the expected names.
    """
    try:
        import qa_chain  # noqa: PLC0415 - path set up at runtime
    except ImportError:
        logger.warning("Could not import qa_chain to read its refusal messages")
        return ()

    messages = []
    for name in ("_NO_DOCS_MSG", "_NO_MATCH_MSG"):
        value = getattr(qa_chain, name, None)
        if isinstance(value, str) and value.strip():
            messages.append(value.strip().lower())
        else:
            logger.warning(
                "QA chain does not expose an expected refusal message",
                extra={"constant": name},
            )
    return tuple(messages)


def is_refusal(answer_text: str, grounded: bool, system_messages: tuple[str, ...]) -> bool:
    """Decide whether an answer amounts to a refusal.

    Three routes count. The chain reports ``grounded=False`` when it declined
    before reaching the model. The answer may be one of the system's fixed
    refusal messages. Or the model may have been given context and declined in
    its own words, which is the case retrieval-level metrics cannot see.

    Args:
        answer_text: The generated answer.
        grounded: Whether the chain considered the answer grounded in chunks.
        system_messages: The chain's own refusal strings, lowercased.

    Returns:
        True if the answer declines to answer.
    """
    if not grounded:
        return True

    lowered = (answer_text or "").strip().lower()
    if not lowered:
        return True

    if any(msg and msg in lowered for msg in system_messages):
        return True

    return any(pattern in lowered for pattern in _REFUSAL_PATTERNS)


def _numbers_in(text: str) -> list[str]:
    """Extract the numeric tokens from a piece of text.

    Trailing zeros are not normalised, because in this corpus ``4.0`` and ``4``
    are the same threshold but ``0.045`` and ``0.45`` are not, and normalising
    aggressively would hide exactly the error this metric exists to catch.

    Args:
        text: Text to scan.

    Returns:
        The numeric tokens found, in order.
    """
    return _NUMBER_RE.findall(text or "")


def _number_matches(number: str, haystack: str) -> bool:
    """Check whether a number appears in text, allowing trailing-zero variation.

    Args:
        number: The numeric token from the answer.
        haystack: The retrieved context.

    Returns:
        True if the number, or an equivalent spelling of it, is present.
    """
    if number in haystack:
        return True
    # 4.0 in the answer against 4 in the source, and the reverse.
    if "." in number:
        trimmed = number.rstrip("0").rstrip(".")
        if trimmed and trimmed in haystack:
            return True
    else:
        if f"{number}.0" in haystack:
            return True
    return False


@dataclass
class GenerationResult:
    """The generation outcome for one ground truth question."""

    question_id: str
    scope: str
    domain: str
    question: str
    answer: str = ""
    grounded: bool = False
    llm_used: bool = False
    used_chunks: int = 0
    refused: bool = False
    keywords_expected: list[str] = field(default_factory=list)
    keywords_found: list[str] = field(default_factory=list)
    correct: bool | None = None
    numbers_in_answer: list[str] = field(default_factory=list)
    numbers_ungrounded: list[str] = field(default_factory=list)
    numerically_grounded: bool | None = None
    cited: bool = False
    seconds: float = 0.0
    error: str | None = None


def load_ground_truth(path: Path) -> list[dict[str, Any]]:
    """Load the ground truth question set.

    Args:
        path: Location of ``ground_truth.json``.

    Returns:
        The parsed question list.

    Raises:
        EvaluationSetupError: If the file is missing or malformed.
    """
    try:
        questions = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationSetupError(
            f"Could not read the ground truth file at {path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise EvaluationSetupError(
            f"Ground truth file at {path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(questions, list) or not questions:
        raise EvaluationSetupError(
            f"Ground truth file at {path} must contain a non-empty list."
        )

    logger.info(
        "Ground truth loaded",
        extra={"path": str(path), "questions": len(questions)},
    )
    return questions


def evaluate_question(
    dataset_id: int,
    item: dict[str, Any],
    system_messages: tuple[str, ...],
) -> GenerationResult:
    """Generate an answer for one question and score it.

    Retrieval is run separately from the answer call so that the retrieved text
    is available for the groundedness check. Retrieval is deterministic for a
    given index and query, so the context scored here is the context the answer
    was built from.

    Args:
        dataset_id: Dataset to answer against.
        item: One ground truth entry.
        system_messages: The chain's fixed refusal strings, lowercased.

    Returns:
        The scored outcome. A failure is recorded on the result rather than
        raised, so that one bad question does not discard the other twenty-nine.
    """
    from qa_chain import answer_question  # noqa: PLC0415 - path set at runtime
    from retriever import retrieve  # noqa: PLC0415 - path set at runtime

    result = GenerationResult(
        question_id=item["id"],
        scope=item["scope"],
        domain=item.get("domain", "unknown"),
        question=item["question"],
        keywords_expected=list(item.get("answer_keywords") or []),
    )

    started = time.monotonic()
    try:
        retrieval = retrieve(dataset_id, item["question"])
        context = " ".join(c.content or "" for c in retrieval.chunks)

        answer = answer_question(dataset_id, item["question"])
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        logger.exception(
            "Generation failed for a question",
            extra={"question_id": item["id"], "dataset_id": dataset_id},
        )
        result.error = f"{type(exc).__name__}: {exc}"
        result.seconds = round(time.monotonic() - started, 2)
        return result

    result.seconds = round(time.monotonic() - started, 2)
    result.answer = answer.answer
    result.grounded = answer.grounded
    result.llm_used = answer.llm_used
    result.used_chunks = answer.used_chunks
    result.refused = is_refusal(answer.answer, answer.grounded, system_messages)
    result.cited = bool(_CITATION_RE.search(answer.answer or ""))

    lowered_answer = (answer.answer or "").lower()

    if result.keywords_expected:
        result.keywords_found = [
            k for k in result.keywords_expected if k.lower() in lowered_answer
        ]
        result.correct = len(result.keywords_found) == len(result.keywords_expected)

    if not result.refused:
        numbers = [n for n in _numbers_in(answer.answer) if n not in _TRIVIAL_NUMBERS]
        result.numbers_in_answer = numbers
        result.numbers_ungrounded = [
            n for n in numbers if not _number_matches(n, context)
        ]
        result.numerically_grounded = not result.numbers_ungrounded

    logger.info(
        "Question answered",
        extra={
            "question_id": item["id"],
            "scope": item["scope"],
            "refused": result.refused,
            "correct": result.correct,
            "seconds": result.seconds,
        },
    )
    return result


def summarise(results: list[GenerationResult]) -> dict[str, Any]:
    """Aggregate per-question outcomes into the reported metrics.

    Args:
        results: One entry per ground truth question.

    Returns:
        A nested summary suitable for writing to JSON and rendering on the
        evaluation dashboard.
    """
    ok = [r for r in results if not r.error]
    answerable = [r for r in ok if r.scope != "out_of_scope"]
    out_of_scope = [r for r in ok if r.scope == "out_of_scope"]
    failed = [r for r in results if r.error]

    summary: dict[str, Any] = {
        "counts": {
            "total": len(results),
            "answerable": len(answerable),
            "out_of_scope": len(out_of_scope),
            "errors": len(failed),
        },
        "refusal": {},
        "correctness": {},
        "groundedness": {},
        "citation": {},
        "by_scope": {},
    }

    if out_of_scope:
        refused = sum(1 for r in out_of_scope if r.refused)
        summary["refusal"]["refusal_accuracy"] = round(refused / len(out_of_scope), 4)
        summary["refusal"]["correctly_refused"] = refused
        summary["refusal"]["out_of_scope_questions"] = len(out_of_scope)

    if answerable:
        wrongly_refused = sum(1 for r in answerable if r.refused)
        summary["refusal"]["false_refusal_rate"] = round(
            wrongly_refused / len(answerable), 4
        )
        summary["refusal"]["wrongly_refused"] = wrongly_refused

        scored = [r for r in answerable if r.correct is not None]
        if scored:
            correct = sum(1 for r in scored if r.correct)
            summary["correctness"]["answer_correctness"] = round(
                correct / len(scored), 4
            )
            summary["correctness"]["correct"] = correct
            summary["correctness"]["scored"] = len(scored)

        checked = [r for r in answerable if r.numerically_grounded is not None]
        if checked:
            grounded = sum(1 for r in checked if r.numerically_grounded)
            summary["groundedness"]["numeric_groundedness"] = round(
                grounded / len(checked), 4
            )
            summary["groundedness"]["grounded"] = grounded
            summary["groundedness"]["checked"] = len(checked)
            summary["groundedness"]["ungrounded_examples"] = [
                {"id": r.question_id, "numbers": r.numbers_ungrounded}
                for r in checked
                if r.numbers_ungrounded
            ][:5]

        cited = sum(1 for r in answerable if r.cited)
        summary["citation"]["citation_rate"] = round(cited / len(answerable), 4)

        llm_used = sum(1 for r in answerable if r.llm_used)
        summary["counts"]["llm_used"] = llm_used
        summary["counts"]["extractive_fallback"] = len(answerable) - llm_used

    for scope in ("in_scope", "distractor"):
        subset = [r for r in answerable if r.scope == scope]
        scored = [r for r in subset if r.correct is not None]
        if not scored:
            continue
        summary["by_scope"][scope] = {
            "questions": len(scored),
            "answer_correctness": round(
                sum(1 for r in scored if r.correct) / len(scored), 4
            ),
        }

    if ok:
        summary["counts"]["mean_seconds"] = round(
            sum(r.seconds for r in ok) / len(ok), 2
        )

    if failed:
        summary["failed_questions"] = [
            {"id": r.question_id, "error": r.error} for r in failed
        ]

    return summary


def run(
    dataset_id: int,
    ground_truth_path: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the full generation evaluation.

    Args:
        dataset_id: Dataset holding the indexed evaluation corpus.
        ground_truth_path: Location of the ground truth file.
        limit: Stop after this many questions. Intended for a quick check that
            the wiring works before committing to a full run.

    Returns:
        A report containing the configuration, the aggregate metrics, and the
        per-question detail needed to explain any individual answer.
    """
    _add_rag_to_path()

    questions = load_ground_truth(ground_truth_path)
    if limit:
        questions = questions[:limit]
        logger.warning(
            "Running a truncated evaluation; results are not reportable",
            extra={"limit": limit},
        )

    system_messages = _system_refusal_messages()

    results: list[GenerationResult] = []
    for index, item in enumerate(questions, start=1):
        logger.info(
            "Evaluating question %s of %s", index, len(questions),
            extra={"question_id": item["id"]},
        )
        results.append(evaluate_question(dataset_id, item, system_messages))

    summary = summarise(results)

    logger.info(
        "Generation evaluation complete",
        extra={
            "dataset_id": dataset_id,
            "questions": len(results),
            "errors": summary["counts"]["errors"],
        },
    )

    return {
        "configuration": {
            "dataset_id": dataset_id,
            "ground_truth": str(ground_truth_path),
            "questions_run": len(results),
            "truncated": bool(limit),
            "judge": "none - all metrics are deterministic string and numeric checks",
        },
        "summary": summary,
        "questions": [
            {
                "id": r.question_id,
                "scope": r.scope,
                "domain": r.domain,
                "question": r.question,
                "answer": r.answer,
                "refused": r.refused,
                "grounded": r.grounded,
                "llm_used": r.llm_used,
                "used_chunks": r.used_chunks,
                "keywords_expected": r.keywords_expected,
                "keywords_found": r.keywords_found,
                "correct": r.correct,
                "numbers_in_answer": r.numbers_in_answer,
                "numbers_ungrounded": r.numbers_ungrounded,
                "numerically_grounded": r.numerically_grounded,
                "cited": r.cited,
                "seconds": r.seconds,
                "error": r.error,
            }
            for r in results
        ],
    }


def _print_report(report: dict[str, Any]) -> None:
    """Print a short human-readable summary of a report."""
    summary = report["summary"]
    counts = summary["counts"]

    print("=" * 66)
    print(" Generation Evaluation")
    print("=" * 66)
    print(
        f" dataset={report['configuration']['dataset_id']} "
        f"questions={counts['total']} "
        f"mean={counts.get('mean_seconds', 0):.1f}s per question"
    )
    if report["configuration"]["truncated"]:
        print(" TRUNCATED RUN - not reportable")
    print()

    refusal = summary["refusal"]
    if refusal:
        print(" Refusal behaviour")
        print(" " + "-" * 62)
        if "refusal_accuracy" in refusal:
            print(
                f"   Refusal accuracy      {refusal['refusal_accuracy']:.2%}  "
                f"({refusal['correctly_refused']}/"
                f"{refusal['out_of_scope_questions']} out-of-scope refused)"
            )
        if "false_refusal_rate" in refusal:
            print(
                f"   False refusal rate    {refusal['false_refusal_rate']:.2%}  "
                f"({refusal['wrongly_refused']} answerable questions refused)"
            )
        print()

    if summary["correctness"]:
        c = summary["correctness"]
        print(" Answer correctness")
        print(" " + "-" * 62)
        print(
            f"   Correctness           {c['answer_correctness']:.2%}  "
            f"({c['correct']}/{c['scored']} with all keywords present)"
        )
        for scope, stats in summary["by_scope"].items():
            print(
                f"     {scope:<12} n={stats['questions']:<3} "
                f"{stats['answer_correctness']:.2%}"
            )
        print()

    if summary["groundedness"]:
        g = summary["groundedness"]
        print(" Numeric groundedness (faithfulness proxy)")
        print(" " + "-" * 62)
        print(
            f"   Groundedness          {g['numeric_groundedness']:.2%}  "
            f"({g['grounded']}/{g['checked']} answers with every figure "
            "traceable to context)"
        )
        for example in g.get("ungrounded_examples", []):
            print(f"     {example['id']}: figures not in context "
                  f"{example['numbers']}")
        print()

    if summary["citation"]:
        print(" Citation")
        print(" " + "-" * 62)
        print(f"   Citation rate         {summary['citation']['citation_rate']:.2%}")
        print()

    if "llm_used" in counts:
        print(
            f" Model used on {counts['llm_used']} answers; "
            f"{counts['extractive_fallback']} fell back to extraction."
        )
        print()

    if counts["errors"]:
        print(f" WARNING: {counts['errors']} question(s) failed to run.")
        for failure in summary.get("failed_questions", []):
            print(f"   {failure['id']}: {failure['error']}")
        print()

    print("=" * 66)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point.

    Args:
        argv: Argument list, defaulting to the process arguments.

    Returns:
        Process exit code: 0 on a completed run, 1 on a setup failure.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate RAG generation against the hand-written ground truth."
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=DEFAULT_DATASET_ID,
        help=f"Dataset holding the indexed corpus (default: {DEFAULT_DATASET_ID})",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path(__file__).resolve().parent / "ground_truth.json",
        help="Path to ground_truth.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N questions, to check the wiring",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the full JSON report to this path",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    try:
        report = run(args.dataset_id, args.ground_truth, args.limit)
    except EvaluationSetupError as exc:
        logger.error("Evaluation could not run: %s", exc)
        return 1

    _print_report(report)

    if args.out:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.error("Could not write the report to %s: %s", args.out, exc)
            return 1
        logger.info("Report written", extra={"path": str(args.out)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
