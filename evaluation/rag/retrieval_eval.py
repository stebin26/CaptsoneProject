"""Retrieval evaluation for the RAG document assistant.

Scores the retrieval half of the pipeline against the hand-written ground truth
in ``ground_truth.json``. Retrieval is evaluated separately from generation
because the two fail for different reasons and a combined score hides which one
is at fault: an answer can be wrong because the wrong passage was fetched, or
because the right passage was fetched and the model ignored it.

Relevance is resolved from the document filename and the answer keywords rather
than from chunk identifiers. Chunk ids change whenever the chunk size changes,
and the planned chunk-size ablation would otherwise invalidate the entire
ground truth set. Resolving relevance from content keeps one evaluation set
valid across every configuration.

Two levels of hit rate are reported:

* Document level -- the correct document appears in the top k. With a corpus of
  five documents this is a weak signal and is reported for context only.
* Passage level -- a chunk from the correct document containing the answer
  keywords appears in the top k. This is the metric that carries meaning.

Out-of-scope questions are scored on abstention: the correct behaviour is for
retrieval to return nothing once the distance cutoff has been applied.

Run inside the API container::

    docker compose exec api sh -c "cd /app && python -m evaluation.rag.retrieval_eval"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Ranks at which hit rate, recall and precision are reported. Hit Rate@1 is the
# discriminating figure; larger k values are reported for context because a
# small corpus inflates them.
K_VALUES = (1, 3, 5)

# The dataset the evaluation corpus is indexed against.
DEFAULT_DATASET_ID = 5

# Candidate locations for the RAG service package. The modules inside it import
# one another by bare name (``from embedder import ...``), so the package
# directory itself has to be on sys.path rather than its parent.
_RAG_PATH_CANDIDATES = (
    Path("/app/services/rag"),
    Path(__file__).resolve().parents[2] / "services" / "rag",
)


class EvaluationSetupError(RuntimeError):
    """Raised when the evaluation cannot be run at all.

    Distinguished from a poor score: a setup failure means the numbers are
    absent, not bad, and must never be reported as a result.
    """


def _add_rag_to_path() -> Path:
    """Put the RAG service package on ``sys.path``.

    Returns:
        The directory that was added.

    Raises:
        EvaluationSetupError: If no candidate directory contains the retriever.
    """
    for candidate in _RAG_PATH_CANDIDATES:
        if (candidate / "retriever.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            logger.debug("RAG package located", extra={"path": str(candidate)})
            return candidate

    tried = ", ".join(str(p) for p in _RAG_PATH_CANDIDATES)
    raise EvaluationSetupError(
        f"Could not locate the RAG service package. Tried: {tried}. "
        "Run this inside the API container, or from the repository root."
    )


@dataclass
class QuestionResult:
    """Retrieval outcome for a single ground truth question."""

    question_id: str
    scope: str
    domain: str
    question: str
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    gold_chunk_count: int = 0
    relevant_ranks: list[int] = field(default_factory=list)
    doc_ranks: list[int] = field(default_factory=list)
    abstained: bool = False
    error: str | None = None

    @property
    def first_relevant_rank(self) -> int | None:
        """Return the 1-based rank of the first relevant chunk, if any."""
        return min(self.relevant_ranks) if self.relevant_ranks else None

    @property
    def first_doc_rank(self) -> int | None:
        """Return the 1-based rank of the first chunk from the gold document."""
        return min(self.doc_ranks) if self.doc_ranks else None


def load_ground_truth(path: Path) -> list[dict[str, Any]]:
    """Load and sanity-check the ground truth file.

    Args:
        path: Location of ``ground_truth.json``.

    Returns:
        The parsed question list.

    Raises:
        EvaluationSetupError: If the file is missing, malformed, or fails the
            structural checks that would otherwise produce silently wrong
            metrics.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationSetupError(
            f"Could not read the ground truth file at {path}: {exc}"
        ) from exc

    try:
        questions = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvaluationSetupError(
            f"Ground truth file at {path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(questions, list) or not questions:
        raise EvaluationSetupError(
            f"Ground truth file at {path} must contain a non-empty list."
        )

    seen: set[str] = set()
    for index, item in enumerate(questions):
        qid = item.get("id")
        if not qid:
            raise EvaluationSetupError(f"Question at position {index} has no id.")
        if qid in seen:
            raise EvaluationSetupError(f"Duplicate question id: {qid}")
        seen.add(qid)

        scope = item.get("scope")
        if scope not in {"in_scope", "distractor", "out_of_scope"}:
            raise EvaluationSetupError(f"{qid}: unknown scope {scope!r}")

        if scope == "out_of_scope":
            if item.get("document") is not None:
                raise EvaluationSetupError(
                    f"{qid}: out_of_scope questions must have a null document."
                )
        elif not item.get("document"):
            raise EvaluationSetupError(f"{qid}: answerable question has no document.")

    logger.info(
        "Ground truth loaded",
        extra={"path": str(path), "questions": len(questions)},
    )
    return questions


def resolve_gold_chunks(
    dataset_id: int,
    questions: list[dict[str, Any]],
) -> dict[str, set[int]]:
    """Find, for each answerable question, the chunks that contain its answer.

    A chunk is treated as relevant when it belongs to the question's gold
    document and contains every answer keyword. Requiring all keywords rather
    than any of them keeps multi-part answers -- where both a level and a
    duration matter -- from being satisfied by half the answer.

    Resolving this from stored chunk content, rather than recording chunk ids in
    the ground truth, is what allows the same ground truth to score any chunking
    configuration.

    Args:
        dataset_id: Dataset whose indexed chunks are searched.
        questions: Parsed ground truth entries.

    Returns:
        Mapping of question id to the set of relevant chunk ids. Out-of-scope
        questions are absent from the mapping.

    Raises:
        EvaluationSetupError: If the chunk store cannot be read, or if an
            answerable question resolves to no chunks at all -- which means the
            ground truth and the corpus disagree, and every metric built on it
            would be meaningless.
    """
    from vector_store import _conn  # noqa: PLC0415 - path set up at runtime

    try:
        with _conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, d.filename, c.content
                FROM rag.chunks c
                JOIN rag.documents d ON d.id = c.document_id
                WHERE d.dataset_id = %s
                """,
                (dataset_id,),
            )
            rows = cur.fetchall()
    except Exception as exc:
        logger.exception(
            "Could not read indexed chunks",
            extra={"dataset_id": dataset_id},
        )
        raise EvaluationSetupError(
            f"Could not read indexed chunks for dataset {dataset_id}: {exc}"
        ) from exc

    if not rows:
        raise EvaluationSetupError(
            f"Dataset {dataset_id} has no indexed chunks. Upload and index the "
            "evaluation corpus before running this."
        )

    logger.info(
        "Indexed corpus read",
        extra={"dataset_id": dataset_id, "chunks": len(rows)},
    )

    gold: dict[str, set[int]] = {}
    unresolved: list[str] = []

    for item in questions:
        if item["scope"] == "out_of_scope":
            continue

        wanted_doc = item["document"]
        keywords = [k.lower() for k in item.get("answer_keywords") or []]
        matches: set[int] = set()

        for chunk_id, filename, content in rows:
            if filename != wanted_doc:
                continue
            haystack = (content or "").lower()
            if all(k in haystack for k in keywords):
                matches.add(chunk_id)

        if not matches:
            unresolved.append(item["id"])
        gold[item["id"]] = matches

    if unresolved:
        raise EvaluationSetupError(
            "These questions resolved to no chunk in their gold document, so "
            "the ground truth and the indexed corpus disagree: "
            f"{', '.join(unresolved)}. Check the document name and the answer "
            "keywords before trusting any score."
        )

    sizes = [len(v) for v in gold.values()]
    logger.info(
        "Gold chunks resolved",
        extra={
            "questions": len(gold),
            "min_gold_chunks": min(sizes),
            "max_gold_chunks": max(sizes),
        },
    )
    return gold


def evaluate_question(
    dataset_id: int,
    item: dict[str, Any],
    gold_chunk_ids: set[int],
    top_k: int,
) -> QuestionResult:
    """Run retrieval for one question and record where the relevant chunks landed.

    Args:
        dataset_id: Dataset to search.
        item: One ground truth entry.
        gold_chunk_ids: Chunk ids holding the answer; empty for out-of-scope.
        top_k: Number of chunks to request.

    Returns:
        The outcome, including any retrieval error, so that one failing question
        does not abort the whole run.
    """
    from retriever import retrieve  # noqa: PLC0415 - path set up at runtime

    result = QuestionResult(
        question_id=item["id"],
        scope=item["scope"],
        domain=item.get("domain", "unknown"),
        question=item["question"],
        gold_chunk_count=len(gold_chunk_ids),
    )

    try:
        retrieval = retrieve(dataset_id, item["question"], top_k=top_k)
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        logger.exception(
            "Retrieval failed for a question",
            extra={"question_id": item["id"], "dataset_id": dataset_id},
        )
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    result.abstained = retrieval.is_empty

    for rank, chunk in enumerate(retrieval.chunks, start=1):
        result.retrieved.append(
            {
                "rank": rank,
                "chunk_id": getattr(chunk, "chunk_id", None),
                "filename": chunk.filename,
                "distance": round(chunk.distance, 4),
            }
        )
        if item["scope"] == "out_of_scope":
            continue
        if chunk.filename == item["document"]:
            result.doc_ranks.append(rank)
        chunk_id = getattr(chunk, "chunk_id", None)
        if chunk_id is not None and chunk_id in gold_chunk_ids:
            result.relevant_ranks.append(rank)

    return result


def _hit_at(rank: int | None, k: int) -> int:
    """Return 1 when a rank exists and falls within the top k, else 0."""
    return 1 if rank is not None and rank <= k else 0


def summarise(results: list[QuestionResult]) -> dict[str, Any]:
    """Aggregate per-question outcomes into the reported metrics.

    Answerable questions (in-scope and distractor) contribute to hit rate, MRR,
    recall and precision. Out-of-scope questions contribute only to abstention
    accuracy, since there is no correct passage for them to retrieve.

    Args:
        results: One entry per ground truth question.

    Returns:
        A nested summary suitable for writing to JSON and rendering on the
        evaluation dashboard.
    """
    answerable = [r for r in results if r.scope != "out_of_scope" and not r.error]
    out_of_scope = [r for r in results if r.scope == "out_of_scope" and not r.error]
    failed = [r for r in results if r.error]

    summary: dict[str, Any] = {
        "counts": {
            "total": len(results),
            "answerable": len(answerable),
            "out_of_scope": len(out_of_scope),
            "errors": len(failed),
        },
        "passage_level": {},
        "document_level": {},
        "abstention": {},
        "by_scope": {},
    }

    if answerable:
        for k in K_VALUES:
            hits = sum(_hit_at(r.first_relevant_rank, k) for r in answerable)
            doc_hits = sum(_hit_at(r.first_doc_rank, k) for r in answerable)
            summary["passage_level"][f"hit_rate@{k}"] = round(
                hits / len(answerable), 4
            )
            summary["document_level"][f"hit_rate@{k}"] = round(
                doc_hits / len(answerable), 4
            )

            recalls = []
            precisions = []
            for r in answerable:
                found = sum(1 for rank in r.relevant_ranks if rank <= k)
                recalls.append(found / r.gold_chunk_count if r.gold_chunk_count else 0)
                precisions.append(found / k)
            summary["passage_level"][f"recall@{k}"] = round(
                sum(recalls) / len(recalls), 4
            )
            summary["passage_level"][f"precision@{k}"] = round(
                sum(precisions) / len(precisions), 4
            )

        reciprocals = [
            1 / r.first_relevant_rank if r.first_relevant_rank else 0.0
            for r in answerable
        ]
        summary["passage_level"]["mrr"] = round(
            sum(reciprocals) / len(reciprocals), 4
        )

    if out_of_scope:
        correct = sum(1 for r in out_of_scope if r.abstained)
        summary["abstention"] = {
            "questions": len(out_of_scope),
            "correctly_refused": correct,
            "accuracy": round(correct / len(out_of_scope), 4),
        }

    for scope in ("in_scope", "distractor"):
        subset = [r for r in answerable if r.scope == scope]
        if not subset:
            continue
        summary["by_scope"][scope] = {
            "questions": len(subset),
            **{
                f"hit_rate@{k}": round(
                    sum(_hit_at(r.first_relevant_rank, k) for r in subset)
                    / len(subset),
                    4,
                )
                for k in K_VALUES
            },
        }

    if failed:
        summary["failed_questions"] = [
            {"id": r.question_id, "error": r.error} for r in failed
        ]

    return summary


def run(
    dataset_id: int,
    ground_truth_path: Path,
    top_k: int,
) -> dict[str, Any]:
    """Run the full retrieval evaluation.

    Args:
        dataset_id: Dataset holding the indexed evaluation corpus.
        ground_truth_path: Location of the ground truth file.
        top_k: Number of chunks to request per question.

    Returns:
        A report containing the run configuration, the aggregate metrics, and
        the per-question detail needed to explain any individual failure.
    """
    _add_rag_to_path()

    questions = load_ground_truth(ground_truth_path)
    gold = resolve_gold_chunks(dataset_id, questions)

    results = [
        evaluate_question(dataset_id, item, gold.get(item["id"], set()), top_k)
        for item in questions
    ]

    summary = summarise(results)

    logger.info(
        "Retrieval evaluation complete",
        extra={
            "dataset_id": dataset_id,
            "questions": len(results),
            "errors": summary["counts"]["errors"],
        },
    )

    return {
        "configuration": {
            "dataset_id": dataset_id,
            "top_k": top_k,
            "ground_truth": str(ground_truth_path),
            "k_values": list(K_VALUES),
        },
        "summary": summary,
        "questions": [
            {
                "id": r.question_id,
                "scope": r.scope,
                "domain": r.domain,
                "question": r.question,
                "gold_chunk_count": r.gold_chunk_count,
                "first_relevant_rank": r.first_relevant_rank,
                "first_document_rank": r.first_doc_rank,
                "abstained": r.abstained,
                "retrieved": r.retrieved,
                "error": r.error,
            }
            for r in results
        ],
    }


def _print_report(report: dict[str, Any]) -> None:
    """Print a short human-readable summary of a report."""
    summary = report["summary"]
    counts = summary["counts"]

    print("=" * 62)
    print(" Retrieval Evaluation")
    print("=" * 62)
    print(
        f" dataset={report['configuration']['dataset_id']} "
        f"top_k={report['configuration']['top_k']} "
        f"questions={counts['total']}"
    )
    print()

    if summary["passage_level"]:
        print(" Passage level (the metric that carries meaning)")
        print(" " + "-" * 58)
        for k in K_VALUES:
            print(
                f"   Hit Rate@{k:<2} {summary['passage_level'][f'hit_rate@{k}']:.2%}"
                f"     Recall@{k:<2} {summary['passage_level'][f'recall@{k}']:.2%}"
                f"     Precision@{k:<2} {summary['passage_level'][f'precision@{k}']:.2%}"
            )
        print(f"   MRR         {summary['passage_level']['mrr']:.4f}")
        print()

    if summary["document_level"]:
        print(" Document level (context only - the corpus has five documents)")
        print(" " + "-" * 58)
        for k in K_VALUES:
            print(
                f"   Hit Rate@{k:<2} {summary['document_level'][f'hit_rate@{k}']:.2%}"
            )
        print()

    if summary["by_scope"]:
        print(" By question type")
        print(" " + "-" * 58)
        for scope, stats in summary["by_scope"].items():
            print(
                f"   {scope:<12} n={stats['questions']:<3} "
                f"Hit@1 {stats['hit_rate@1']:.2%}   "
                f"Hit@5 {stats['hit_rate@5']:.2%}"
            )
        print()

    if summary["abstention"]:
        abst = summary["abstention"]
        print(" Abstention on out-of-scope questions")
        print(" " + "-" * 58)
        print(
            f"   {abst['correctly_refused']}/{abst['questions']} correctly "
            f"refused   accuracy {abst['accuracy']:.2%}"
        )
        print()

    if counts["errors"]:
        print(f" WARNING: {counts['errors']} question(s) failed to run.")
        for failure in summary.get("failed_questions", []):
            print(f"   {failure['id']}: {failure['error']}")
        print()

    print("=" * 62)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point.

    Args:
        argv: Argument list, defaulting to the process arguments.

    Returns:
        Process exit code: 0 on a completed run, 1 on a setup failure.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval against the hand-written ground truth."
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=DEFAULT_DATASET_ID,
        help=f"Dataset holding the indexed corpus (default: {DEFAULT_DATASET_ID})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=max(K_VALUES),
        help="Chunks to request per question (default: %(default)s)",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path(__file__).resolve().parent / "ground_truth.json",
        help="Path to ground_truth.json",
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

    if args.top_k < max(K_VALUES):
        logger.warning(
            "top_k is below the largest reported k; higher-k metrics will be "
            "understated",
            extra={"top_k": args.top_k, "max_k": max(K_VALUES)},
        )

    try:
        report = run(args.dataset_id, args.ground_truth, args.top_k)
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
