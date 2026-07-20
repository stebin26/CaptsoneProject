"""Callbacks for /evaluation.

Renders the committed model-evaluation reports read-only. Every number shown is
read from the report JSON served by the API; nothing on this page is hard-coded,
so the page always reflects the last evaluation run. Each report is fetched
independently and a failure to load one is reported in place rather than blanking
the whole page, because a partially generated evaluation is a normal state.
"""

from __future__ import annotations

from typing import Any

from app import feedback, ids
from app.api_client import APIError, evaluation_report
from app.components import ui
from app.logging_setup import get_logger
from app.utils import fmt
from dash import Input, Output, State, callback, html

logger = get_logger(__name__)


# ============================================================
# Small formatting helpers
# ============================================================


def _pct(value: Any) -> str:
    """Format a 0-1 ratio as a percentage string, tolerant of missing values.

    Args:
        value: A ratio between 0 and 1, or None.

    Returns:
        The value as a percentage to two decimals, or an em dash when missing.

    """
    if value is None:
        return "\u2014"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _num(value: Any, places: int = 4) -> str:
    """Format a raw number for display, tolerant of missing values.

    Args:
        value: The number to format, or None.
        places: Decimal places to show.

    Returns:
        The formatted number, or an em dash when missing.

    """
    if value is None:
        return "\u2014"
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return str(value)


def _signed(value: Any, places: int = 4) -> str:
    """Format a number with an explicit sign, used for skill scores.

    Args:
        value: The number to format, or None.
        places: Decimal places to show.

    Returns:
        The signed number, or an em dash when missing.

    """
    if value is None:
        return "\u2014"
    try:
        return f"{float(value):+.{places}f}"
    except (TypeError, ValueError):
        return str(value)


def _kpi_grid(*cards: Any) -> html.Div:
    """Lay out KPI cards in a four-column grid with spacing below.

    Args:
        *cards: The KPI cards to arrange.

    Returns:
        The rendered grid.

    """
    return html.Div(ui.grid(*cards, cols=4), style={"marginBottom": "1.5rem"})


# ============================================================
# RAG rendering
# ============================================================


def _render_retrieval(report: dict[str, Any]) -> Any:
    """Render the retrieval report as KPIs and a per-k table.

    Args:
        report: The parsed retrieval report.

    Returns:
        The rendered section.

    """
    summary = report.get("summary", {})
    passage = summary.get("passage_level", {})
    counts = summary.get("counts", {})

    kpis = _kpi_grid(
        ui.kpi("Hit Rate@5", _pct(passage.get("hit_rate@5")), "gold passage in top 5"),
        ui.kpi("MRR", _num(passage.get("mrr")), "mean reciprocal rank"),
        ui.kpi("Precision@1", _pct(passage.get("precision@1")), "top result correct"),
        ui.kpi("Questions", counts.get("total", 0), "hand-written ground truth"),
    )

    rows = []
    for k in (1, 3, 5):
        rows.append(
            [
                f"@{k}",
                _pct(passage.get(f"hit_rate@{k}")),
                _pct(passage.get(f"recall@{k}")),
                _pct(passage.get(f"precision@{k}")),
            ]
        )
    table = ui.table(
        ["k", "Hit rate", "Recall", "Precision"],
        rows,
        note=(
            "Precision above k=1 is bounded by the single-gold-passage ground "
            "truth, not by the retriever, so only Precision@1 is interpretable."
        ),
    )

    return ui.section("Retrieval", kpis, ui.card(table))


def _render_threshold(report: dict[str, Any]) -> Any:
    """Render the distance-threshold separation and sweep.

    Args:
        report: The parsed threshold report.

    Returns:
        The rendered section.

    """
    separation = report.get("separation", {})
    answerable = separation.get("answerable", {})
    out_of_scope = separation.get("out_of_scope", {})

    kpis = _kpi_grid(
        ui.kpi(
            "Answerable median",
            _num(answerable.get("median")),
            "cosine distance",
        ),
        ui.kpi(
            "Out-of-scope median",
            _num(out_of_scope.get("median")),
            "cosine distance",
            tone="warn",
        ),
        ui.kpi(
            "Distribution gap",
            _num(separation.get("gap")),
            "negative means overlap",
            tone="danger",
        ),
        ui.kpi(
            "Separable",
            "No" if separation.get("separable") is False else "Yes",
            "by a single cutoff",
        ),
    )

    rows = []
    for point in report.get("sweep", []):
        rows.append(
            [
                _num(point.get("cutoff"), 2),
                _pct(point.get("abstention_accuracy")),
                _pct(point.get("hit_rate@1")),
                _pct(point.get("hit_rate@3")),
                _pct(point.get("hit_rate@5")),
            ]
        )
    table = ui.table(
        ["Cutoff", "Abstention", "Hit@1", "Hit@3", "Hit@5"],
        rows,
        note=(
            "The answerable and out-of-scope distance distributions overlap, so "
            "no single cutoff separates them. Refusal is therefore handled at "
            "generation, not by distance thresholding."
        ),
    )

    return ui.section("Distance thresholding", kpis, ui.card(table))


def _render_generation(report: dict[str, Any]) -> Any:
    """Render the generation report, including the retrieval-vs-generation split.

    Args:
        report: The parsed generation report.

    Returns:
        The rendered section.

    """
    summary = report.get("summary", {})
    refusal = summary.get("refusal", {})
    correctness = summary.get("correctness", {})
    groundedness = summary.get("groundedness", {})
    citation = summary.get("citation", {})

    kpis = _kpi_grid(
        ui.kpi(
            "Correctness",
            _pct(correctness.get("answer_correctness")),
            "all keywords present",
        ),
        ui.kpi(
            "Groundedness",
            _pct(groundedness.get("numeric_groundedness")),
            "figures traceable to context",
        ),
        ui.kpi(
            "Refusal accuracy",
            _pct(refusal.get("refusal_accuracy")),
            "out-of-scope declined",
        ),
        ui.kpi(
            "False refusal",
            _pct(refusal.get("false_refusal_rate")),
            "answerable wrongly declined",
            tone="warn",
        ),
    )

    second = _kpi_grid(
        ui.kpi(
            "Citation rate",
            _pct(citation.get("citation_rate")),
            "source marker present",
        ),
        ui.kpi(
            "Mean latency",
            f"{fmt(summary.get('counts', {}).get('mean_seconds'), 1)}s",
            "CPU inference per question",
        ),
        ui.kpi(
            "Model answers",
            summary.get("counts", {}).get("llm_used", 0),
            "vs extractive fallback",
        ),
        ui.kpi(
            "Correct",
            f"{correctness.get('correct', 0)} / {correctness.get('scored', 0)}",
            "answerable questions",
        ),
    )

    note = html.P(
        "Groundedness above correctness is the central finding: the model does "
        "not fabricate figures, it under-extracts them. Correctness is limited "
        "by extraction precision, not by hallucination.",
        className="page-subtitle",
    )

    return ui.section("Generation", kpis, second, note)


def _render_failure_matrix(
    retrieval: dict[str, Any] | None, generation: dict[str, Any] | None
) -> Any:
    """Render the 2x2 retrieval-vs-generation failure attribution.

    Both reports carry per-question records. A question counts as a retrieval hit
    when a gold passage reached the top five, and as generation-correct when the
    generation report marked it correct. Out-of-scope questions are excluded,
    since they have no gold passage and no correct answer to attribute.

    Args:
        retrieval: The parsed retrieval report, or None if it failed to load.
        generation: The parsed generation report, or None if it failed to load.

    Returns:
        The rendered section, or a note explaining why it cannot be shown.

    """
    if not retrieval or not generation:
        return ui.section(
            "Failure attribution",
            feedback.empty(
                "Both the retrieval and generation reports are needed to build "
                "the failure attribution. Run whichever is missing."
            ),
        )

    ret_hit: dict[str, bool] = {}
    for q in retrieval.get("questions", []):
        rank = q.get("first_relevant_rank")
        ret_hit[q.get("id")] = rank is not None and rank <= 5

    hit_correct = hit_fail = miss_correct = miss_fail = 0
    for q in generation.get("questions", []):
        if q.get("scope") == "out_of_scope":
            continue
        correct = bool(q.get("correct"))
        hit = ret_hit.get(q.get("id"), False)
        if hit and correct:
            hit_correct += 1
        elif hit and not correct:
            hit_fail += 1
        elif not hit and correct:
            miss_correct += 1
        else:
            miss_fail += 1

    matrix = ui.table(
        ["", "Generation correct", "Generation failed"],
        [
            ["Gold passage retrieved", str(hit_correct), str(hit_fail)],
            ["Gold passage missed", str(miss_correct), str(miss_fail)],
        ],
        note=(
            f"Retrieval supplied a gold passage for {hit_correct + hit_fail} of "
            f"{hit_correct + hit_fail + miss_correct + miss_fail} answerable "
            f"questions. Of the failures, {hit_fail} occurred despite correct "
            f"retrieval and {miss_fail} from retrieval misses \u2014 generation "
            "is the dominant bottleneck."
        ),
    )

    return ui.section("Failure attribution", ui.card(matrix))


# ============================================================
# ML rendering
# ============================================================


def _render_forecast(report: dict[str, Any]) -> Any:
    """Render the forecasting backtest with its baseline comparison.

    Args:
        report: The parsed forecast report.

    Returns:
        The rendered section.

    """
    summary = report.get("summary", {})
    overall = summary.get("overall", {})
    counts = summary.get("counts", {})
    baselines = overall.get("baselines", {})

    kpis = _kpi_grid(
        ui.kpi("MAE", _num(overall.get("mae"), 2), "mean absolute error"),
        ui.kpi("RMSE", _num(overall.get("rmse"), 2), "root mean squared error"),
        ui.kpi("sMAPE", f"{_num(overall.get('smape'), 2)}%", "symmetric percentage"),
        ui.kpi(
            "Interval coverage",
            _pct(overall.get("interval_coverage")),
            "against nominal 95%",
        ),
    )

    rows = []
    for name, stats in baselines.items():
        rows.append(
            [
                name.replace("_", " "),
                _num(stats.get("mae"), 2),
                _signed(stats.get("skill_score_mae")),
                f"{stats.get('beaten_in_folds', 0)} / {overall.get('folds', 0)}",
            ]
        )
    table = ui.table(
        ["Baseline", "Baseline MAE", "Model skill", "Folds model won"],
        rows,
        note=(
            "Skill above zero means the model beat the baseline. A strongly "
            "negative skill against the seasonal-naive baseline shows the series "
            "carry weekly seasonality the model is not configured to capture."
        ),
    )

    header = html.P(
        f"Rolling-origin backtest across {counts.get('folds_scored', 0)} folds "
        f"from {counts.get('series_total', 0)} series.",
        className="page-subtitle",
    )

    return ui.section("Forecasting", header, kpis, ui.card(table))


def _render_anomaly(report: dict[str, Any]) -> Any:
    """Render the anomaly detection benchmark by injection magnitude.

    Args:
        report: The parsed anomaly report.

    Returns:
        The rendered section.

    """
    benchmark = report.get("injection_benchmark", {})
    overall = benchmark.get("overall", {})
    raw = overall.get("raw", {})
    adjusted = overall.get("adjusted", {})
    profile = report.get("real_data_profile", {})

    kpis = _kpi_grid(
        ui.kpi("Recall", _pct(overall.get("recall")), "injected faults recovered"),
        ui.kpi(
            "Precision (raw)",
            _pct(raw.get("precision")),
            "bounded by contamination",
            tone="warn",
        ),
        ui.kpi(
            "Precision (adjusted)",
            _pct(adjusted.get("precision")),
            "contamination ceiling removed",
        ),
        ui.kpi("Flag rate", _pct(profile.get("flag_rate")), "on unmodified data"),
    )

    rows = []
    for label, stats in benchmark.get("by_magnitude", {}).items():
        stats_raw = stats.get("raw", {})
        stats_adj = stats.get("adjusted", {})
        rows.append(
            [
                label.replace("_", " "),
                _pct(stats.get("recall") or stats_raw.get("recall")),
                _pct(stats_raw.get("precision")),
                _pct(stats_adj.get("precision")),
            ]
        )
    table = ui.table(
        ["Magnitude", "Recall", "Precision (raw)", "Precision (adj.)"],
        rows,
        note=(
            "Detection is reliable for large deviations and unreliable for small "
            "ones. Raw precision is capped by the detector's contamination "
            "setting; the adjusted column removes that structural ceiling."
        ),
    )

    return ui.section("Anomaly detection", kpis, ui.card(table))


def _render_risk(report: dict[str, Any]) -> Any:
    """Render the risk scoring evaluation across its scored domains.

    Args:
        report: The parsed risk report.

    Returns:
        The rendered section.

    """
    domains = report.get("domains", {})

    blocks: list[Any] = [
        html.P(
            "No classification metrics are reported: the scored entities carry "
            "no failure labels. What can be measured without labels is shown "
            "instead \u2014 weight sensitivity and agreement with an independent "
            "observable the score never reads.",
            className="page-subtitle",
        )
    ]

    for domain, block in domains.items():
        if not block.get("entities"):
            continue
        sensitivity = block.get("weight_sensitivity", {})
        validity = block.get("convergent_validity", {}).get("metrics", {})

        rows = []
        for component, stats in sensitivity.items():
            rows.append(
                [
                    component,
                    _num(stats.get("weight"), 2),
                    _num(stats.get("rank_agreement_tau")),
                    _num(stats.get("mean_absolute_score_change"), 2),
                ]
            )
        sensitivity_table = ui.table(
            ["Component", "Weight", "Rank agreement \u03c4", "Mean score change"],
            rows,
            note=(
                "A rank agreement near 1 means zeroing that component barely "
                "changes the ordering \u2014 the component is not contributing "
                "despite its weight."
            ),
        )

        validity_rows = []
        for metric, stats in validity.items():
            if "note" in stats:
                validity_rows.append(
                    [metric, str(stats.get("entities", "")), stats["note"]]
                )
            else:
                validity_rows.append(
                    [
                        metric,
                        str(stats.get("entities", "")),
                        _num(stats.get("spearman")),
                    ]
                )
        validity_table = ui.table(
            ["Observable", "Entities", "Spearman"],
            validity_rows,
            note=(
                "Correlation with an observable the score never reads. Weak or "
                "negative agreement shows the score ranks deterioration, not "
                "current condition."
            ),
        )

        blocks.append(
            ui.section(
                f"{domain.capitalize()} \u2014 {block.get('entities', 0)} entities",
                ui.card(sensitivity_table),
                ui.card(validity_table),
            )
        )

    return ui.section("Risk scoring", *blocks)


# ============================================================
# Loaders
# ============================================================


def _safe_report(slug: str, token: str | None) -> tuple[dict[str, Any] | None, Any]:
    """Fetch one report, returning either its content or a rendered message.

    A missing or failed report is not fatal to the page: the message is returned
    to be shown in place of that one section while the others still render.

    Args:
        slug: The report slug to fetch.
        token: Caller's access token.

    Returns:
        A tuple of the report content (or None) and a feedback element (or None).

    """
    try:
        return evaluation_report(slug, token=token), None
    except APIError as exc:
        logger.warning(
            "Callback evaluation could not load report",
            extra={"slug": slug},
            exc_info=True,
        )
        message = (
            feedback.empty(str(exc))
            if exc.status_code == 404
            else feedback.error(f"Could not load the {slug} report: {exc}")
        )
        return None, message


@callback(
    Output(ids.EVAL_RAG_SECTION, "children"),
    Output(ids.EVAL_ML_SECTION, "children"),
    Input(ids.EVAL_INIT, "n_intervals"),
    State(ids.ACCESS_TOKEN, "data"),
)
def load_evaluation(_init: int | None, token: str | None) -> tuple[Any, Any]:
    """Load every evaluation report and render the RAG and ML sections.

    Each report is fetched independently so that one missing report degrades a
    single section rather than the whole page.

    Args:
        _init: Interval tick that triggers the initial load.
        token: Caller's access token.

    Returns:
        The rendered RAG section and ML section.

    """
    retrieval, retrieval_msg = _safe_report("retrieval", token)
    threshold, threshold_msg = _safe_report("threshold", token)
    generation, generation_msg = _safe_report("generation", token)
    forecast, forecast_msg = _safe_report("forecast", token)
    anomaly, anomaly_msg = _safe_report("anomaly", token)
    risk, risk_msg = _safe_report("risk", token)

    rag_children = [
        html.H2("Retrieval-augmented generation", className="section-title")
    ]
    rag_children.append(retrieval_msg or _render_retrieval(retrieval))
    rag_children.append(threshold_msg or _render_threshold(threshold))
    rag_children.append(generation_msg or _render_generation(generation))
    rag_children.append(_render_failure_matrix(retrieval, generation))

    ml_children = [html.H2("Machine learning", className="section-title")]
    ml_children.append(forecast_msg or _render_forecast(forecast))
    ml_children.append(anomaly_msg or _render_anomaly(anomaly))
    ml_children.append(risk_msg or _render_risk(risk))

    logger.info(
        "Evaluation page rendered",
        extra={
            "rag_loaded": sum(
                x is not None for x in (retrieval, threshold, generation)
            ),
            "ml_loaded": sum(x is not None for x in (forecast, anomaly, risk)),
        },
    )

    return html.Div(rag_children), html.Div(ml_children, style={"marginTop": "3rem"})