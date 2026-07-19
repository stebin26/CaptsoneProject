"""Unit tests for the copilot and the document assistant.

The agent runs on a 3B local model, so the design assumption throughout is that
the model will sometimes emit nonsense: a tool that does not exist, arguments of
the wrong shape, or a call that makes the tool itself fail. These tests pin the
contract that every one of those returns a failed result the loop can hand back
to the model, rather than an exception that ends the conversation.

There is deliberately no planner test: this project has no LLM planner. That was
an empirical decision -- the 3B model could not follow multi-step plans -- so
tool *selection* is what is tested instead.

No test here contacts Ollama or the vector store.
"""

from __future__ import annotations

import pytest

# Skipped rather than failed where a service is not mounted, so the same suite
# stays meaningful in every image it runs in.
base = pytest.importorskip("agent.tools.base")
selector = pytest.importorskip("agent.tools.selector")
llm = pytest.importorskip("agent.llm")


# ============================================================
# Tool dispatch -- the model is assumed to misbehave
# ============================================================


def test_unknown_tool_returns_a_failed_result_listing_what_exists():
    """A hallucinated tool name comes back as a result, not an exception."""
    result = base.dispatch("teleport", {}, {"hub_overview": lambda: None})

    assert result.ok is False
    assert "teleport" in result.summary
    assert "hub_overview" in result.summary


def test_unknown_tool_with_an_empty_registry_still_answers():
    """An empty registry is reported plainly instead of failing to format."""
    result = base.dispatch("anything", {}, {})

    assert result.ok is False
    assert "none" in result.summary.lower()


def test_a_tool_that_raises_is_reported_not_propagated():
    """A crash inside a tool is converted into a failed result.

    A raised exception here would end the conversation; a failed result lets the
    model try something else.
    """

    def broken_tool() -> str:
        raise RuntimeError("the hub is unreachable")

    result = base.dispatch("broken_tool", {}, {"broken_tool": broken_tool})

    assert result.ok is False
    assert isinstance(result.summary, str) and result.summary


def test_wrong_arguments_are_reported_not_propagated():
    """Arguments the tool cannot accept produce a failed result."""

    def needs_dataset(dataset_id: int) -> str:
        return f"dataset {dataset_id}"

    result = base.dispatch(
        "needs_dataset", {"nonexistent_argument": 1}, {"needs_dataset": needs_dataset}
    )

    assert result.ok is False


def test_a_successful_tool_result_reaches_the_model_as_its_summary():
    """The model observes the summary, not the raw structured payload."""
    result = base.ToolResult(ok=True, summary="3 datasets found", data={"count": 3})

    assert result.to_model_text() == "3 datasets found"


# ============================================================
# Tool selection -- the pre-filter that replaces a planner
# ============================================================


def test_selection_always_exposes_a_usable_baseline():
    """Even an unrecognised question gets a working set of tools."""
    schemas, functions, groups = selector.select_tools("hello there")

    assert schemas, "the model must never be handed an empty tool list"
    assert set(functions) <= {s["function"]["name"] for s in schemas}


def test_causal_questions_expose_more_tools_than_a_bare_greeting():
    """Why-style questions widen the exposed surface.

    The selector co-selects the ML and intelligence groups for causal vocabulary,
    because answering "why" needs evidence a lookup alone cannot provide.
    """
    _, _, causal_groups = selector.select_tools(
        "why did production drop last month and what caused it"
    )
    _, _, greeting_groups = selector.select_tools("hello")

    assert len(causal_groups) > len(greeting_groups)


def test_every_selected_schema_is_backed_by_an_implementation():
    """A schema the model can call always has a function behind it."""
    schemas, functions, _ = selector.select_tools("show me the risk scores")

    for schema in schemas:
        assert schema["function"]["name"] in functions


# ============================================================
# Model client configuration
# ============================================================


def test_malformed_model_settings_fall_back_to_defaults(clean_env, caplog):
    """A typo in one setting must not stop the copilot from starting."""
    clean_env.setenv("OPS_AGENT_TEMPERATURE", "hot")
    clean_env.setenv("OPS_AGENT_LLM_RETRIES", "three")

    with caplog.at_level("WARNING"):
        config = llm.LLMConfig()

    assert config.temperature == 0.1
    assert config.max_retries == 3
    assert "OPS_AGENT_TEMPERATURE" in caplog.text


def test_valid_model_settings_are_honoured(clean_env):
    """Well-formed settings are used as given."""
    clean_env.setenv("OPS_AGENT_TEMPERATURE", "0.7")
    clean_env.setenv("OPS_AGENT_NUM_CTX", "8192")

    config = llm.LLMConfig()

    assert config.temperature == pytest.approx(0.7)
    assert config.num_ctx == 8192


# ============================================================
# Document chunking
# ============================================================


def test_a_document_with_no_text_produces_no_chunks():
    """An empty or image-only document yields nothing to embed."""
    chunker = pytest.importorskip("chunker")
    extractor = pytest.importorskip("extractor")

    doc = extractor.ExtractedDocument(
        filename="blank.pdf",
        file_type="pdf",
        pages=[extractor.ExtractedPage(page_number=1, text="   ")],
    )

    assert chunker.chunk_document(doc) == []


def test_chunks_carry_their_page_and_position():
    """Every chunk keeps the provenance an answer needs to cite."""
    chunker = pytest.importorskip("chunker")
    extractor = pytest.importorskip("extractor")

    doc = extractor.ExtractedDocument(
        filename="manual.pdf",
        file_type="pdf",
        pages=[
            extractor.ExtractedPage(page_number=1, text="Error E12 means overheating. "
                                    * 40),
            extractor.ExtractedPage(page_number=2, text="Reset the controller. " * 40),
        ],
    )

    chunks = chunker.chunk_document(doc)

    assert chunks
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page_number for c in chunks} <= {1, 2}


def test_an_unsupported_file_type_is_rejected(tmp_path):
    """A file the extractor cannot read is refused before any parsing."""
    extractor = pytest.importorskip("extractor")
    path = tmp_path / "archive.zip"
    path.write_bytes(b"PK\x03\x04")

    with pytest.raises(ValueError):
        extractor.extract_document(path)


def test_a_missing_document_is_reported_as_an_extraction_error(tmp_path):
    """A path that no longer exists names the file it was looking for."""
    extractor = pytest.importorskip("extractor")

    with pytest.raises(extractor.ExtractionError, match="not found"):
        extractor.extract_document(tmp_path / "gone.pdf")
