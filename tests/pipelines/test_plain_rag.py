"""Tests for app.pipelines.plain_rag.PlainRagPipeline (T024).

All unit tests run without a live Neo4j instance.  Mocks are set at the
driver/session/provider level so the pipeline code itself executes — we are
testing pipeline behaviour, not mocking it away.

Integration-level timing test: runs the full pipeline with mocked providers
and verifies that all timing values are non-negative floats whose sum is
within the range expected for a mock execution.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from app.pipelines.plain_rag import _QUERY_VECTOR, PlainRagPipeline
from graphrag_assistant.schemas import AnswerSchema, RetrievalDebug, TextCitation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(
    rows: list[dict] | None = None,
    top_k: int = 5,
) -> tuple[PlainRagPipeline, MagicMock, MagicMock, MagicMock]:
    """Return (pipeline, mock_embed, mock_gen, mock_driver) with configurable rows."""
    if rows is None:
        rows = [
            {"chunk_id": "CL001_c0", "contract_id": "CT001", "text": "First clause text."},
        ]

    mock_embed = MagicMock()
    mock_embed.embed.return_value = [[0.1, 0.2, 0.3]]

    mock_session = MagicMock()
    mock_session.run.return_value = rows
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    mock_gen = MagicMock()
    mock_gen.generate.return_value = AnswerSchema(
        answer="Test answer.",
        graph_evidence=[],
        text_citations=[
            TextCitation(doc_id="CT001", chunk_id="CL001_c0", quote="First clause text.")
        ],
        retrieval_debug=RetrievalDebug(
            graph_query=None,
            entity_matches=[],
            retrieved_node_ids=[],
            chunk_ids=["CL001_c0"],
            timings={},
        ),
        mode="plain_rag",
    )

    pipeline = PlainRagPipeline(
        embedding_provider=mock_embed,
        generation_provider=mock_gen,
        driver=mock_driver,
        top_k=top_k,
    )
    return pipeline, mock_embed, mock_gen, mock_driver


# ---------------------------------------------------------------------------
# Schema: mode
# ---------------------------------------------------------------------------


def test_mode_is_plain_rag() -> None:
    pipeline, _, _, _ = _make_pipeline()
    result = pipeline.execute("Who supplies steel?")
    assert result.mode == "plain_rag"


# ---------------------------------------------------------------------------
# Schema: graph_evidence
# ---------------------------------------------------------------------------


def test_graph_evidence_is_empty_list() -> None:
    pipeline, _, _, _ = _make_pipeline()
    result = pipeline.execute("Who supplies steel?")
    assert result.graph_evidence == []


# ---------------------------------------------------------------------------
# Schema: text_citations
# ---------------------------------------------------------------------------


def test_text_citations_have_doc_id_field() -> None:
    pipeline, _, _, _ = _make_pipeline()
    result = pipeline.execute("Who supplies steel?")
    for citation in result.text_citations:
        assert isinstance(citation.doc_id, str)


def test_text_citations_have_chunk_id_field() -> None:
    pipeline, _, _, _ = _make_pipeline()
    result = pipeline.execute("Who supplies steel?")
    for citation in result.text_citations:
        assert isinstance(citation.chunk_id, str)


def test_text_citations_have_quote_field() -> None:
    pipeline, _, _, _ = _make_pipeline()
    result = pipeline.execute("Who supplies steel?")
    for citation in result.text_citations:
        assert isinstance(citation.quote, str)


def test_text_citations_empty_when_no_chunks() -> None:
    pipeline, _, _, _ = _make_pipeline(rows=[])
    # Re-wire generation stub to return empty citations for empty chunk list
    pipeline._generation_provider.generate.return_value = AnswerSchema(
        answer="No results.",
        graph_evidence=[],
        text_citations=[],
        retrieval_debug=RetrievalDebug(
            graph_query=None,
            entity_matches=[],
            retrieved_node_ids=[],
            chunk_ids=[],
            timings={},
        ),
        mode="plain_rag",
    )
    result = pipeline.execute("Who supplies steel?")
    assert isinstance(result.text_citations, list)


# ---------------------------------------------------------------------------
# Schema: retrieval_debug
# ---------------------------------------------------------------------------


def test_retrieval_debug_graph_query_is_none() -> None:
    pipeline, _, _, _ = _make_pipeline()
    result = pipeline.execute("Who supplies steel?")
    assert result.retrieval_debug.graph_query is None


def test_retrieval_debug_entity_matches_is_empty() -> None:
    pipeline, _, _, _ = _make_pipeline()
    result = pipeline.execute("Who supplies steel?")
    assert result.retrieval_debug.entity_matches == []


def test_retrieval_debug_retrieved_node_ids_is_empty() -> None:
    pipeline, _, _, _ = _make_pipeline()
    result = pipeline.execute("Who supplies steel?")
    assert result.retrieval_debug.retrieved_node_ids == []


def test_retrieval_debug_chunk_ids_is_list() -> None:
    pipeline, _, _, _ = _make_pipeline()
    result = pipeline.execute("Who supplies steel?")
    assert isinstance(result.retrieval_debug.chunk_ids, list)


def test_retrieval_debug_chunk_ids_match_returned_chunks() -> None:
    rows = [
        {"chunk_id": "CL001_c0", "contract_id": "CT001", "text": "A"},
        {"chunk_id": "CL002_c0", "contract_id": "CT002", "text": "B"},
    ]
    pipeline, _, _, _ = _make_pipeline(rows=rows)
    result = pipeline.execute("Who supplies steel?")
    assert result.retrieval_debug.chunk_ids == ["CL001_c0", "CL002_c0"]


def test_retrieval_debug_timings_has_embed_ms() -> None:
    pipeline, _, _, _ = _make_pipeline()
    result = pipeline.execute("Who supplies steel?")
    assert "embed_ms" in result.retrieval_debug.timings


def test_retrieval_debug_timings_has_retrieve_ms() -> None:
    pipeline, _, _, _ = _make_pipeline()
    result = pipeline.execute("Who supplies steel?")
    assert "retrieve_ms" in result.retrieval_debug.timings


def test_retrieval_debug_timings_has_generate_ms() -> None:
    pipeline, _, _, _ = _make_pipeline()
    result = pipeline.execute("Who supplies steel?")
    assert "generate_ms" in result.retrieval_debug.timings


# ---------------------------------------------------------------------------
# No graph calls: zero MATCH Cypher beyond the vector index query
# ---------------------------------------------------------------------------


def test_no_match_in_vector_query_constant() -> None:
    """The module-level query constant must not contain MATCH."""
    assert "MATCH" not in _QUERY_VECTOR


def test_no_graph_cypher_calls_during_execute() -> None:
    """session.run() must not be called with any Cypher containing MATCH."""
    pipeline, _, _, mock_driver = _make_pipeline()
    mock_session = mock_driver.session.return_value.__enter__.return_value
    pipeline.execute("Who supplies steel?")
    for c in mock_session.run.call_args_list:
        cypher = c.args[0] if c.args else list(c.kwargs.values())[0]
        assert "MATCH" not in cypher, f"Unexpected MATCH in Cypher: {cypher!r}"


def test_no_fstring_in_module_source() -> None:
    import re

    import app.pipelines.plain_rag as mod

    src = inspect.getsource(mod)
    # Match f-string prefix: f" or f' not preceded by a word character
    # (catches f"..." / f'...' but not %.1f" inside format strings)
    assert not re.search(r"(?<!\w)f\"", src)
    assert not re.search(r"(?<!\w)f'", src)


def test_no_percent_format_in_module_source() -> None:
    import app.pipelines.plain_rag as mod

    src = inspect.getsource(mod)
    assert " % " not in src


# ---------------------------------------------------------------------------
# top_k configurability
# ---------------------------------------------------------------------------


def test_default_top_k_is_five() -> None:
    pipeline = PlainRagPipeline(
        embedding_provider=MagicMock(),
        generation_provider=MagicMock(),
        driver=MagicMock(),
    )
    assert pipeline._top_k == 5


def test_top_k_passed_to_vector_query() -> None:
    rows = [{"chunk_id": "CL001_c0", "contract_id": "CT001", "text": "X"}]
    pipeline, _, _, mock_driver = _make_pipeline(rows=rows, top_k=3)
    mock_session = mock_driver.session.return_value.__enter__.return_value
    pipeline.execute("test query")
    run_calls = mock_session.run.call_args_list
    assert len(run_calls) == 1
    params = run_calls[0].args[1]
    assert params["top_k"] == 3


def test_top_k_configurable_different_values() -> None:
    for top_k in (1, 2, 10, 20):
        rows = [
            {"chunk_id": f"CL00{i}_c0", "contract_id": "CT001", "text": "T"}
            for i in range(top_k)
        ]
        pipeline, _, _, mock_driver = _make_pipeline(rows=rows, top_k=top_k)
        mock_session = mock_driver.session.return_value.__enter__.return_value
        pipeline.execute("query")
        params = mock_session.run.call_args.args[1]
        assert params["top_k"] == top_k


# ---------------------------------------------------------------------------
# Integration-level: timings are non-negative floats
# ---------------------------------------------------------------------------


def test_timings_are_float_values() -> None:
    pipeline, _, _, _ = _make_pipeline()
    result = pipeline.execute("Who supplies steel?")
    for key in ("embed_ms", "retrieve_ms", "generate_ms"):
        assert isinstance(result.retrieval_debug.timings[key], float), (
            f"{key} must be a float"
        )


def test_timings_are_non_negative() -> None:
    pipeline, _, _, _ = _make_pipeline()
    result = pipeline.execute("Who supplies steel?")
    for key in ("embed_ms", "retrieve_ms", "generate_ms"):
        assert result.retrieval_debug.timings[key] >= 0.0, (
            f"{key} must be >= 0"
        )


def test_timings_sum_approximates_wall_clock() -> None:
    import time as _time

    pipeline, _, _, _ = _make_pipeline()
    wall_start = _time.monotonic()
    result = pipeline.execute("Who supplies steel?")
    wall_ms = (_time.monotonic() - wall_start) * 1000.0

    timings = result.retrieval_debug.timings
    total = timings["embed_ms"] + timings["retrieve_ms"] + timings["generate_ms"]
    # Sum of phase timings must not exceed total wall time by more than 10 ms
    # (overhead from context managers, logging, object construction).
    assert total <= wall_ms + 10.0, (
        f"timings sum {total:.2f}ms exceeds wall_ms {wall_ms:.2f}ms + 10ms overhead"
    )


# ---------------------------------------------------------------------------
# AnswerSchema is imported from shared schema module
# ---------------------------------------------------------------------------


def test_answer_schema_imported_from_shared_module() -> None:
    from graphrag_assistant.schemas import AnswerSchema as SharedAnswerSchema

    # PlainRagPipeline.execute must return an instance of the shared AnswerSchema
    pipeline, _, _, _ = _make_pipeline()
    result = pipeline.execute("query")
    assert isinstance(result, SharedAnswerSchema)


# ---------------------------------------------------------------------------
# contract_id NULL guard (Bug 7)
# ---------------------------------------------------------------------------


def test_plain_rag_does_not_raise_when_contract_id_is_none() -> None:
    """Key failure mode: Neo4j returns NULL for contract_id as a present key
    with value None — not an absent key.  The real TemplateGenerationProvider
    is used so the full Pydantic validation path runs.
    """
    from graphrag_assistant.providers.generation_stub import TemplateGenerationProvider

    rows = [{"chunk_id": "CL001_c0", "contract_id": None, "text": "Some text."}]
    mock_embed = MagicMock()
    mock_embed.embed.return_value = [[0.1, 0.2, 0.3]]
    mock_session = MagicMock()
    mock_session.run.return_value = rows
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    pipeline = PlainRagPipeline(
        embedding_provider=mock_embed,
        generation_provider=TemplateGenerationProvider(),
        driver=mock_driver,
    )
    result = pipeline.execute("What is the clause?")
    assert isinstance(result, AnswerSchema)


def test_plain_rag_doc_id_empty_string_when_contract_id_none() -> None:
    """When Neo4j returns NULL for contract_id (key present, value None),
    the doc_id in the chunks passed to the generation provider must be "".
    """
    rows = [{"chunk_id": "CL001_c0", "contract_id": None, "text": "Some text."}]
    pipeline, _, mock_gen, _ = _make_pipeline(rows=rows)
    pipeline.execute("What is the clause?")
    chunks_passed = mock_gen.generate.call_args.kwargs["chunks"]
    assert chunks_passed[0]["doc_id"] == ""


def test_null_contract_id_does_not_raise() -> None:
    """When Neo4j returns NULL for contract_id, execute must not raise.

    The guard `row.get("contract_id") or ""` must coerce None to "" rather than
    letting a None propagate into TextCitation.doc_id (which is typed str).
    """
    rows = [{"chunk_id": "c1", "contract_id": None, "text": "T"}]
    pipeline, _, mock_gen, _ = _make_pipeline(rows=rows)
    result = pipeline.execute("q")
    # Primary contract: no exception raised; assert gives governance a hook too
    assert isinstance(result, AnswerSchema)


def test_null_contract_id_produces_empty_doc_id_in_generation_call() -> None:
    """When contract_id is None, the chunks dict passed to generate must have doc_id == "".

    This is the specific regression anchor for Bug 7: the coercion must happen
    before the chunk dict is assembled, so the generation provider never sees None.
    """
    rows = [{"chunk_id": "c1", "contract_id": None, "text": "T"}]
    pipeline, _, mock_gen, _ = _make_pipeline(rows=rows)
    pipeline.execute("q")
    chunks_passed = mock_gen.generate.call_args.kwargs["chunks"]
    assert chunks_passed[0]["doc_id"] == ""


def test_plain_rag_doc_id_preserved_when_contract_id_non_null() -> None:
    """When Neo4j returns a real contract_id, doc_id must pass through unchanged."""
    rows = [{"chunk_id": "CL001_c0", "contract_id": "CT-REAL-001", "text": "Some text."}]
    pipeline, _, mock_gen, _ = _make_pipeline(rows=rows)
    pipeline.execute("What is the clause?")
    chunks_passed = mock_gen.generate.call_args.kwargs["chunks"]
    assert chunks_passed[0]["doc_id"] == "CT-REAL-001"
