"""Tests for app.routers.graph_rag — POST /query/graph-rag (T030.c).

Covers: request validation (2), retrieval_debug completeness for plain-rag
(3) and graph-rag (3), graph_evidence format (3), happy-path HTTP 200 (2),
doc_id provenance through chunks_dicts (3).
Total: 16 tests.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.pipelines.constrained_retrieval import RankedChunk
from app.pipelines.entity_resolver import EntityMatch
from app.pipelines.graph_traversal import GraphTraversalResult, Triple
from app.routers.graph_rag_router import router
from app.routers.plain_rag_router import router as plain_rag_router
from graphrag_assistant.providers.generation_stub import TemplateGenerationProvider
from graphrag_assistant.schemas import (
    AnswerSchema,
    GraphFact,
    RetrievalDebug,
    TextCitation,
)

_VALID_KEY = "dev-key-change-in-prod"

# ---------------------------------------------------------------------------
# Answer fixtures
# ---------------------------------------------------------------------------


def _make_plain_rag_answer() -> AnswerSchema:
    return AnswerSchema(
        answer="Plain answer.",
        graph_evidence=[],
        text_citations=[
            TextCitation(doc_id="CT001", chunk_id="CL001_c0", quote="Text.")
        ],
        retrieval_debug=RetrievalDebug(
            graph_query=None,
            entity_matches=[],
            retrieved_node_ids=[],
            chunk_ids=["CL001_c0"],
            timings={"embed_ms": 1.0, "retrieve_ms": 2.0, "generate_ms": 0.5},
        ),
        mode="plain_rag",
    )


def _make_graph_answer() -> AnswerSchema:
    return AnswerSchema(
        answer="Graph answer.",
        graph_evidence=[
            GraphFact(source_id="n1", target_id="n2", label="PARTY_TO")
        ],
        text_citations=[
            TextCitation(doc_id="CT001", chunk_id="c1", quote="Some text.")
        ],
        retrieval_debug=RetrievalDebug(
            graph_query="GRAPH_TRAVERSAL_2HOP",
            entity_matches=["C001"],
            retrieved_node_ids=["C001"],
            chunk_ids=["c1"],
            timings={"graph_ms": 1.0, "retrieve_ms": 2.0, "generate_ms": 0.5},
        ),
        mode="graph_rag",
    )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _call_plain_rag() -> object:
    """POST /query/plain-rag with mocked PlainRagPipeline."""
    answer = _make_plain_rag_answer()
    with patch("app.routers.plain_rag_router.PlainRagPipeline") as cls:
        cls.return_value.execute.return_value = answer
        app = FastAPI()
        app.include_router(plain_rag_router)
        app.state.neo4j_driver = MagicMock()
        app.state.embedding_provider = MagicMock()
        app.state.generation_provider = MagicMock()
        client = TestClient(app)
        return client.post(
            "/query/plain-rag",
            json={"question": "Who supplies steel?"},
            headers={"X-Api-Key": _VALID_KEY},
        )


def _call_graph_rag() -> object:
    """POST /query/graph-rag with mocked pipeline stages."""
    matches = [
        EntityMatch(node_id="C001", label="Company", name="Acme", score=1.0)
    ]
    triple = Triple(src="n1", rel="PARTY_TO", dst="n2")
    traversal = GraphTraversalResult(chunk_ids=["c1"], triples=[triple])
    chunks = [RankedChunk(chunk_id="c1", text="Some text.", score=0.9)]
    generated = _make_graph_answer()
    p_re = "app.routers.graph_rag_router.resolve_entities"
    p_ta = "app.routers.graph_rag_router.traverse_from_anchors"
    p_rc = "app.routers.graph_rag_router.retrieve_constrained"
    with patch(p_re, return_value=matches), \
         patch(p_ta, return_value=traversal), \
         patch(p_rc, return_value=chunks):
        app = FastAPI()
        app.include_router(router)
        app.state.neo4j_driver = MagicMock()
        app.state.embedding_provider = MagicMock()
        app.state.generation_provider = MagicMock()
        app.state.generation_provider.generate.return_value = generated
        client = TestClient(app)
        return client.post(
            "/query/graph-rag",
            json={"question": "Who supplies steel?"},
            headers={"X-Api-Key": _VALID_KEY},
        )


# ---------------------------------------------------------------------------
# Validation (2)
# ---------------------------------------------------------------------------


def test_empty_question_returns_422() -> None:
    app = FastAPI()
    app.include_router(router)
    app.state.neo4j_driver = MagicMock()
    app.state.embedding_provider = MagicMock()
    app.state.generation_provider = MagicMock()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/query/graph-rag",
        json={"question": ""},
        headers={"X-Api-Key": _VALID_KEY},
    )
    assert resp.status_code == 422


def test_missing_question_returns_422() -> None:
    app = FastAPI()
    app.include_router(router)
    app.state.neo4j_driver = MagicMock()
    app.state.embedding_provider = MagicMock()
    app.state.generation_provider = MagicMock()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/query/graph-rag",
        json={},
        headers={"X-Api-Key": _VALID_KEY},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Plain-rag retrieval_debug completeness (3)
# ---------------------------------------------------------------------------


def test_plain_rag_debug_has_graph_query_key() -> None:
    resp = _call_plain_rag()
    assert "graph_query" in resp.json()["retrieval_debug"]


def test_plain_rag_debug_chunk_ids_is_list() -> None:
    resp = _call_plain_rag()
    assert isinstance(resp.json()["retrieval_debug"]["chunk_ids"], list)


def test_plain_rag_debug_timings_is_dict() -> None:
    resp = _call_plain_rag()
    assert isinstance(resp.json()["retrieval_debug"]["timings"], dict)


# ---------------------------------------------------------------------------
# Graph-rag retrieval_debug completeness (3)
# ---------------------------------------------------------------------------


def test_graph_rag_debug_has_all_five_fields() -> None:
    resp = _call_graph_rag()
    debug = resp.json()["retrieval_debug"]
    expected = {
        "graph_query", "entity_matches", "retrieved_node_ids",
        "chunk_ids", "timings",
    }
    assert expected.issubset(debug.keys())


def test_graph_rag_debug_graph_query_is_non_null() -> None:
    resp = _call_graph_rag()
    gq = resp.json()["retrieval_debug"]["graph_query"]
    assert gq is not None
    assert isinstance(gq, str)


def test_graph_rag_debug_timings_contains_keys() -> None:
    resp = _call_graph_rag()
    timings = resp.json()["retrieval_debug"]["timings"]
    assert isinstance(timings, dict)
    assert len(timings) > 0


# ---------------------------------------------------------------------------
# Graph-rag graph_evidence format (3)
# ---------------------------------------------------------------------------


def test_graph_evidence_entry_has_source_id() -> None:
    resp = _call_graph_rag()
    evidence = resp.json()["graph_evidence"]
    assert len(evidence) > 0
    assert "source_id" in evidence[0]


def test_graph_evidence_entry_has_target_id() -> None:
    resp = _call_graph_rag()
    evidence = resp.json()["graph_evidence"]
    assert len(evidence) > 0
    assert "target_id" in evidence[0]


def test_graph_evidence_entry_has_label() -> None:
    resp = _call_graph_rag()
    evidence = resp.json()["graph_evidence"]
    assert len(evidence) > 0
    assert "label" in evidence[0]


# ---------------------------------------------------------------------------
# Happy-path HTTP 200 (2)
# ---------------------------------------------------------------------------


def test_plain_rag_happy_path_returns_200() -> None:
    resp = _call_plain_rag()
    assert resp.status_code == 200


def test_graph_rag_happy_path_returns_200() -> None:
    resp = _call_graph_rag()
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# doc_id provenance through chunks_dicts (3)
# ---------------------------------------------------------------------------


def test_graph_rag_text_citation_doc_id_from_ranked_chunk() -> None:
    """doc_id from RankedChunk must flow through chunks_dicts to TextCitation.

    Uses a real TemplateGenerationProvider so the full dict-to-TextCitation
    mapping executes end-to-end.  Asserts the response JSON carries the
    exact doc_id that was on the RankedChunk.
    """
    matches = [
        EntityMatch(node_id="C001", label="Company", name="Acme", score=1.0)
    ]
    traversal = GraphTraversalResult(chunk_ids=["c1"], triples=[])
    chunks = [
        RankedChunk(chunk_id="c1", doc_id="CT-ACME-001", text="Acme supplies steel.", score=0.9)
    ]

    p_re = "app.routers.graph_rag_router.resolve_entities"
    p_ta = "app.routers.graph_rag_router.traverse_from_anchors"
    p_rc = "app.routers.graph_rag_router.retrieve_constrained"
    with patch(p_re, return_value=matches), \
         patch(p_ta, return_value=traversal), \
         patch(p_rc, return_value=chunks):
        app = FastAPI()
        app.include_router(router)
        app.state.neo4j_driver = MagicMock()
        app.state.embedding_provider = MagicMock()
        app.state.generation_provider = TemplateGenerationProvider()
        client = TestClient(app)
        resp = client.post(
            "/query/graph-rag",
            json={"question": "Who supplies steel?"},
            headers={"X-Api-Key": _VALID_KEY},
        )

    assert resp.status_code == 200
    citations = resp.json()["text_citations"]
    assert len(citations) == 1
    assert citations[0]["doc_id"] == "CT-ACME-001"


def test_graph_rag_text_citation_doc_id_empty_when_ranked_chunk_doc_id_empty() -> None:
    """When RankedChunk.doc_id is empty, TextCitation.doc_id must also be empty.

    Verifies the fix does not introduce a default or fall-back value —
    an absent doc_id on the chunk stays absent in the citation.
    """
    matches = [
        EntityMatch(node_id="C001", label="Company", name="Acme", score=1.0)
    ]
    traversal = GraphTraversalResult(chunk_ids=["c1"], triples=[])
    # doc_id defaults to "" when not provided
    chunks = [RankedChunk(chunk_id="c1", text="Acme supplies steel.", score=0.9)]

    p_re = "app.routers.graph_rag_router.resolve_entities"
    p_ta = "app.routers.graph_rag_router.traverse_from_anchors"
    p_rc = "app.routers.graph_rag_router.retrieve_constrained"
    with patch(p_re, return_value=matches), \
         patch(p_ta, return_value=traversal), \
         patch(p_rc, return_value=chunks):
        app = FastAPI()
        app.include_router(router)
        app.state.neo4j_driver = MagicMock()
        app.state.embedding_provider = MagicMock()
        app.state.generation_provider = TemplateGenerationProvider()
        client = TestClient(app)
        resp = client.post(
            "/query/graph-rag",
            json={"question": "Who supplies steel?"},
            headers={"X-Api-Key": _VALID_KEY},
        )

    assert resp.status_code == 200
    citations = resp.json()["text_citations"]
    assert len(citations) == 1
    assert citations[0]["doc_id"] == ""


def test_graph_rag_generate_called_with_chunk_doc_id() -> None:
    """Spy confirms the exact doc_id value is passed into generate().

    Wraps TemplateGenerationProvider.generate with a spy so the call
    arguments are inspectable.  Asserts chunks[0]["doc_id"] in the generate
    call matches the doc_id on the RankedChunk.
    """
    matches = [
        EntityMatch(node_id="C001", label="Company", name="Acme", score=1.0)
    ]
    traversal = GraphTraversalResult(chunk_ids=["c1"], triples=[])
    chunks = [RankedChunk(chunk_id="c1", doc_id="CT-SPY-002", text="Spy test text.", score=0.8)]

    real_provider = TemplateGenerationProvider()
    spy_generate = MagicMock(wraps=real_provider.generate)
    real_provider.generate = spy_generate  # type: ignore[method-assign]

    p_re = "app.routers.graph_rag_router.resolve_entities"
    p_ta = "app.routers.graph_rag_router.traverse_from_anchors"
    p_rc = "app.routers.graph_rag_router.retrieve_constrained"
    with patch(p_re, return_value=matches), \
         patch(p_ta, return_value=traversal), \
         patch(p_rc, return_value=chunks):
        app = FastAPI()
        app.include_router(router)
        app.state.neo4j_driver = MagicMock()
        app.state.embedding_provider = MagicMock()
        app.state.generation_provider = real_provider
        client = TestClient(app)
        client.post(
            "/query/graph-rag",
            json={"question": "Who supplies steel?"},
            headers={"X-Api-Key": _VALID_KEY},
        )

    spy_generate.assert_called_once()
    chunks_arg = spy_generate.call_args.kwargs["chunks"]
    assert chunks_arg[0]["doc_id"] == "CT-SPY-002"


def test_generation_provider_receives_chunk_doc_id() -> None:
    """Router must pass chunk.doc_id into the chunks dict sent to generate().

    Mocks the generation provider to inspect call_args directly.  Asserts the
    router assembled chunks_dicts[0]["doc_id"] == "DOC-42" from the RankedChunk,
    not from a hardcoded fallback.
    """
    matches = [
        EntityMatch(node_id="C001", label="Company", name="Acme", score=1.0)
    ]
    traversal = GraphTraversalResult(chunk_ids=["c1"], triples=[])
    chunks = [RankedChunk(chunk_id="c1", doc_id="DOC-42", text="Some text.", score=0.9)]

    p_re = "app.routers.graph_rag_router.resolve_entities"
    p_ta = "app.routers.graph_rag_router.traverse_from_anchors"
    p_rc = "app.routers.graph_rag_router.retrieve_constrained"
    with patch(p_re, return_value=matches), \
         patch(p_ta, return_value=traversal), \
         patch(p_rc, return_value=chunks):
        app = FastAPI()
        app.include_router(router)
        app.state.neo4j_driver = MagicMock()
        app.state.embedding_provider = MagicMock()
        mock_provider = MagicMock()
        mock_provider.generate.return_value = _make_graph_answer()
        app.state.generation_provider = mock_provider
        client = TestClient(app)
        client.post(
            "/query/graph-rag",
            json={"question": "Who supplies steel?"},
            headers={"X-Api-Key": _VALID_KEY},
        )

    mock_provider.generate.assert_called_once()
    # The router calls generate(..., chunks=chunks_dicts) with keyword arg.
    chunks_arg = mock_provider.generate.call_args.kwargs["chunks"]
    assert chunks_arg[0]["doc_id"] == "DOC-42"


def test_text_citations_doc_id_non_empty_when_ranked_chunk_carries_doc_id() -> None:
    """doc_id from RankedChunk must appear in the HTTP response text_citations.

    Uses the real TemplateGenerationProvider so the full path from
    RankedChunk → chunks_dicts → TextCitation is exercised end-to-end.
    """
    matches = [
        EntityMatch(node_id="C001", label="Company", name="Acme", score=1.0)
    ]
    traversal = GraphTraversalResult(chunk_ids=["c1"], triples=[])
    chunks = [RankedChunk(chunk_id="c1", doc_id="DOC-42", text="Acme supplies steel.", score=0.9)]

    p_re = "app.routers.graph_rag_router.resolve_entities"
    p_ta = "app.routers.graph_rag_router.traverse_from_anchors"
    p_rc = "app.routers.graph_rag_router.retrieve_constrained"
    with patch(p_re, return_value=matches), \
         patch(p_ta, return_value=traversal), \
         patch(p_rc, return_value=chunks):
        app = FastAPI()
        app.include_router(router)
        app.state.neo4j_driver = MagicMock()
        app.state.embedding_provider = MagicMock()
        app.state.generation_provider = TemplateGenerationProvider()
        client = TestClient(app)
        resp = client.post(
            "/query/graph-rag",
            json={"question": "Who supplies steel?"},
            headers={"X-Api-Key": _VALID_KEY},
        )

    assert resp.status_code == 200
    citations = resp.json()["text_citations"]
    assert len(citations) == 1
    assert citations[0]["doc_id"] == "DOC-42"
