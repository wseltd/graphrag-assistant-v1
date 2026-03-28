"""Tests for app.routers.graph_rag — POST /query/graph-rag (T030.c).

Covers: request validation (2), retrieval_debug completeness for plain-rag
(3) and graph-rag (3), graph_evidence format (3), happy-path HTTP 200 (2).
Total: 13 tests.
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
