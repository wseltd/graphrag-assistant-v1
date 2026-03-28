"""Tests for app.routers.plain_rag — POST /query/plain-rag (T030.b).

Covers: request validation, retrieval_debug completeness, graph_evidence
format, missing-debug-field 500, and happy-path responses.
All tests run without a live Neo4j instance.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.plain_rag_router import router
from graphrag_assistant.schemas import AnswerSchema, GraphFact, RetrievalDebug

_VALID_KEY = "dev-key-change-in-prod"
_URL = "/query/plain-rag"
_PATCH = "app.routers.plain_rag_router.PlainRagPipeline"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _debug(
    *,
    graph_query: str | None = None,
    entity_matches: list | None = None,
    retrieved_node_ids: list | None = None,
    chunk_ids: list | None = None,
    timings: dict | None = None,
) -> RetrievalDebug:
    return RetrievalDebug(
        graph_query=graph_query,
        entity_matches=entity_matches if entity_matches is not None else [],
        retrieved_node_ids=retrieved_node_ids if retrieved_node_ids is not None else [],
        chunk_ids=chunk_ids if chunk_ids is not None else [],
        timings=timings if timings is not None else {},
    )


def _answer(
    *,
    answer: str = "ok",
    graph_evidence: list | None = None,
    text_citations: list | None = None,
    debug: RetrievalDebug | None = None,
    mode: str = "plain_rag",
) -> AnswerSchema:
    return AnswerSchema(
        answer=answer,
        graph_evidence=graph_evidence or [],
        text_citations=text_citations or [],
        retrieval_debug=debug or _debug(),
        mode=mode,
    )


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.neo4j_driver = MagicMock()
    app.state.embedding_provider = MagicMock()
    app.state.generation_provider = MagicMock()
    return app


# ---------------------------------------------------------------------------
# Group A: request validation (2 tests)
# ---------------------------------------------------------------------------


def test_empty_question_field_rejected() -> None:
    """POST with question='' must return 422 (min_length=1 constraint)."""
    app = _make_app()
    with patch(_PATCH) as cls:
        cls.return_value.execute.return_value = _answer()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(_URL, json={"question": ""}, headers={"X-Api-Key": _VALID_KEY})
    assert resp.status_code == 422


def test_missing_question_key_rejected() -> None:
    """POST without a 'question' key must return 422."""
    app = _make_app()
    with patch(_PATCH) as cls:
        cls.return_value.execute.return_value = _answer()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(_URL, json={"top_k": 5}, headers={"X-Api-Key": _VALID_KEY})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Group B: plain-rag retrieval_debug completeness (3 tests)
# ---------------------------------------------------------------------------


def test_plain_rag_debug_has_all_five_fields() -> None:
    """Normal result: all five debug fields are present in the response."""
    dbg = _debug(
        graph_query=None,
        entity_matches=[],
        retrieved_node_ids=[],
        chunk_ids=["c1"],
        timings={"embed_ms": 1.0, "retrieve_ms": 2.0, "generate_ms": 0.5},
    )
    app = _make_app()
    with patch(_PATCH) as cls:
        cls.return_value.execute.return_value = _answer(debug=dbg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            _URL,
            json={"question": "Who supplies steel?"},
            headers={"X-Api-Key": _VALID_KEY},
        )
    assert resp.status_code == 200
    rd = resp.json()["retrieval_debug"]
    for field in ("graph_query", "entity_matches", "retrieved_node_ids", "chunk_ids", "timings"):
        assert field in rd, f"retrieval_debug missing field: {field}"


def test_plain_rag_debug_fields_empty_on_zero_results() -> None:
    """Zero-result retrieval: debug fields present and empty, not null."""
    dbg = _debug(
        entity_matches=[],
        retrieved_node_ids=[],
        chunk_ids=[],
        timings={"embed_ms": 0.1, "retrieve_ms": 0.2, "generate_ms": 0.05},
    )
    app = _make_app()
    with patch(_PATCH) as cls:
        cls.return_value.execute.return_value = _answer(debug=dbg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            _URL,
            json={"question": "Unmatched query"},
            headers={"X-Api-Key": _VALID_KEY},
        )
    assert resp.status_code == 200
    rd = resp.json()["retrieval_debug"]
    assert rd["entity_matches"] == []
    assert rd["retrieved_node_ids"] == []
    assert rd["chunk_ids"] == []
    assert rd["timings"] is not None


def test_missing_debug_field_triggers_500() -> None:
    """Pipeline result whose debug object lacks a field must return 500."""
    broken_debug = MagicMock(spec=[])  # spec=[] — hasattr always False
    broken_result = MagicMock(spec=AnswerSchema)
    broken_result.retrieval_debug = broken_debug

    app = _make_app()
    with patch(_PATCH) as cls:
        cls.return_value.execute.return_value = broken_result
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            _URL,
            json={"question": "test"},
            headers={"X-Api-Key": _VALID_KEY},
        )
    assert resp.status_code == 500
    assert "retrieval_debug" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Group C: simulated graph-rag debug completeness (3 tests)
# ---------------------------------------------------------------------------


def test_simulated_graph_debug_has_graph_query_field() -> None:
    """When pipeline returns a graph_query string, the field is preserved."""
    dbg = _debug(
        graph_query="MATCH (c:Company) RETURN c",
        entity_matches=["Axiom Corp"],
        retrieved_node_ids=["C001"],
        chunk_ids=["CT001_c0"],
        timings={"embed_ms": 1.0, "retrieve_ms": 3.0, "generate_ms": 1.5},
    )
    app = _make_app()
    with patch(_PATCH) as cls:
        cls.return_value.execute.return_value = _answer(debug=dbg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            _URL,
            json={"question": "Who are Axiom Corp directors?"},
            headers={"X-Api-Key": _VALID_KEY},
        )
    assert resp.status_code == 200
    rd = resp.json()["retrieval_debug"]
    assert "graph_query" in rd


def test_simulated_graph_debug_entity_matches_populated() -> None:
    """entity_matches field is returned when populated by pipeline."""
    dbg = _debug(
        entity_matches=["Axiom Corp", "Bob Smith"],
        retrieved_node_ids=["C001", "P002"],
        chunk_ids=["CT001_c0"],
        timings={"embed_ms": 1.0, "retrieve_ms": 3.0, "generate_ms": 1.5},
    )
    app = _make_app()
    with patch(_PATCH) as cls:
        cls.return_value.execute.return_value = _answer(debug=dbg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            _URL,
            json={"question": "Axiom Corp directors"},
            headers={"X-Api-Key": _VALID_KEY},
        )
    assert resp.status_code == 200
    rd = resp.json()["retrieval_debug"]
    assert "entity_matches" in rd
    assert len(rd["entity_matches"]) == 2


def test_simulated_graph_debug_retrieved_node_ids_populated() -> None:
    """retrieved_node_ids field is returned when populated by pipeline."""
    dbg = _debug(
        retrieved_node_ids=["C001", "C002", "P003"],
        chunk_ids=["CT001_c0", "CT002_c0"],
        timings={"embed_ms": 1.0, "retrieve_ms": 3.0, "generate_ms": 1.5},
    )
    app = _make_app()
    with patch(_PATCH) as cls:
        cls.return_value.execute.return_value = _answer(debug=dbg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            _URL,
            json={"question": "Supply chain for Axiom"},
            headers={"X-Api-Key": _VALID_KEY},
        )
    assert resp.status_code == 200
    rd = resp.json()["retrieval_debug"]
    assert "retrieved_node_ids" in rd
    assert len(rd["retrieved_node_ids"]) == 3


# ---------------------------------------------------------------------------
# Group D: graph_evidence format (3 tests)
# ---------------------------------------------------------------------------


def test_graph_evidence_entry_has_source_id() -> None:
    """Each graph_evidence entry must contain source_id."""
    ge = [GraphFact(source_id="C001", target_id="P002", label="DIRECTOR_OF")]
    app = _make_app()
    with patch(_PATCH) as cls:
        cls.return_value.execute.return_value = _answer(graph_evidence=ge)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            _URL,
            json={"question": "Director query"},
            headers={"X-Api-Key": _VALID_KEY},
        )
    assert resp.status_code == 200
    entry = resp.json()["graph_evidence"][0]
    assert "source_id" in entry
    assert entry["source_id"] == "C001"


def test_graph_evidence_entry_has_target_id() -> None:
    """Each graph_evidence entry must contain target_id."""
    ge = [GraphFact(source_id="C001", target_id="P002", label="DIRECTOR_OF")]
    app = _make_app()
    with patch(_PATCH) as cls:
        cls.return_value.execute.return_value = _answer(graph_evidence=ge)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            _URL,
            json={"question": "Director query"},
            headers={"X-Api-Key": _VALID_KEY},
        )
    assert resp.status_code == 200
    entry = resp.json()["graph_evidence"][0]
    assert "target_id" in entry
    assert entry["target_id"] == "P002"


def test_graph_evidence_entry_has_label() -> None:
    """Each graph_evidence entry must contain label."""
    ge = [GraphFact(source_id="C001", target_id="P002", label="DIRECTOR_OF")]
    app = _make_app()
    with patch(_PATCH) as cls:
        cls.return_value.execute.return_value = _answer(graph_evidence=ge)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            _URL,
            json={"question": "Director query"},
            headers={"X-Api-Key": _VALID_KEY},
        )
    assert resp.status_code == 200
    entry = resp.json()["graph_evidence"][0]
    assert "label" in entry
    assert entry["label"] == "DIRECTOR_OF"


# ---------------------------------------------------------------------------
# Group E: happy-path integration (2 tests)
# ---------------------------------------------------------------------------


def test_happy_path_returns_200_with_answer_schema() -> None:
    """Valid request with valid key returns 200 and all five top-level fields."""
    dbg = _debug(
        chunk_ids=["CT001_c0"],
        timings={"embed_ms": 1.0, "retrieve_ms": 2.0, "generate_ms": 0.5},
    )
    ans = _answer(answer="Axiom Corp is supplied by Steel Corp.", debug=dbg)
    app = _make_app()
    with patch(_PATCH) as cls:
        cls.return_value.execute.return_value = ans
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            _URL,
            json={"question": "Who supplies steel to Axiom Corp?"},
            headers={"X-Api-Key": _VALID_KEY},
        )
    assert resp.status_code == 200
    body = resp.json()
    for field in ("answer", "graph_evidence", "text_citations", "retrieval_debug", "mode"):
        assert field in body, f"response missing top-level field: {field}"


def test_happy_path_mode_is_plain_rag() -> None:
    """Response mode must be 'plain_rag'."""
    app = _make_app()
    with patch(_PATCH) as cls:
        cls.return_value.execute.return_value = _answer(mode="plain_rag")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            _URL,
            json={"question": "test question"},
            headers={"X-Api-Key": _VALID_KEY},
        )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "plain_rag"
