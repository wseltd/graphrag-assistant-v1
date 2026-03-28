"""Tests for app.routers.query — POST /query/plain-rag (T024).

Unit tests for the HTTP layer: auth, validation, response shape.
All tests run without a live Neo4j instance.  The FastAPI app is built
inline so each test controls app.state (driver, embedding_provider,
generation_provider) directly.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.query import PlainRagRequest, router
from graphrag_assistant.schemas import AnswerSchema, RetrievalDebug, TextCitation

_VALID_KEY = "dev-key-change-in-prod"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _make_answer(query: str = "q") -> AnswerSchema:
    return AnswerSchema(
        answer="Test answer.",
        graph_evidence=[],
        text_citations=[
            TextCitation(doc_id="CT001", chunk_id="CL001_c0", quote="Some text.")
        ],
        retrieval_debug=RetrievalDebug(
            graph_query=None,
            entity_matches=[],
            retrieved_node_ids=[],
            chunk_ids=["CL001_c0"],
            timings={"embed_ms": 1.5, "retrieve_ms": 2.0, "generate_ms": 0.5},
        ),
        mode="plain_rag",
    )


def _build_app(answer: AnswerSchema | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.neo4j_driver = MagicMock()
    app.state.embedding_provider = MagicMock()
    app.state.generation_provider = MagicMock()
    _answer = answer or _make_answer()

    with patch("app.routers.query.PlainRagPipeline") as mock_pipeline_cls:
        mock_instance = MagicMock()
        mock_instance.execute.return_value = _answer
        mock_pipeline_cls.return_value = mock_instance
        # Capture the class mock so tests can patch it later
        app.state._mock_pipeline_cls = mock_pipeline_cls
        app.state._mock_pipeline = mock_instance

    return app


# ---------------------------------------------------------------------------
# PlainRagRequest model
# ---------------------------------------------------------------------------


def test_plain_rag_request_defaults() -> None:
    req = PlainRagRequest(query="test")
    assert req.top_k == 5


def test_plain_rag_request_repr() -> None:
    req = PlainRagRequest(query="hello", top_k=3)
    r = repr(req)
    assert "PlainRagRequest" in r
    assert "hello" in r


def test_plain_rag_request_top_k_ge_1() -> None:
    with pytest.raises(ValueError):
        PlainRagRequest(query="test", top_k=0)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def test_missing_api_key_returns_422() -> None:
    answer = _make_answer()
    with patch("app.routers.query.PlainRagPipeline") as cls:
        cls.return_value.execute.return_value = answer
        app = FastAPI()
        app.include_router(router)
        app.state.neo4j_driver = MagicMock()
        app.state.embedding_provider = MagicMock()
        app.state.generation_provider = MagicMock()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/query/plain-rag", json={"query": "test"})
    assert resp.status_code == 422


def test_wrong_api_key_returns_401() -> None:
    answer = _make_answer()
    with patch("app.routers.query.PlainRagPipeline") as cls:
        cls.return_value.execute.return_value = answer
        app = FastAPI()
        app.include_router(router)
        app.state.neo4j_driver = MagicMock()
        app.state.embedding_provider = MagicMock()
        app.state.generation_provider = MagicMock()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/query/plain-rag",
            json={"query": "test"},
            headers={"X-Api-Key": "invalid"},
        )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Integration test: POST /query/plain-rag returns HTTP 200 and valid schema
# ---------------------------------------------------------------------------


def test_valid_request_returns_200() -> None:
    answer = _make_answer()
    with patch("app.routers.query.PlainRagPipeline") as cls:
        cls.return_value.execute.return_value = answer
        app = FastAPI()
        app.include_router(router)
        app.state.neo4j_driver = MagicMock()
        app.state.embedding_provider = MagicMock()
        app.state.generation_provider = MagicMock()
        client = TestClient(app)
        resp = client.post(
            "/query/plain-rag",
            json={"query": "Who supplies steel to Axiom Corp?"},
            headers={"X-Api-Key": _VALID_KEY},
        )
    assert resp.status_code == 200


def test_response_is_valid_answer_schema() -> None:
    answer = _make_answer()
    with patch("app.routers.query.PlainRagPipeline") as cls:
        cls.return_value.execute.return_value = answer
        app = FastAPI()
        app.include_router(router)
        app.state.neo4j_driver = MagicMock()
        app.state.embedding_provider = MagicMock()
        app.state.generation_provider = MagicMock()
        client = TestClient(app)
        resp = client.post(
            "/query/plain-rag",
            json={"query": "test"},
            headers={"X-Api-Key": _VALID_KEY},
        )
    body = resp.json()
    # All required AnswerSchema fields must be present
    assert "answer" in body
    assert "graph_evidence" in body
    assert "text_citations" in body
    assert "retrieval_debug" in body
    assert "mode" in body


def test_response_mode_is_plain_rag() -> None:
    answer = _make_answer()
    with patch("app.routers.query.PlainRagPipeline") as cls:
        cls.return_value.execute.return_value = answer
        app = FastAPI()
        app.include_router(router)
        app.state.neo4j_driver = MagicMock()
        app.state.embedding_provider = MagicMock()
        app.state.generation_provider = MagicMock()
        client = TestClient(app)
        resp = client.post(
            "/query/plain-rag",
            json={"query": "test"},
            headers={"X-Api-Key": _VALID_KEY},
        )
    assert resp.json()["mode"] == "plain_rag"


def test_response_graph_evidence_is_empty() -> None:
    answer = _make_answer()
    with patch("app.routers.query.PlainRagPipeline") as cls:
        cls.return_value.execute.return_value = answer
        app = FastAPI()
        app.include_router(router)
        app.state.neo4j_driver = MagicMock()
        app.state.embedding_provider = MagicMock()
        app.state.generation_provider = MagicMock()
        client = TestClient(app)
        resp = client.post(
            "/query/plain-rag",
            json={"query": "test"},
            headers={"X-Api-Key": _VALID_KEY},
        )
    assert resp.json()["graph_evidence"] == []


def test_response_retrieval_debug_graph_query_is_none() -> None:
    answer = _make_answer()
    with patch("app.routers.query.PlainRagPipeline") as cls:
        cls.return_value.execute.return_value = answer
        app = FastAPI()
        app.include_router(router)
        app.state.neo4j_driver = MagicMock()
        app.state.embedding_provider = MagicMock()
        app.state.generation_provider = MagicMock()
        client = TestClient(app)
        resp = client.post(
            "/query/plain-rag",
            json={"query": "test"},
            headers={"X-Api-Key": _VALID_KEY},
        )
    assert resp.json()["retrieval_debug"]["graph_query"] is None


# ---------------------------------------------------------------------------
# Integration test: timings are positive floats in pipeline output
# ---------------------------------------------------------------------------


def test_timings_are_positive_floats() -> None:
    """Run pipeline with real timing code (mocked providers) and verify timings."""
    from app.pipelines.plain_rag import PlainRagPipeline as RealPipeline

    mock_embed = MagicMock()
    mock_embed.embed.return_value = [[0.1, 0.2, 0.3]]

    mock_session = MagicMock()
    mock_session.run.return_value = [
        {"chunk_id": "CL001_c0", "contract_id": "CT001", "text": "Clause text here."}
    ]
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    from graphrag_assistant.schemas import TextCitation

    mock_gen = MagicMock()
    mock_gen.generate.return_value = AnswerSchema(
        answer="Generated answer.",
        graph_evidence=[],
        text_citations=[
            TextCitation(doc_id="CT001", chunk_id="CL001_c0", quote="Clause text here.")
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

    pipeline = RealPipeline(
        embedding_provider=mock_embed,
        generation_provider=mock_gen,
        driver=mock_driver,
        top_k=5,
    )
    result = pipeline.execute("Who supplies steel?")

    timings = result.retrieval_debug.timings
    for key in ("embed_ms", "retrieve_ms", "generate_ms"):
        assert isinstance(timings[key], float), f"{key} must be float"
        assert timings[key] >= 0.0, f"{key} must be >= 0"


def test_timings_sum_is_finite_and_bounded() -> None:
    """Timings sum must be a finite non-negative number."""
    import math
    import time as _time

    from app.pipelines.plain_rag import PlainRagPipeline as RealPipeline

    mock_embed = MagicMock()
    mock_embed.embed.return_value = [[0.0]]

    mock_session = MagicMock()
    mock_session.run.return_value = []
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    mock_gen = MagicMock()
    mock_gen.generate.return_value = AnswerSchema(
        answer="",
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

    pipeline = RealPipeline(
        embedding_provider=mock_embed,
        generation_provider=mock_gen,
        driver=mock_driver,
    )
    wall_start = _time.monotonic()
    result = pipeline.execute("test")
    wall_ms = (_time.monotonic() - wall_start) * 1000.0

    keys = ("embed_ms", "retrieve_ms", "generate_ms")
    total = sum(result.retrieval_debug.timings[k] for k in keys)
    assert math.isfinite(total)
    assert total >= 0.0
    assert total <= wall_ms + 10.0
