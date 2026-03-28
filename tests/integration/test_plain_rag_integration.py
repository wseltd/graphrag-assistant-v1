"""Integration tests for POST /query/plain-rag (T041).

Covers three cases without mocking Neo4j or the embedding provider:

    test_known_answer    — entity in graph, chunks exist → non-empty citations
    test_unknown_entity  — empty graph → zero results, graceful degradation
    test_empty_corpus    — entity in graph, no Chunk nodes → valid schema, empty citations

Prerequisites: Neo4j reachable at GRAPHRAG_NEO4J_URI (default bolt://localhost:7687).
Skipped automatically when the URI is unreachable.

Design notes
------------
* direct_driver connects to the same Neo4j instance as the app lifespan,
  enabling graph state manipulation (full reset, chunk teardown) between tests.
* The vector index is created once in the direct_driver fixture so queryNodes
  never raises "index not found" on a fresh DB.
* test_unknown_entity runs on an empty graph (no seed): the index exists but has
  no entries, so queryNodes returns zero rows → text_citations=[].
* test_empty_corpus seeds the full dataset, then deletes all Chunk nodes;
  the index auto-updates, so queryNodes again returns zero rows.
* No unittest.mock is used; Neo4j and the embedding provider are real.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient
from neo4j import GraphDatabase

from app.main import create_app
from graphrag_assistant.schemas import AnswerSchema

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_URI = os.getenv("GRAPHRAG_NEO4J_URI", "bolt://localhost:7687")
_USER = os.getenv("GRAPHRAG_NEO4J_USER", "neo4j")
_PASSWORD = os.getenv("GRAPHRAG_NEO4J_PASSWORD", "test")
_API_KEY = os.getenv("API_KEY", "dev-key-change-in-prod")
_HEADERS = {"X-Api-Key": _API_KEY}

# Vector index DDL — dimensions match the local embedding model (384-dim).
# IF NOT EXISTS makes this idempotent across test runs.
_CREATE_VECTOR_IDX = (
    "CREATE VECTOR INDEX chunk_embedding_idx IF NOT EXISTS "
    "FOR (c:Chunk) ON (c.embedding) "
    "OPTIONS {indexConfig: {"
    "`vector.dimensions`: 384, "
    "`vector.similarityFunction`: 'cosine'"
    "}}"
)

_QUERY_KNOWN = "Which contracts involve Meridian Holdings?"
_QUERY_UNKNOWN = "List contracts for Zxqwerty Fictitious Corp Ltd?"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def direct_driver():
    """Auxiliary Neo4j driver for direct graph manipulation.

    Creates the vector index so queryNodes never raises 'index not found'.
    Skips the whole module when Neo4j is unreachable.
    """
    try:
        drv = GraphDatabase.driver(_URI, auth=(_USER, _PASSWORD))
        drv.verify_connectivity()
    except Exception as exc:
        pytest.skip(f"Neo4j not reachable at {_URI}: {exc}")
    with drv.session() as session:
        session.run(_CREATE_VECTOR_IDX)
    yield drv
    drv.close()


@pytest.fixture(scope="module")
def live_client(direct_driver):
    """Full-stack FastAPI TestClient; lifespan initialises all providers."""
    app = create_app()
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_known_answer(live_client, direct_driver) -> None:
    """Known entity + full corpus → HTTP 200, non-empty answer, citations, mode."""
    live_client.post("/seed?reset=true")
    resp = live_client.post(
        "/api/v1/query/plain-rag",
        json={"query": _QUERY_KNOWN},
        headers=_HEADERS,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["text_citations"]
    assert body["mode"] == "plain_rag"


def test_unknown_entity(live_client, direct_driver) -> None:
    """Empty graph → zero vector results, no traceback, valid schema, graceful answer."""
    with direct_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    resp = live_client.post(
        "/api/v1/query/plain-rag",
        json={"query": _QUERY_UNKNOWN},
        headers=_HEADERS,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["graph_evidence"] == []
    assert body["text_citations"] == []
    assert "traceback" not in body
    assert "exception" not in body
    assert body["mode"] == "plain_rag"
    assert body["retrieval_debug"]["entity_matches"] == []


def test_empty_corpus(live_client, direct_driver) -> None:
    """Entity in graph, no Chunk nodes → valid schema, empty citations, timings populated."""
    live_client.post("/seed?reset=true")
    with direct_driver.session() as session:
        session.run("MATCH (c:Chunk) DETACH DELETE c")
    resp = live_client.post(
        "/api/v1/query/plain-rag",
        json={"query": _QUERY_KNOWN},
        headers=_HEADERS,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["text_citations"] == []
    assert body["retrieval_debug"]["chunk_ids"] == []
    assert body["retrieval_debug"]["timings"]
    assert body["mode"] == "plain_rag"
    AnswerSchema.model_validate(body)
