"""Integration tests for Neo4jVectorProvider against a live Neo4j instance.

Run with:  pytest -m integration

Requires a reachable Neo4j instance.  Connection is configured via env vars:
  GRAPHRAG_NEO4J_URI      (default: bolt://localhost:7687)
  GRAPHRAG_NEO4J_USER     (default: neo4j)
  GRAPHRAG_NEO4J_PASSWORD (default: test)

Tests use the label :_TestChunk and a dedicated index to avoid touching
production data.  Both are removed before and after each test.
"""
from __future__ import annotations

import os
import time

import pytest
from neo4j import GraphDatabase

from graphrag_assistant.providers.neo4j_vector import (
    Neo4jVectorProvider,
    ProviderError,
)

_INDEX = "chunk_embedding_idx_test"
_DIMS = 4  # small dimension keeps test embeddings trivial

# Neo4j DDL does not support parameterised index names; build static strings once.
_DDL_DROP = "DROP INDEX " + _INDEX + " IF EXISTS"
_DDL_CREATE = (
    "CREATE VECTOR INDEX " + _INDEX + " IF NOT EXISTS "
    "FOR (c:_TestChunk) ON (c.embedding) "
    "OPTIONS {indexConfig: {`vector.dimensions`: $dims, "
    "`vector.similarityFunction`: 'cosine'}}"
)


def _uri() -> str:
    return os.getenv("GRAPHRAG_NEO4J_URI", "bolt://localhost:7687")


def _auth() -> tuple[str, str]:
    return (
        os.getenv("GRAPHRAG_NEO4J_USER", "neo4j"),
        os.getenv("GRAPHRAG_NEO4J_PASSWORD", "test"),
    )


@pytest.fixture(scope="module")
def neo4j_driver():
    driver = GraphDatabase.driver(_uri(), auth=_auth())
    try:
        driver.verify_connectivity()
    except Exception as exc:
        pytest.skip(f"Neo4j not reachable: {exc}")
    yield driver
    driver.close()


@pytest.fixture(autouse=True)
def clean_state(neo4j_driver):
    _cleanup(neo4j_driver)
    yield
    _cleanup(neo4j_driver)


def _cleanup(driver) -> None:
    with driver.session() as s:
        s.run("MATCH (c:_TestChunk) DETACH DELETE c")
        s.run(_DDL_DROP)


def _seed_and_index(driver, embeddings: list[list[float]]) -> None:
    with driver.session() as s:
        for i, emb in enumerate(embeddings):
            s.run(
                "CREATE (:_TestChunk {chunk_id: $cid, doc_id: $did, text: $txt, embedding: $emb})",
                {"cid": f"chunk_{i}", "did": f"doc_{i}", "txt": f"text {i}", "emb": emb},
            )
        s.run(_DDL_CREATE, {"dims": _DIMS})
    _await_index(driver)


def _await_index(driver, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with driver.session() as s:
            rows = list(s.run("SHOW INDEXES WHERE name = $name", {"name": _INDEX}))
            if rows and rows[0]["state"] == "ONLINE":
                return
        time.sleep(0.5)
    pytest.fail(f"Index {_INDEX!r} did not come ONLINE within {timeout}s")


# ---------------------------------------------------------------------------
# Integration test 1: seed 5 chunks, query k=3, assert descending score order
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_query_returns_k_results_in_score_order(neo4j_driver) -> None:
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    _seed_and_index(neo4j_driver, embeddings)

    provider = Neo4jVectorProvider(neo4j_driver, _INDEX)
    results = provider.query([1.0, 0.0, 0.0, 0.0], k=3)

    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), (
        "Results must be in strictly descending score order"
    )


# ---------------------------------------------------------------------------
# Integration test 2: query before index exists → ProviderError
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_query_before_index_raises_provider_error(neo4j_driver) -> None:
    # clean_state already removed the index; no seed here
    provider = Neo4jVectorProvider(neo4j_driver, _INDEX)

    with pytest.raises(ProviderError) as exc_info:
        provider.query([1.0, 0.0, 0.0, 0.0], k=3)

    assert _INDEX in str(exc_info.value)
