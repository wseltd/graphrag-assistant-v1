"""Tests for app.routers.ingest — POST /ingest/contracts (T029.c).

Unit tests (4): mock ingest_contract, no live Neo4j.
Integration tests (4): require a live Neo4j instance; skipped otherwise.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_api_key
from app.routers.ingest import router

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CONTRACT_TEXT = (
    "This agreement is entered into between Acme Corp and BetaCo. "
    "The supplier shall deliver goods within thirty calendar days of order confirmation. "
    "Payment terms are net-60 from invoice date.  All disputes shall be resolved "
    "by binding arbitration under the rules of the International Chamber of Commerce."
)


def _build_app(neo4j_driver=None, embedding_provider=None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.neo4j_driver = neo4j_driver or MagicMock()
    app.state.embedding_provider = embedding_provider or MagicMock()
    # Bypass auth — these tests exercise ingest logic, not authentication.
    app.dependency_overrides[require_api_key] = lambda: "test-key"
    return app


def _fake_ingest_result(contract_id: str, first: bool = True) -> dict:
    if first:
        return {
            "contract_id": contract_id,
            "chunks_merged": 1,
            "nodes_merged": 2,
            "edges_merged": 1,
        }
    return {
        "contract_id": contract_id,
        "chunks_merged": 0,
        "nodes_merged": 0,
        "edges_merged": 0,
    }


# ---------------------------------------------------------------------------
# Unit test 1: single valid file → 200 with results list
# ---------------------------------------------------------------------------


def test_single_file_returns_200_with_results() -> None:
    """Happy path: one .md file returns HTTP 200 and a non-empty results list."""
    app = _build_app()
    client = TestClient(app)

    with patch(
        "app.routers.ingest.ingest_contract",
        return_value=_fake_ingest_result("acme_contract_md"),
    ):
        resp = client.post(
            "/ingest/contracts",
            files=[("files", ("acme_contract.md", _CONTRACT_TEXT.encode(), "text/plain"))],
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert len(body["results"]) == 1


# ---------------------------------------------------------------------------
# Unit test 2: single valid file → all required fields present
# ---------------------------------------------------------------------------


def test_single_file_response_has_all_result_fields() -> None:
    """Response for a single file must include all four per-file fields."""
    app = _build_app()
    client = TestClient(app)

    with patch(
        "app.routers.ingest.ingest_contract",
        return_value=_fake_ingest_result("acme_contract_md"),
    ):
        resp = client.post(
            "/ingest/contracts",
            files=[("files", ("acme_contract.md", _CONTRACT_TEXT.encode(), "text/plain"))],
        )

    result = resp.json()["results"][0]
    assert "contract_id" in result
    assert "chunks_merged" in result
    assert "nodes_merged" in result
    assert "edges_merged" in result
    assert result["contract_id"] == "acme_contract_md"
    assert result["chunks_merged"] == 1
    assert result["nodes_merged"] == 2
    assert result["edges_merged"] == 1


# ---------------------------------------------------------------------------
# Unit test 3: zero-byte file → 422
# ---------------------------------------------------------------------------


def test_empty_file_returns_422() -> None:
    """Uploading a zero-byte file must return HTTP 422."""
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post(
        "/ingest/contracts",
        files=[("files", ("empty.md", b"", "text/plain"))],
    )

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Unit test 4: no files provided → 422
# ---------------------------------------------------------------------------


def test_no_files_provided_returns_422() -> None:
    """Sending the request with no multipart files must return HTTP 422."""
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/ingest/contracts")

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Integration fixtures
# ---------------------------------------------------------------------------


class _DummyEmbeddingProvider:
    """Returns zero vectors so integration tests do not need sentence-transformers."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]


def _neo4j_uri() -> str:
    return os.getenv("GRAPHRAG_NEO4J_URI", "bolt://localhost:7687")


def _neo4j_auth() -> tuple[str, str]:
    return (
        os.getenv("GRAPHRAG_NEO4J_USER", "neo4j"),
        os.getenv("GRAPHRAG_NEO4J_PASSWORD", "test"),
    )


@pytest.fixture(scope="module")
def neo4j_driver():
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(_neo4j_uri(), auth=_neo4j_auth())
    try:
        driver.verify_connectivity()
    except Exception as exc:
        pytest.skip(f"Neo4j not reachable: {exc}")
    yield driver
    driver.close()


@pytest.fixture(autouse=False)
def clean_ingest_graph(neo4j_driver):
    """Delete all Chunk and Contract nodes created by integration tests."""
    _purge_test_data(neo4j_driver)
    yield
    _purge_test_data(neo4j_driver)


def _purge_test_data(driver) -> None:
    with driver.session() as s:
        s.run(
            "MATCH (n) WHERE n:Contract OR n:Chunk DETACH DELETE n"
        )


def _count_nodes(driver, label: str, contract_id: str) -> int:
    with driver.session() as s:
        rows = list(
            s.run(
                f"MATCH (n:{label} {{contract_id: $cid}}) RETURN count(n) AS cnt",
                {"cid": contract_id},
            )
        )
        return rows[0]["cnt"]


def _count_contract_nodes(driver, contract_id: str) -> int:
    with driver.session() as s:
        rows = list(
            s.run(
                "MATCH (n:Contract {contract_id: $cid}) RETURN count(n) AS cnt",
                {"cid": contract_id},
            )
        )
        return rows[0]["cnt"]


def _integration_client(driver) -> TestClient:
    app = _build_app(
        neo4j_driver=driver,
        embedding_provider=_DummyEmbeddingProvider(),
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# Integration test 5: first ingest creates nodes in Neo4j
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_first_ingest_creates_nodes(neo4j_driver, clean_ingest_graph) -> None:
    """First upload of a contract creates Contract and Chunk nodes in Neo4j."""
    client = _integration_client(neo4j_driver)

    resp = client.post(
        "/ingest/contracts",
        files=[("files", ("int_contract_a.md", _CONTRACT_TEXT.encode(), "text/plain"))],
    )

    assert resp.status_code == 200
    result = resp.json()["results"][0]
    assert result["contract_id"] == "int_contract_a_md"
    assert result["nodes_merged"] > 0, "First ingest must create at least one node"
    assert result["edges_merged"] > 0, "First ingest must create FROM_CONTRACT edges"
    assert _count_contract_nodes(neo4j_driver, "int_contract_a_md") == 1


# ---------------------------------------------------------------------------
# Integration test 6: second identical ingest returns zero counts
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_second_ingest_is_noop(neo4j_driver, clean_ingest_graph) -> None:
    """Second upload of the same file returns zeros for all counts."""
    client = _integration_client(neo4j_driver)
    file_arg = [("files", ("int_contract_b.md", _CONTRACT_TEXT.encode(), "text/plain"))]

    first = client.post("/ingest/contracts", files=file_arg)
    assert first.status_code == 200

    second = client.post("/ingest/contracts", files=file_arg)
    assert second.status_code == 200

    r = second.json()["results"][0]
    assert r["chunks_merged"] == 0, "Second ingest must report 0 new chunks"
    assert r["nodes_merged"] == 0, "Second ingest must report 0 new nodes"
    assert r["edges_merged"] == 0, "Second ingest must report 0 new edges"


# ---------------------------------------------------------------------------
# Integration test 7: third ingest after manual delete recreates nodes
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_third_ingest_after_delete_recreates(neo4j_driver, clean_ingest_graph) -> None:
    """After manually deleting Contract and Chunk nodes, a fresh ingest recreates them."""
    client = _integration_client(neo4j_driver)
    file_arg = [("files", ("int_contract_c.md", _CONTRACT_TEXT.encode(), "text/plain"))]
    cid = "int_contract_c_md"

    # First ingest — creates nodes
    r1 = client.post("/ingest/contracts", files=file_arg)
    assert r1.status_code == 200
    assert r1.json()["results"][0]["nodes_merged"] > 0

    # Manual delete — simulate data loss
    with neo4j_driver.session() as s:
        s.run(
            "MATCH (n) WHERE (n:Contract AND n.contract_id = $cid) "
            "OR (n:Chunk AND n.contract_id = $cid) DETACH DELETE n",
            {"cid": cid},
        )
    assert _count_contract_nodes(neo4j_driver, cid) == 0

    # Third ingest — must recreate all nodes
    r3 = client.post("/ingest/contracts", files=file_arg)
    assert r3.status_code == 200
    r = r3.json()["results"][0]
    assert r["nodes_merged"] > 0, "Re-ingest after delete must recreate nodes"
    assert r["edges_merged"] > 0, "Re-ingest after delete must recreate edges"
    assert _count_contract_nodes(neo4j_driver, cid) == 1


# ---------------------------------------------------------------------------
# Integration test 8: five distinct files → five distinct contract_ids
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_five_distinct_files_create_five_contracts(neo4j_driver, clean_ingest_graph) -> None:
    """Uploading five different files in one request creates five distinct contracts."""
    client = _integration_client(neo4j_driver)

    files = [
        (
            "files",
            (
                f"int_multi_{i}.md",
                f"Contract {i}: {_CONTRACT_TEXT}".encode(),
                "text/plain",
            ),
        )
        for i in range(1, 6)
    ]

    resp = client.post("/ingest/contracts", files=files)

    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 5

    seen_ids = {r["contract_id"] for r in results}
    assert len(seen_ids) == 5, f"Expected 5 distinct contract_ids, got {seen_ids!r}"

    # All contracts must appear as Contract nodes in Neo4j
    for r in results:
        assert _count_contract_nodes(neo4j_driver, r["contract_id"]) == 1, (
            f"Contract node missing for {r['contract_id']!r}"
        )
