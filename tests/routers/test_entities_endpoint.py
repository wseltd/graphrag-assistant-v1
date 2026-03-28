"""Tests for app.routers.entities — GET /entities/{entity_type}/{entity_id} (T031).

Unit tests (4): whitelist rejection and missing-entity 404 — no live Neo4j.
Integration tests (10): edge direction, zero-edge node, happy path — require Neo4j.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_api_key
from app.routers.entities import router

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _build_app(neo4j_driver=None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.neo4j_driver = neo4j_driver or MagicMock()
    # Bypass auth — these tests exercise entity logic, not authentication.
    app.dependency_overrides[require_api_key] = lambda: "test-key"
    return app


# ---------------------------------------------------------------------------
# Unit test 1 — unknown entity_type returns 404 (whitelist rejection)
# ---------------------------------------------------------------------------


def test_unknown_type_returns_404() -> None:
    """GET /entities/Foobar/x returns 404 before touching Neo4j."""
    mock_driver = MagicMock()
    app = _build_app(mock_driver)
    client = TestClient(app)

    resp = client.get("/entities/Foobar/some-id")

    assert resp.status_code == 404
    mock_driver.session.assert_not_called()


# ---------------------------------------------------------------------------
# Unit test 2 — near-miss label spelling still rejected
# ---------------------------------------------------------------------------


def test_near_miss_label_returns_404() -> None:
    """GET /entities/Chunks/x (plural) returns 404 — 'Chunk' is in whitelist, 'Chunks' is not."""
    mock_driver = MagicMock()
    app = _build_app(mock_driver)
    client = TestClient(app)

    resp = client.get("/entities/Chunks/some-id")

    assert resp.status_code == 404
    mock_driver.session.assert_not_called()


# ---------------------------------------------------------------------------
# Unit test 3 — nonexistent entity_id returns 404
# ---------------------------------------------------------------------------


def test_nonexistent_entity_id_returns_404() -> None:
    """When Neo4j returns no rows the endpoint returns HTTP 404."""
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = iter([])

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    app = _build_app(mock_driver)
    client = TestClient(app)

    resp = client.get("/entities/Company/nonexistent-id")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Unit test 4 — 404 detail message names the entity
# ---------------------------------------------------------------------------


def test_nonexistent_entity_id_detail_contains_entity_type() -> None:
    """404 detail must mention the entity type so the caller knows what was missing."""
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = iter([])

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    app = _build_app(mock_driver)
    client = TestClient(app)

    resp = client.get("/entities/Person/ghost-id")

    assert resp.status_code == 404
    assert "Person" in resp.json()["detail"]


# ===========================================================================
# Integration fixtures
# ===========================================================================


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
def clean_entity_graph(neo4j_driver):
    """Remove test nodes created by this test module before and after each test."""
    _purge(neo4j_driver)
    yield
    _purge(neo4j_driver)


def _purge(driver) -> None:
    with driver.session() as s:
        s.run(
            "MATCH (n) WHERE n.id STARTS WITH 'test_entity_' DETACH DELETE n"
        )


def _seed_nodes(driver) -> None:
    """Create a small subgraph used by integration tests.

    Graph:
        (c:Company {id:'test_entity_co1'})
            -[:REGISTERED_AT]-> (a:Address {id:'test_entity_addr1'})
        (p:Person  {id:'test_entity_p1'})
            -[:DIRECTOR_OF]->   (c)
        (iso:Company {id:'test_entity_iso'})   -- isolated node, no edges
    """
    with driver.session() as s:
        s.run(
            """
            MERGE (c:Company {id: 'test_entity_co1'})
              SET c.name = 'TestCo'
            MERGE (a:Address {id: 'test_entity_addr1'})
              SET a.street = '1 Test St'
            MERGE (p:Person {id: 'test_entity_p1'})
              SET p.name = 'Alice'
            MERGE (iso:Company {id: 'test_entity_iso'})
              SET iso.name = 'IsolatedCo'
            MERGE (c)-[:REGISTERED_AT]->(a)
            MERGE (p)-[:DIRECTOR_OF]->(c)
            """
        )


def _integration_client(driver) -> TestClient:
    app = _build_app(driver)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Integration test 5 — outgoing edges are present in the response
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_outgoing_edges_present(neo4j_driver, clean_entity_graph) -> None:
    """Company node has a REGISTERED_AT out-edge; it must appear in the response."""
    _seed_nodes(neo4j_driver)
    client = _integration_client(neo4j_driver)

    resp = client.get("/entities/Company/test_entity_co1")

    assert resp.status_code == 200
    edges = resp.json()["edges"]
    out_edges = [e for e in edges if e["direction"] == "out"]
    assert len(out_edges) >= 1, f"Expected at least 1 out-edge, got edges={edges}"


# ---------------------------------------------------------------------------
# Integration test 6 — outgoing edge rel_type is correct
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_outgoing_edge_rel_type_correct(neo4j_driver, clean_entity_graph) -> None:
    """The outgoing REGISTERED_AT edge must have rel_type='REGISTERED_AT'."""
    _seed_nodes(neo4j_driver)
    client = _integration_client(neo4j_driver)

    resp = client.get("/entities/Company/test_entity_co1")

    edges = resp.json()["edges"]
    out_edge = next((e for e in edges if e["direction"] == "out"), None)
    assert out_edge is not None
    assert out_edge["rel_type"] == "REGISTERED_AT"


# ---------------------------------------------------------------------------
# Integration test 7 — outgoing edge neighbour_id is correct
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_outgoing_edge_neighbour_id_correct(neo4j_driver, clean_entity_graph) -> None:
    """The outgoing REGISTERED_AT edge must point to the Address node id."""
    _seed_nodes(neo4j_driver)
    client = _integration_client(neo4j_driver)

    resp = client.get("/entities/Company/test_entity_co1")

    edges = resp.json()["edges"]
    out_edge = next(
        (e for e in edges if e["direction"] == "out" and e["rel_type"] == "REGISTERED_AT"),
        None,
    )
    assert out_edge is not None
    assert out_edge["neighbour_id"] == "test_entity_addr1"


# ---------------------------------------------------------------------------
# Integration test 8 — incoming edges are present in the response
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_incoming_edges_present(neo4j_driver, clean_entity_graph) -> None:
    """Company node has a DIRECTOR_OF in-edge from Person; it must appear in response."""
    _seed_nodes(neo4j_driver)
    client = _integration_client(neo4j_driver)

    resp = client.get("/entities/Company/test_entity_co1")

    assert resp.status_code == 200
    edges = resp.json()["edges"]
    in_edges = [e for e in edges if e["direction"] == "in"]
    assert len(in_edges) >= 1, f"Expected at least 1 in-edge, got edges={edges}"


# ---------------------------------------------------------------------------
# Integration test 9 — incoming edge rel_type is correct
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_incoming_edge_rel_type_correct(neo4j_driver, clean_entity_graph) -> None:
    """The incoming edge on the Company node must have rel_type='DIRECTOR_OF'."""
    _seed_nodes(neo4j_driver)
    client = _integration_client(neo4j_driver)

    resp = client.get("/entities/Company/test_entity_co1")

    edges = resp.json()["edges"]
    in_edge = next((e for e in edges if e["direction"] == "in"), None)
    assert in_edge is not None
    assert in_edge["rel_type"] == "DIRECTOR_OF"


# ---------------------------------------------------------------------------
# Integration test 10 — incoming edge neighbour_id is correct
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_incoming_edge_neighbour_id_correct(neo4j_driver, clean_entity_graph) -> None:
    """The DIRECTOR_OF in-edge must identify the Person as the source neighbour."""
    _seed_nodes(neo4j_driver)
    client = _integration_client(neo4j_driver)

    resp = client.get("/entities/Company/test_entity_co1")

    edges = resp.json()["edges"]
    in_edge = next(
        (e for e in edges if e["direction"] == "in" and e["rel_type"] == "DIRECTOR_OF"),
        None,
    )
    assert in_edge is not None
    assert in_edge["neighbour_id"] == "test_entity_p1"


# ---------------------------------------------------------------------------
# Integration test 11 — isolated node returns HTTP 200
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_node_with_no_edges_returns_200(neo4j_driver, clean_entity_graph) -> None:
    """A node that exists but has zero edges must return HTTP 200, not 404."""
    _seed_nodes(neo4j_driver)
    client = _integration_client(neo4j_driver)

    resp = client.get("/entities/Company/test_entity_iso")

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Integration test 12 — isolated node has an empty edges list
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_node_with_no_edges_has_empty_edges_list(neo4j_driver, clean_entity_graph) -> None:
    """A node with no relationships must return an empty edges list, not an error."""
    _seed_nodes(neo4j_driver)
    client = _integration_client(neo4j_driver)

    resp = client.get("/entities/Company/test_entity_iso")

    assert resp.json()["edges"] == []


# ---------------------------------------------------------------------------
# Integration test 13 — happy path: all top-level fields present
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_happy_path_returns_all_top_level_fields(neo4j_driver, clean_entity_graph) -> None:
    """Response for a valid entity must include id, label, properties, and edges keys."""
    _seed_nodes(neo4j_driver)
    client = _integration_client(neo4j_driver)

    resp = client.get("/entities/Company/test_entity_co1")

    assert resp.status_code == 200
    body = resp.json()
    assert "id" in body
    assert "label" in body
    assert "properties" in body
    assert "edges" in body


# ---------------------------------------------------------------------------
# Integration test 14 — happy path: label in response matches entity_type param
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_happy_path_label_matches_entity_type(neo4j_driver, clean_entity_graph) -> None:
    """The label field in the response must equal the entity_type path segment."""
    _seed_nodes(neo4j_driver)
    client = _integration_client(neo4j_driver)

    resp = client.get("/entities/Company/test_entity_co1")

    assert resp.json()["label"] == "Company"
