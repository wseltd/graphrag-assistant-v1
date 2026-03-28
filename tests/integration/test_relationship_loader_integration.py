"""Integration tests for graphrag_assistant.loaders.relationship_loader (T016).

Run with:  pytest -m integration

Requires a reachable Neo4j instance.  Connection is configured via env vars:
  GRAPHRAG_NEO4J_URI      (default: bolt://localhost:7687)
  GRAPHRAG_NEO4J_USER     (default: neo4j)
  GRAPHRAG_NEO4J_PASSWORD (default: test)

Three tests:
  1. test_integration_edge_counts_after_full_seed
       Load entities then relationships; assert exact edge counts per type.
  2. test_integration_second_load_zero_new_edges
       Second call to load_relationships produces zero new edges.
  3. test_integration_data_integrity_error_no_edges_written
       Deleting a required node causes DataIntegrityError; zero DIRECTOR_OF
       edges are written.
"""
from __future__ import annotations

import os

import pytest
from neo4j import GraphDatabase

from graphrag_assistant.loaders.entity_loader import load_entities
from graphrag_assistant.loaders.relationship_loader import (
    DataIntegrityError,
    load_relationships,
)

# Expected counts derived from data/raw/ CSV files (T016 seed).
_EXPECTED_DIRECTOR_OF = 20
_EXPECTED_REGISTERED_AT = 15
_EXPECTED_SUPPLIES = 15
_EXPECTED_TOTAL = _EXPECTED_DIRECTOR_OF + _EXPECTED_REGISTERED_AT + _EXPECTED_SUPPLIES


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
def clean_graph(neo4j_driver):
    _delete_all(neo4j_driver)
    yield
    _delete_all(neo4j_driver)


def _delete_all(driver) -> None:
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")


def _count_rel(driver, rel_type: str) -> int:
    with driver.session() as s:
        rows = list(s.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS cnt"))
        return rows[0]["cnt"]


# ---------------------------------------------------------------------------
# Integration test 1: correct edge counts after a full seed
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_edge_counts_after_full_seed(neo4j_driver) -> None:
    load_entities(neo4j_driver)
    result = load_relationships(neo4j_driver)

    assert result.edges_created == _EXPECTED_TOTAL, (
        f"Expected {_EXPECTED_TOTAL} edges created on first load, got {result.edges_created}"
    )
    assert result.edges_merged == 0

    assert _count_rel(neo4j_driver, "DIRECTOR_OF") == _EXPECTED_DIRECTOR_OF
    assert _count_rel(neo4j_driver, "REGISTERED_AT") == _EXPECTED_REGISTERED_AT
    assert _count_rel(neo4j_driver, "SUPPLIES") == _EXPECTED_SUPPLIES


# ---------------------------------------------------------------------------
# Integration test 2: second load produces zero new edges
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_second_load_zero_new_edges(neo4j_driver) -> None:
    load_entities(neo4j_driver)
    load_relationships(neo4j_driver)
    second = load_relationships(neo4j_driver)

    assert second.edges_created == 0, (
        f"Second call must be idempotent (0 created), got {second.edges_created}"
    )
    assert second.edges_merged == _EXPECTED_TOTAL

    # Graph counts unchanged
    assert _count_rel(neo4j_driver, "DIRECTOR_OF") == _EXPECTED_DIRECTOR_OF
    assert _count_rel(neo4j_driver, "REGISTERED_AT") == _EXPECTED_REGISTERED_AT
    assert _count_rel(neo4j_driver, "SUPPLIES") == _EXPECTED_SUPPLIES


# ---------------------------------------------------------------------------
# Integration test 3: DataIntegrityError and no edges written
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_data_integrity_error_no_edges_written(neo4j_driver) -> None:
    """Deleting a Person node before loading relationships causes DataIntegrityError.
    Zero DIRECTOR_OF edges must be written (all-or-nothing per relationship type).
    """
    load_entities(neo4j_driver)

    # Remove one Person to simulate a missing endpoint
    with neo4j_driver.session() as s:
        s.run("MATCH (p:Person {id: 'P001'}) DETACH DELETE p")

    with pytest.raises(DataIntegrityError) as exc_info:
        load_relationships(neo4j_driver)

    assert "P001" in exc_info.value.missing_ids, (
        f"Expected 'P001' in missing_ids, got {exc_info.value.missing_ids!r}"
    )

    # No DIRECTOR_OF edges should have been written
    assert _count_rel(neo4j_driver, "DIRECTOR_OF") == 0, (
        "DIRECTOR_OF edges must not be written when validation fails"
    )
