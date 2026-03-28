"""Integration tests for graphrag_assistant.loaders.entity_loader against a live Neo4j instance.

Run with:  pytest -m integration

Requires a reachable Neo4j instance.  Connection is configured via env vars:
  GRAPHRAG_NEO4J_URI      (default: bolt://localhost:7687)
  GRAPHRAG_NEO4J_USER     (default: neo4j)
  GRAPHRAG_NEO4J_PASSWORD (default: test)

Tests delete all Company/Person/Address/Product nodes before and after each
test to avoid polluting any existing graph state.
"""
from __future__ import annotations

import os

import pytest
from neo4j import GraphDatabase

from graphrag_assistant.loaders.entity_loader import load_entities

# Expected row counts matching data/raw/ CSVs
_EXPECTED_COUNTS = {
    "Company": 15,
    "Person": 15,
    "Address": 15,
    "Product": 15,
}


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
def clean_entity_nodes(neo4j_driver):
    _delete_entity_nodes(neo4j_driver)
    yield
    _delete_entity_nodes(neo4j_driver)


def _delete_entity_nodes(driver) -> None:
    with driver.session() as s:
        s.run("MATCH (n) WHERE n:Company OR n:Person OR n:Address OR n:Product DETACH DELETE n")


_COUNT_QUERIES: dict[str, str] = {
    "Company": "MATCH (n:Company) RETURN count(n) AS cnt",
    "Person": "MATCH (n:Person) RETURN count(n) AS cnt",
    "Address": "MATCH (n:Address) RETURN count(n) AS cnt",
    "Product": "MATCH (n:Product) RETURN count(n) AS cnt",
}


def _count_nodes(driver, label: str) -> int:
    with driver.session() as s:
        result = list(s.run(_COUNT_QUERIES[label]))
        return result[0]["cnt"]


# ---------------------------------------------------------------------------
# Integration test 1: node counts per label match CSV row counts
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_node_counts_match_csv_rows(neo4j_driver) -> None:
    result = load_entities(neo4j_driver)

    assert result.nodes_created == sum(_EXPECTED_COUNTS.values()), (
        f"Expected {sum(_EXPECTED_COUNTS.values())} nodes created, got {result.nodes_created}"
    )
    assert result.nodes_merged == 0
    assert result.constraints_ensured == 4

    for label, expected in _EXPECTED_COUNTS.items():
        actual = _count_nodes(neo4j_driver, label)
        assert actual == expected, (
            f"Expected {expected} {label} nodes, found {actual}"
        )


# ---------------------------------------------------------------------------
# Integration test 2: second call produces zero new nodes
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_second_load_produces_zero_new_nodes(neo4j_driver) -> None:
    load_entities(neo4j_driver)
    second = load_entities(neo4j_driver)

    assert second.nodes_created == 0, (
        f"Second call must be idempotent (0 created), got {second.nodes_created}"
    )
    assert second.nodes_merged == sum(_EXPECTED_COUNTS.values())
    assert second.constraints_ensured == 4

    for label, expected in _EXPECTED_COUNTS.items():
        actual = _count_nodes(neo4j_driver, label)
        assert actual == expected, (
            f"Idempotency violated for {label}: expected {expected} records, found {actual}"
        )
