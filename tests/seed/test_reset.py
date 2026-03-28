"""Tests for graphrag_assistant.seed.reset (T019.a).

Unit tests (8)
--------------
 1. test_empty_graph_exits_after_one_iteration
 2. test_single_batch_loops_until_empty
 3. test_multiple_batches_loops_until_empty
 4. test_cypher_contains_limit_clause
 5. test_cypher_uses_in_transactions
 6. test_cypher_uses_call_subquery_form
 7. test_driver_error_propagates
 8. test_returns_none

Integration tests (2) — require a reachable Neo4j instance (pytest -m integration)
------------------------------------------------------------------------------------
 9. test_integration_reset_empty_graph_is_noop
10. test_integration_reset_populated_graph_leaves_zero_nodes
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from neo4j import GraphDatabase

from graphrag_assistant.seed.reset import _BATCH_DELETE_CYPHER, reset_graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_driver(deleted_sequence: list[int]) -> MagicMock:
    """Return a mock driver whose session().run().consume().counters.nodes_deleted
    cycles through *deleted_sequence* across successive loop iterations."""
    summaries = []
    for count in deleted_sequence:
        s = MagicMock()
        s.counters.nodes_deleted = count
        summaries.append(s)

    mock_result = MagicMock()
    mock_result.consume.side_effect = summaries

    mock_session = MagicMock()
    mock_session.run.return_value = mock_result

    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_driver.session.return_value.__exit__.return_value = False

    return mock_driver


# ---------------------------------------------------------------------------
# Unit test 1: empty graph — loop exits after the first no-op iteration
# ---------------------------------------------------------------------------


def test_empty_graph_exits_after_one_iteration() -> None:
    driver = _make_driver([0])
    reset_graph(driver)
    # session opened exactly once
    assert driver.session.call_count == 1


# ---------------------------------------------------------------------------
# Unit test 2: single batch — one real delete then one empty check
# ---------------------------------------------------------------------------


def test_single_batch_loops_until_empty() -> None:
    driver = _make_driver([500, 0])
    reset_graph(driver)
    # session opened twice: once to delete 500, once to confirm empty
    assert driver.session.call_count == 2


# ---------------------------------------------------------------------------
# Unit test 3: multiple batches — loop repeats until graph is drained
# ---------------------------------------------------------------------------


def test_multiple_batches_loops_until_empty() -> None:
    driver = _make_driver([1000, 1000, 300, 0])
    reset_graph(driver)
    assert driver.session.call_count == 4


# ---------------------------------------------------------------------------
# Unit test 4: Cypher string contains a LIMIT clause (batched, not unbounded)
# ---------------------------------------------------------------------------


def test_cypher_contains_limit_clause() -> None:
    assert "LIMIT" in _BATCH_DELETE_CYPHER, (
        "Reset Cypher must use LIMIT to bound each batch"
    )


# ---------------------------------------------------------------------------
# Unit test 5: Cypher string uses IN TRANSACTIONS (sub-transaction batching)
# ---------------------------------------------------------------------------


def test_cypher_uses_in_transactions() -> None:
    assert "IN TRANSACTIONS" in _BATCH_DELETE_CYPHER, (
        "Reset Cypher must use IN TRANSACTIONS for batched sub-transactions"
    )


# ---------------------------------------------------------------------------
# Unit test 6: Cypher uses the CALL { ... } subquery form, not bare DELETE
# ---------------------------------------------------------------------------


def test_cypher_uses_call_subquery_form() -> None:
    upper = _BATCH_DELETE_CYPHER.upper()
    assert upper.startswith("CALL {") or upper.startswith("CALL{"), (
        "Reset Cypher must open with a CALL { } subquery"
    )


# ---------------------------------------------------------------------------
# Unit test 7: driver error (e.g. session.run raises) propagates to caller
# ---------------------------------------------------------------------------


def test_driver_error_propagates() -> None:
    mock_session = MagicMock()
    mock_session.run.side_effect = RuntimeError("connection refused")

    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_driver.session.return_value.__exit__.return_value = False

    with pytest.raises(RuntimeError, match="connection refused"):
        reset_graph(mock_driver)


# ---------------------------------------------------------------------------
# Unit test 8: reset_graph returns None (not a counter, not a summary)
# ---------------------------------------------------------------------------


def test_returns_none() -> None:
    driver = _make_driver([0])
    result = reset_graph(driver)
    assert result is None


# ---------------------------------------------------------------------------
# Integration test helpers
# ---------------------------------------------------------------------------


def _uri() -> str:
    return os.getenv("GRAPHRAG_NEO4J_URI", "bolt://localhost:7687")


def _auth() -> tuple[str, str]:
    return (
        os.getenv("GRAPHRAG_NEO4J_USER", "neo4j"),
        os.getenv("GRAPHRAG_NEO4J_PASSWORD", "test"),
    )


def _node_count(driver) -> int:
    with driver.session() as s:
        rows = list(s.run("MATCH (n) RETURN count(n) AS cnt"))
        return rows[0]["cnt"]


@pytest.fixture(scope="module")
def neo4j_driver():
    driver = GraphDatabase.driver(_uri(), auth=_auth())
    try:
        driver.verify_connectivity()
    except Exception as exc:
        pytest.skip(f"Neo4j not reachable: {exc}")
    yield driver
    driver.close()


@pytest.fixture(autouse=False)
def clean_graph(neo4j_driver):
    with neo4j_driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    yield
    with neo4j_driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")


# ---------------------------------------------------------------------------
# Integration test 9: reset on an already-empty graph is a no-op
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_reset_empty_graph_is_noop(neo4j_driver, clean_graph) -> None:
    assert _node_count(neo4j_driver) == 0
    reset_graph(neo4j_driver)  # must not raise
    assert _node_count(neo4j_driver) == 0


# ---------------------------------------------------------------------------
# Integration test 10: reset deletes all nodes and leaves the graph empty
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_reset_populated_graph_leaves_zero_nodes(
    neo4j_driver, clean_graph
) -> None:
    # Seed a small number of nodes with a relationship
    with neo4j_driver.session() as s:
        s.run(
            "CREATE (a:Company {id: 'C001', name: 'Alpha'})"
            "-[:SUPPLIES]->(b:Company {id: 'C002', name: 'Beta'})"
        )

    assert _node_count(neo4j_driver) == 2

    reset_graph(neo4j_driver)

    assert _node_count(neo4j_driver) == 0, (
        "reset_graph must leave the graph with zero nodes"
    )
