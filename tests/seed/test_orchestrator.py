"""Tests for graphrag_assistant.seed.orchestrator (T019.b).

Unit tests (8):
  1. test_seed_no_reset_skips_reset_graph
  2. test_seed_with_reset_calls_reset_graph_first
  3. test_seed_loader_order_is_fixed
  4. test_seed_returns_summed_counts
  5. test_seed_reset_uses_batched_delete_with_limit
  6. test_seed_base_loader_failure_propagates
  7. test_seed_contracts_loader_failure_propagates
  8. test_seed_chunks_loader_failure_propagates

Integration tests (2) — require a reachable Neo4j instance (pytest -m integration):
  9. test_integration_seed_expected_labels_present
 10. test_integration_seed_expected_relationships_present
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from neo4j import GraphDatabase

from graphrag_assistant.seed.orchestrator import seed

# ---------------------------------------------------------------------------
# Module-level patch paths (kept as constants to avoid repetition)
# ---------------------------------------------------------------------------

_P_RESET = "graphrag_assistant.seed.orchestrator.reset_graph"
_P_BASE = "graphrag_assistant.seed.orchestrator.load_base_entities"
_P_CONTRACTS = "graphrag_assistant.seed.orchestrator.load_contracts_and_clauses"
_P_CHUNKS = "graphrag_assistant.seed.orchestrator.load_chunks_and_edges"

_ZERO: dict[str, int] = {"nodes_written": 0, "edges_written": 0}

# ---------------------------------------------------------------------------
# Unit test 1: reset=False — reset_graph must not be called
# ---------------------------------------------------------------------------


def test_seed_no_reset_skips_reset_graph() -> None:
    mock_driver = MagicMock()
    with patch(_P_RESET) as mock_reset, \
         patch(_P_BASE, return_value=_ZERO), \
         patch(_P_CONTRACTS, return_value=_ZERO), \
         patch(_P_CHUNKS, return_value=_ZERO):
        seed(mock_driver, reset=False)

    mock_reset.assert_not_called()
    assert mock_reset.call_count == 0


# ---------------------------------------------------------------------------
# Unit test 2: reset=True — reset_graph called before any loader
# ---------------------------------------------------------------------------


def test_seed_with_reset_calls_reset_graph_first() -> None:
    order: list[str] = []
    mock_driver = MagicMock()

    with patch(_P_RESET, side_effect=lambda d: order.append("reset")), \
         patch(_P_BASE, side_effect=lambda d: order.append("base") or _ZERO), \
         patch(_P_CONTRACTS, side_effect=lambda d: order.append("contracts") or _ZERO), \
         patch(_P_CHUNKS, side_effect=lambda d: order.append("chunks") or _ZERO):
        seed(mock_driver, reset=True)

    assert order[0] == "reset", f"reset_graph must run before loaders; got {order!r}"
    assert order[1:] == ["base", "contracts", "chunks"]


# ---------------------------------------------------------------------------
# Unit test 3: loader execution order is fixed
# ---------------------------------------------------------------------------


def test_seed_loader_order_is_fixed() -> None:
    order: list[str] = []
    mock_driver = MagicMock()

    with patch(_P_RESET), \
         patch(_P_BASE, side_effect=lambda d: order.append("base") or _ZERO), \
         patch(_P_CONTRACTS, side_effect=lambda d: order.append("contracts") or _ZERO), \
         patch(_P_CHUNKS, side_effect=lambda d: order.append("chunks") or _ZERO):
        seed(mock_driver)

    assert order == ["base", "contracts", "chunks"], (
        f"Expected base→contracts→chunks, got {order!r}"
    )


# ---------------------------------------------------------------------------
# Unit test 4: return value is the sum of all loader counts
# ---------------------------------------------------------------------------


def test_seed_returns_summed_counts() -> None:
    mock_driver = MagicMock()
    base_counts = {"nodes_written": 10, "edges_written": 5}
    contract_counts = {"nodes_written": 4, "edges_written": 8}
    chunk_counts = {"nodes_written": 20, "edges_written": 60}

    with patch(_P_RESET), \
         patch(_P_BASE, return_value=base_counts), \
         patch(_P_CONTRACTS, return_value=contract_counts), \
         patch(_P_CHUNKS, return_value=chunk_counts):
        result = seed(mock_driver)

    assert result["nodes_written"] == 10 + 4 + 20
    assert result["edges_written"] == 5 + 8 + 60


# ---------------------------------------------------------------------------
# Unit test 5: reset uses a batched DETACH DELETE with a LIMIT clause
# ---------------------------------------------------------------------------


def test_seed_reset_uses_batched_delete_with_limit() -> None:
    captured: list[str] = []

    def _capturing_run(query: str, *args: object, **kwargs: object) -> MagicMock:
        captured.append(query)
        result = MagicMock()
        result.consume.return_value.counters.nodes_deleted = 0  # exit loop immediately
        return result

    mock_session = MagicMock()
    mock_session.run = _capturing_run

    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__ = lambda s: mock_session
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    with patch(_P_BASE, return_value=_ZERO), \
         patch(_P_CONTRACTS, return_value=_ZERO), \
         patch(_P_CHUNKS, return_value=_ZERO):
        seed(mock_driver, reset=True)

    assert any("LIMIT" in q for q in captured), (
        f"Reset Cypher must contain LIMIT; queries seen: {captured!r}"
    )


# ---------------------------------------------------------------------------
# Unit test 6: first loader failure propagates; remaining loaders not called
# ---------------------------------------------------------------------------


def test_seed_base_loader_failure_propagates() -> None:
    mock_driver = MagicMock()

    with patch(_P_RESET), \
         patch(_P_BASE, side_effect=RuntimeError("base failed")), \
         patch(_P_CONTRACTS, return_value=_ZERO) as mock_contracts, \
         patch(_P_CHUNKS, return_value=_ZERO) as mock_chunks:
        with pytest.raises(RuntimeError, match="base failed"):
            seed(mock_driver)

    mock_contracts.assert_not_called()
    mock_chunks.assert_not_called()


# ---------------------------------------------------------------------------
# Unit test 7: second loader failure propagates; third loader not called
# ---------------------------------------------------------------------------


def test_seed_contracts_loader_failure_propagates() -> None:
    mock_driver = MagicMock()

    with patch(_P_RESET), \
         patch(_P_BASE, return_value=_ZERO), \
         patch(_P_CONTRACTS, side_effect=RuntimeError("contracts failed")), \
         patch(_P_CHUNKS, return_value=_ZERO) as mock_chunks:
        with pytest.raises(RuntimeError, match="contracts failed"):
            seed(mock_driver)

    mock_chunks.assert_not_called()


# ---------------------------------------------------------------------------
# Unit test 8: third loader failure propagates
# ---------------------------------------------------------------------------


def test_seed_chunks_loader_failure_propagates() -> None:
    mock_driver = MagicMock()

    with patch(_P_RESET), \
         patch(_P_BASE, return_value=_ZERO), \
         patch(_P_CONTRACTS, return_value=_ZERO), \
         patch(_P_CHUNKS, side_effect=RuntimeError("chunks failed")):
        with pytest.raises(RuntimeError, match="chunks failed"):
            seed(mock_driver)


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


@pytest.fixture(scope="module")
def neo4j_driver():
    driver = GraphDatabase.driver(_uri(), auth=_auth())
    try:
        driver.verify_connectivity()
    except Exception as exc:
        pytest.skip(f"Neo4j not reachable: {exc}")
    yield driver
    driver.close()


@pytest.fixture
def clean_graph(neo4j_driver):
    with neo4j_driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    yield
    with neo4j_driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")


def _labels_present(driver) -> set[str]:
    with driver.session() as s:
        rows = list(s.run("CALL db.labels() YIELD label RETURN label"))
        return {r["label"] for r in rows}


def _rel_types_present(driver) -> set[str]:
    with driver.session() as s:
        rows = list(s.run(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        ))
        return {r["relationshipType"] for r in rows}


# ---------------------------------------------------------------------------
# Integration test 9: all expected node labels are present after seed
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_seed_expected_labels_present(neo4j_driver, clean_graph) -> None:
    """All expected node labels are present in Neo4j after seed()."""
    with patch(_P_CHUNKS, return_value={"nodes_written": 0, "edges_written": 0}):
        seed(neo4j_driver)

    expected = {"Company", "Person", "Address", "Product", "Contract", "Clause"}
    found = _labels_present(neo4j_driver)
    missing = expected - found
    assert not missing, f"Missing node labels after seed: {missing!r}"


# ---------------------------------------------------------------------------
# Integration test 10: all expected relationship types are present after seed
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_seed_expected_relationships_present(neo4j_driver, clean_graph) -> None:
    """All expected relationship types are present in Neo4j after seed()."""
    with patch(_P_CHUNKS, return_value={"nodes_written": 0, "edges_written": 0}):
        seed(neo4j_driver)

    expected = {"DIRECTOR_OF", "REGISTERED_AT", "SUPPLIES", "HAS_CLAUSE", "PARTY_TO"}
    found = _rel_types_present(neo4j_driver)
    missing = expected - found
    assert not missing, f"Missing relationship types after seed: {missing!r}"
