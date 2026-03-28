"""Unit and integration tests for graphrag_assistant.loaders.contract_loader (T017.a).

Unit tests (8)
--------------
1.  test_load_contracts_single_row
2.  test_load_contracts_all_scalar_properties_set
3.  test_load_contracts_empty_rows_does_not_call_run
4.  test_load_clauses_single_clause
5.  test_zero_clauses_load_clauses_does_not_call_run
6.  test_load_party_to_edges_single_party
7.  test_load_party_to_edges_multi_party_mixed
8.  test_load_party_to_edges_missing_party_logs_warning_no_raise

Integration tests (2) — require a reachable Neo4j instance
-----------------------------------------------------------
9.  test_integration_party_to_count
10. test_integration_has_clause_count

Run integration tests with:  pytest -m integration
"""
from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock

import pytest

from graphrag_assistant.loaders.contract_loader import (
    load_clauses,
    load_contracts,
    load_has_clause_edges,
    load_party_to_edges,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_CONTRACT_ROW = {
    "contract_id": "CTR001",
    "title": "Supply Agreement Alpha",
    "effective_date": "2024-01-01",
    "expiry_date": "2026-12-31",
    "status": "active",
    "value_usd": "500000",
    "party_ids": "C001|P001",
}

_CLAUSE_ROW = {
    "clause_id": "CL001",
    "contract_id": "CTR001",
    "clause_type": "indemnity",
    "clause_order": "1",
    "text": "Each party shall indemnify the other.",
}


def _mock_session_finds(*party_ids: str) -> MagicMock:
    """Return a mock session whose .run() resolves the supplied party_ids in order.

    For each party_id a MATCH call (returns one record) and a MERGE call
    (return value ignored) are queued via side_effect.
    """
    session = MagicMock()
    effects: list = []
    for pid in party_ids:
        effects.append([{"id": pid}])  # MATCH → found
        effects.append(MagicMock())    # MERGE → ignored
    session.run.side_effect = effects
    return session


def _mock_session_missing() -> MagicMock:
    """Return a mock session whose MATCH always returns an empty result."""
    session = MagicMock()
    session.run.return_value = []
    return session


# ---------------------------------------------------------------------------
# Unit test 1: load_contracts — single row calls tx.run once
# ---------------------------------------------------------------------------


def test_load_contracts_single_row() -> None:
    mock_tx = MagicMock()
    load_contracts(mock_tx, [_CONTRACT_ROW])
    mock_tx.run.assert_called_once()
    positional_args = mock_tx.run.call_args[0]
    assert positional_args[1] == {"rows": [_CONTRACT_ROW]}


# ---------------------------------------------------------------------------
# Unit test 2: load_contracts — all scalar property names appear in Cypher
# ---------------------------------------------------------------------------


def test_load_contracts_all_scalar_properties_set() -> None:
    mock_tx = MagicMock()
    load_contracts(mock_tx, [_CONTRACT_ROW])
    cypher: str = mock_tx.run.call_args[0][0]
    for prop in ("contract_id", "title", "effective_date", "expiry_date", "status", "value_usd"):
        assert prop in cypher, f"Expected property '{prop}' in Cypher, got: {cypher}"


# ---------------------------------------------------------------------------
# Unit test 3: load_contracts — empty rows must not call tx.run
# ---------------------------------------------------------------------------


def test_load_contracts_empty_rows_does_not_call_run() -> None:
    mock_tx = MagicMock()
    load_contracts(mock_tx, [])
    assert mock_tx.run.call_count == 0
    mock_tx.run.assert_not_called()


# ---------------------------------------------------------------------------
# Unit test 4: load_clauses — single clause calls tx.run once
# ---------------------------------------------------------------------------


def test_load_clauses_single_clause() -> None:
    mock_tx = MagicMock()
    load_clauses(mock_tx, [_CLAUSE_ROW])
    mock_tx.run.assert_called_once()
    cypher: str = mock_tx.run.call_args[0][0]
    for prop in ("clause_id", "contract_id", "clause_type", "clause_order", "text"):
        assert prop in cypher, f"Expected property '{prop}' in Cypher, got: {cypher}"


# ---------------------------------------------------------------------------
# Unit test 5: load_clauses — zero rows must not call tx.run
# ---------------------------------------------------------------------------


def test_zero_clauses_load_clauses_does_not_call_run() -> None:
    mock_tx = MagicMock()
    load_clauses(mock_tx, [])
    assert mock_tx.run.call_count == 0
    mock_tx.run.assert_not_called()


# ---------------------------------------------------------------------------
# Unit test 6: load_party_to_edges — single party resolved → 1 edge returned
# ---------------------------------------------------------------------------


def test_load_party_to_edges_single_party() -> None:
    session = _mock_session_finds("C001")
    rows = [{"contract_id": "CTR001", "party_ids": "C001"}]
    result = load_party_to_edges(session, rows)
    assert result == 1, f"Expected 1 edge written, got {result}"
    # MATCH call + MERGE call
    assert session.run.call_count == 2


# ---------------------------------------------------------------------------
# Unit test 7: load_party_to_edges — multi-party (Company + Person) → 2 edges
# ---------------------------------------------------------------------------


def test_load_party_to_edges_multi_party_mixed() -> None:
    session = _mock_session_finds("C001", "P001")
    rows = [{"contract_id": "CTR001", "party_ids": "C001|P001"}]
    result = load_party_to_edges(session, rows)
    assert result == 2, f"Expected 2 edges written for two parties, got {result}"
    # Two parties × (1 MATCH + 1 MERGE) = 4 calls
    assert session.run.call_count == 4


# ---------------------------------------------------------------------------
# Unit test 8: load_party_to_edges — missing party logs WARNING, does not raise
# ---------------------------------------------------------------------------


def test_load_party_to_edges_missing_party_logs_warning_no_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _mock_session_missing()
    rows = [{"contract_id": "CTR001", "party_ids": "GHOST99"}]
    with caplog.at_level(logging.WARNING, logger="graphrag_assistant.loaders.contract_loader"):
        result = load_party_to_edges(session, rows)
    assert result == 0, f"Expected 0 edges when party is missing, got {result}"
    assert any("GHOST99" in msg for msg in caplog.messages), (
        "Expected a WARNING mentioning the missing party_id 'GHOST99'"
    )


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
    from neo4j import GraphDatabase

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


def _seed_minimal(driver) -> None:
    """Seed one Company, one Person, one Contract, and one Clause for integration tests."""
    with driver.session() as s:
        s.run("MERGE (c:Company {id: 'C001'}) SET c.name = 'Acme Corp'")
        s.run("MERGE (p:Person {id: 'P001'}) SET p.name = 'Jane Smith'")


def _count_rel(driver, rel_type: str) -> int:
    with driver.session() as s:
        rows = list(s.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS cnt"))
        return rows[0]["cnt"]


# ---------------------------------------------------------------------------
# Integration test 9: PARTY_TO count matches CSV party rows
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_party_to_count(neo4j_driver, clean_graph) -> None:
    _seed_minimal(neo4j_driver)

    contract_rows = [{"contract_id": "CTR001", "party_ids": "C001|P001"}]

    with neo4j_driver.session() as s:
        s.run(
            "MERGE (c:Contract {contract_id: 'CTR001'}) "
            "SET c.title = 'Test', c.effective_date = '2024-01-01', "
            "c.expiry_date = '2025-01-01', c.status = 'active', c.value_usd = 1000.0"
        )

    edges_written = load_party_to_edges(neo4j_driver.session(), contract_rows)

    assert edges_written == 2, f"Expected 2 PARTY_TO edges, got {edges_written}"
    assert _count_rel(neo4j_driver, "PARTY_TO") == 2, (
        "Graph PARTY_TO count must match number of resolved party rows"
    )


# ---------------------------------------------------------------------------
# Integration test 10: HAS_CLAUSE count matches clause rows
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_has_clause_count(neo4j_driver, clean_graph) -> None:
    _seed_minimal(neo4j_driver)

    clause_rows = [
        {
            "clause_id": "CL001",
            "contract_id": "CTR001",
            "clause_type": "indemnity",
            "clause_order": "1",
            "text": "Indemnity clause text.",
        },
        {
            "clause_id": "CL002",
            "contract_id": "CTR001",
            "clause_type": "payment",
            "clause_order": "2",
            "text": "Payment terms text.",
        },
    ]
    has_clause_rows = [
        {"contract_id": "CTR001", "clause_id": "CL001", "clause_order": "1"},
        {"contract_id": "CTR001", "clause_id": "CL002", "clause_order": "2"},
    ]

    with neo4j_driver.session() as s:
        s.execute_write(
            lambda tx: load_contracts(
                tx,
                [
                    {
                        "contract_id": "CTR001",
                        "title": "Test",
                        "effective_date": "2024-01-01",
                        "expiry_date": "2025-01-01",
                        "status": "active",
                        "value_usd": "1000",
                    }
                ],
            )
        )
        s.execute_write(lambda tx: load_clauses(tx, clause_rows))
        s.execute_write(lambda tx: load_has_clause_edges(tx, has_clause_rows))

    assert _count_rel(neo4j_driver, "HAS_CLAUSE") == 2, (
        "HAS_CLAUSE count must match number of clause rows"
    )
