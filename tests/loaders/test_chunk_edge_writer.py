"""Tests for graphrag_assistant.loaders.chunk_edge_writer (T018.c).

Unit tests (12)
---------------
 1. test_write_chunk_contract_edges_empty_input_does_not_call_session
 2. test_write_chunk_contract_edges_single_chunk_calls_session_once
 3. test_write_chunk_contract_edges_multiple_chunks_batched_in_single_unwind
 4. test_write_chunk_contract_edges_missing_contract_id_logs_warning_and_skips
 5. test_write_chunk_contract_edges_partial_missing_skips_only_invalid
 6. test_write_chunk_contract_edges_idempotency_same_call_structure_on_second_run
 7. test_write_chunk_company_edges_empty_input_does_not_call_session
 8. test_write_chunk_company_edges_no_company_ids_does_not_call_session
 9. test_write_chunk_company_edges_single_company_sends_correct_row
10. test_write_chunk_company_edges_multiple_companies_sends_one_row_per_company
11. test_write_chunk_company_edges_multiple_chunks_batched_in_single_unwind
12. test_write_chunk_company_edges_idempotency_same_call_structure_on_second_run

Integration tests (2) — require a reachable Neo4j instance (pytest -m integration)
------------------------------------------------------------------------------------
13. test_integration_from_contract_count_equals_chunk_count
14. test_integration_at_least_one_about_company_edge_exists
"""
from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock

import pytest

from graphrag_assistant.loaders.chunk_edge_writer import (
    write_chunk_company_edges,
    write_chunk_contract_edges,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHUNK_A = {
    "chunk_id": "K001_0",
    "contract_id": "K001",
    "text": "Indemnity clause.",
    "company_ids": ["C001"],
}

_CHUNK_B = {
    "chunk_id": "K001_1",
    "contract_id": "K001",
    "text": "Governing law clause.",
    "company_ids": ["C001", "C002"],
}


def _make_session() -> MagicMock:
    session = MagicMock()
    session.run.return_value = MagicMock()
    return session


# ---------------------------------------------------------------------------
# Unit test 1: write_chunk_contract_edges — empty input does not touch session
# ---------------------------------------------------------------------------


def test_write_chunk_contract_edges_empty_input_does_not_call_session() -> None:
    session = _make_session()
    write_chunk_contract_edges(session, [])
    assert session.run.call_count == 0


# ---------------------------------------------------------------------------
# Unit test 2: single valid chunk — session.run called once with correct row
# ---------------------------------------------------------------------------


def test_write_chunk_contract_edges_single_chunk_calls_session_once() -> None:
    session = _make_session()
    write_chunk_contract_edges(session, [_CHUNK_A])

    assert session.run.call_count == 1
    _, params = session.run.call_args[0]
    assert len(params["rows"]) == 1
    assert params["rows"][0] == {"chunk_id": "K001_0", "contract_id": "K001"}


# ---------------------------------------------------------------------------
# Unit test 3: multiple chunks batched into a single UNWIND call
# ---------------------------------------------------------------------------


def test_write_chunk_contract_edges_multiple_chunks_batched_in_single_unwind() -> None:
    chunks = [
        {"chunk_id": f"K001_{i}", "contract_id": "K001", "text": f"Chunk {i}."}
        for i in range(4)
    ]
    session = _make_session()
    write_chunk_contract_edges(session, chunks)

    assert session.run.call_count == 1
    _, params = session.run.call_args[0]
    assert len(params["rows"]) == 4


# ---------------------------------------------------------------------------
# Unit test 4: missing contract_id — warning logged, chunk skipped, no session call
# ---------------------------------------------------------------------------


def test_write_chunk_contract_edges_missing_contract_id_logs_warning_and_skips(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chunk = {"chunk_id": "K001_0", "contract_id": None, "text": "No contract."}
    session = _make_session()

    with caplog.at_level(
        logging.WARNING,
        logger="graphrag_assistant.loaders.chunk_edge_writer",
    ):
        write_chunk_contract_edges(session, [chunk])

    assert session.run.call_count == 0
    assert any("K001_0" in msg for msg in caplog.messages), (
        "Expected a WARNING mentioning the chunk_id of the skipped chunk"
    )


# ---------------------------------------------------------------------------
# Unit test 5: mixed valid/invalid — only valid chunks sent, invalid skipped
# ---------------------------------------------------------------------------


def test_write_chunk_contract_edges_partial_missing_skips_only_invalid(
    caplog: pytest.LogCaptureFixture,
) -> None:
    valid_chunk = {"chunk_id": "K001_0", "contract_id": "K001", "text": "Valid."}
    invalid_chunk = {"chunk_id": "K001_1", "contract_id": "", "text": "Invalid."}
    session = _make_session()

    with caplog.at_level(
        logging.WARNING,
        logger="graphrag_assistant.loaders.chunk_edge_writer",
    ):
        write_chunk_contract_edges(session, [valid_chunk, invalid_chunk])

    assert session.run.call_count == 1
    _, params = session.run.call_args[0]
    assert len(params["rows"]) == 1
    assert params["rows"][0]["chunk_id"] == "K001_0"
    assert any("K001_1" in msg for msg in caplog.messages), (
        "Expected WARNING mentioning the invalid chunk_id"
    )


# ---------------------------------------------------------------------------
# Unit test 6: idempotency — two calls produce the same session call structure
# ---------------------------------------------------------------------------


def test_write_chunk_contract_edges_idempotency_same_call_structure_on_second_run() -> None:
    session_a = _make_session()
    session_b = _make_session()

    write_chunk_contract_edges(session_a, [_CHUNK_A])
    write_chunk_contract_edges(session_b, [_CHUNK_A])

    assert session_a.run.call_count == session_b.run.call_count == 1
    params_a = session_a.run.call_args[0][1]
    params_b = session_b.run.call_args[0][1]
    assert params_a["rows"] == params_b["rows"]


# ---------------------------------------------------------------------------
# Unit test 7: write_chunk_company_edges — empty input does not touch session
# ---------------------------------------------------------------------------


def test_write_chunk_company_edges_empty_input_does_not_call_session() -> None:
    session = _make_session()
    write_chunk_company_edges(session, [])
    assert session.run.call_count == 0


# ---------------------------------------------------------------------------
# Unit test 8: chunk with no company_ids — no rows, session not called
# ---------------------------------------------------------------------------


def test_write_chunk_company_edges_no_company_ids_does_not_call_session() -> None:
    chunk = {
        "chunk_id": "K001_0", "contract_id": "K001",
        "text": "No companies.", "company_ids": [],
    }
    session = _make_session()
    write_chunk_company_edges(session, [chunk])
    assert session.run.call_count == 0


# ---------------------------------------------------------------------------
# Unit test 9: single company — session called once with correct row
# ---------------------------------------------------------------------------


def test_write_chunk_company_edges_single_company_sends_correct_row() -> None:
    chunk = {
        "chunk_id": "K001_0", "contract_id": "K001",
        "text": "One company.", "company_ids": ["C001"],
    }
    session = _make_session()
    write_chunk_company_edges(session, [chunk])

    assert session.run.call_count == 1
    _, params = session.run.call_args[0]
    assert len(params["rows"]) == 1
    assert params["rows"][0] == {"chunk_id": "K001_0", "company_id": "C001"}


# ---------------------------------------------------------------------------
# Unit test 10: multiple companies — one row per company_id
# ---------------------------------------------------------------------------


def test_write_chunk_company_edges_multiple_companies_sends_one_row_per_company() -> None:
    session = _make_session()
    write_chunk_company_edges(session, [_CHUNK_B])

    assert session.run.call_count == 1
    _, params = session.run.call_args[0]
    company_ids_sent = sorted(r["company_id"] for r in params["rows"])
    assert company_ids_sent == ["C001", "C002"]
    assert all(r["chunk_id"] == "K001_1" for r in params["rows"])


# ---------------------------------------------------------------------------
# Unit test 11: multiple chunks — all company rows batched in one UNWIND call
# ---------------------------------------------------------------------------


def test_write_chunk_company_edges_multiple_chunks_batched_in_single_unwind() -> None:
    chunks = [
        {"chunk_id": "K001_0", "contract_id": "K001", "text": "A.", "company_ids": ["C001"]},
        {"chunk_id": "K001_1", "contract_id": "K001", "text": "B.",
         "company_ids": ["C002", "C003"]},
    ]
    session = _make_session()
    write_chunk_company_edges(session, chunks)

    assert session.run.call_count == 1
    _, params = session.run.call_args[0]
    assert len(params["rows"]) == 3
    company_ids_sent = sorted(r["company_id"] for r in params["rows"])
    assert company_ids_sent == ["C001", "C002", "C003"]


# ---------------------------------------------------------------------------
# Unit test 12: idempotency — two calls produce the same session call structure
# ---------------------------------------------------------------------------


def test_write_chunk_company_edges_idempotency_same_call_structure_on_second_run() -> None:
    session_a = _make_session()
    session_b = _make_session()

    write_chunk_company_edges(session_a, [_CHUNK_B])
    write_chunk_company_edges(session_b, [_CHUNK_B])

    assert session_a.run.call_count == session_b.run.call_count == 1
    params_a = session_a.run.call_args[0][1]
    params_b = session_b.run.call_args[0][1]
    assert sorted(r["company_id"] for r in params_a["rows"]) == sorted(
        r["company_id"] for r in params_b["rows"]
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


@pytest.fixture()
def clean_graph(neo4j_driver):
    with neo4j_driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    yield
    with neo4j_driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")


def _seed_graph(driver) -> None:
    """Seed Contract, Chunk, and Company nodes for integration tests."""
    with driver.session() as s:
        s.run(
            "MERGE (ct:Contract {contract_id: 'CTR_EW'}) "
            "SET ct.title = 'Edge Writer Test Contract'"
        )
        for i in range(3):
            s.run(
                "MERGE (ch:Chunk {chunk_id: $cid}) "
                "SET ch.contract_id = 'CTR_EW', ch.text = $text, ch.embedding = [0.1, 0.2]",
                {"cid": f"EW_{i}", "text": f"Edge writer chunk {i}."},
            )
        s.run("MERGE (co:Company {id: 'CO_EW1'}) SET co.name = 'EdgeCorp'")


def _count_rel(driver, rel_type: str) -> int:
    with driver.session() as s:
        rows = list(s.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS cnt"))
        return rows[0]["cnt"]


# ---------------------------------------------------------------------------
# Integration test 13: FROM_CONTRACT edge count equals chunk count
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_from_contract_count_equals_chunk_count(
    neo4j_driver, clean_graph
) -> None:
    _seed_graph(neo4j_driver)

    chunks = [
        {"chunk_id": f"EW_{i}", "contract_id": "CTR_EW", "text": f"Chunk {i}."}
        for i in range(3)
    ]

    with neo4j_driver.session() as session:
        write_chunk_contract_edges(session, chunks)

    graph_count = _count_rel(neo4j_driver, "FROM_CONTRACT")
    assert graph_count == len(chunks), (
        f"FROM_CONTRACT edge count {graph_count} must equal chunk count {len(chunks)}"
    )


# ---------------------------------------------------------------------------
# Integration test 14: at least one ABOUT_COMPANY edge exists after loading
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_at_least_one_about_company_edge_exists(
    neo4j_driver, clean_graph
) -> None:
    _seed_graph(neo4j_driver)

    chunks = [
        {
            "chunk_id": "EW_0",
            "contract_id": "CTR_EW",
            "text": "Chunk referencing a company.",
            "company_ids": ["CO_EW1"],
        }
    ]

    with neo4j_driver.session() as session:
        write_chunk_company_edges(session, chunks)

    graph_count = _count_rel(neo4j_driver, "ABOUT_COMPANY")
    assert graph_count >= 1, (
        f"MATCH ()-[:ABOUT_COMPANY]->() returned {graph_count}, expected >= 1"
    )
