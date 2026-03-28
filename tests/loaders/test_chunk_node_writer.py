"""Tests for graphrag_assistant.loaders.chunk_node_writer (T018.b).

Unit tests (12)
---------------
 1. test_write_chunk_nodes_empty_input_does_not_call_session
 2. test_write_chunk_nodes_single_chunk_passes_correct_row_to_session
 3. test_write_chunk_nodes_multiple_chunks_batched_in_single_unwind
 4. test_write_chunk_graph_empty_input_returns_zero_result_no_embed
 5. test_write_chunk_graph_no_company_ids_skips_about_company_query
 6. test_write_chunk_graph_multiple_companies_sends_one_row_per_company
 7. test_write_chunk_graph_mixed_related_entities_sends_one_row_per_entity
 8. test_write_chunk_graph_missing_contract_logs_warning_and_skips_chunk
 9. test_write_chunk_graph_embedding_called_exactly_once_per_run
10. test_write_chunk_graph_embedding_called_once_for_multiple_chunks
11. test_write_chunk_graph_idempotency_same_call_structure_on_second_run
12. test_write_chunk_graph_no_related_entities_skips_related_to_query

Integration tests (2) — require a reachable Neo4j instance (pytest -m integration)
------------------------------------------------------------------------------------
13. test_integration_from_contract_count_equals_chunk_count
14. test_integration_at_least_one_about_company_edge_exists
"""
from __future__ import annotations

import logging
import os
from typing import Any
from unittest.mock import MagicMock

import pytest

from graphrag_assistant.loaders.chunk_node_writer import (
    ChunkWriteResult,
    write_chunk_graph,
    write_chunk_nodes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_CHUNK = {
    "chunk_id": "K001_0",
    "contract_id": "K001",
    "text": "Indemnity clause text.",
    "company_ids": ["C001"],
    "related_entity_ids": ["P001"],
}

_EMBEDDED_CHUNK = {**_VALID_CHUNK, "embedding": [0.1, 0.2, 0.3, 0.4]}


def _make_write_session() -> MagicMock:
    """Session mock for write_chunk_nodes: run() returns None."""
    session = MagicMock()
    session.run.return_value = None
    return session


def _make_graph_session(found_contract_ids: list[str] | None = None) -> MagicMock:
    """Session mock for write_chunk_graph.

    Call 1 (MATCH_CONTRACT): returns rows [{"cid": id} ...] for found contracts.
    Subsequent calls (writes): return a mock with consume().counters.relationships_created=0.
    """
    session = MagicMock()
    found = found_contract_ids or []

    _write_result = MagicMock()
    _write_result.consume.return_value.counters.relationships_created = 0

    call_count: list[int] = [0]

    def _side_effect(cypher: str, params: Any = None) -> Any:
        call_count[0] += 1
        if call_count[0] == 1:
            return [{"cid": cid} for cid in found]
        return _write_result

    session.run.side_effect = _side_effect
    return session


def _stub_embedder(dim: int = 4) -> MagicMock:
    """Deterministic embedding stub: returns vectors of length dim."""
    embedder = MagicMock()
    embedder.embed.side_effect = lambda texts: [[0.1 * i] * dim for i in range(len(texts))]
    return embedder


# ---------------------------------------------------------------------------
# Unit test 1: write_chunk_nodes — empty input does not touch session
# ---------------------------------------------------------------------------


def test_write_chunk_nodes_empty_input_does_not_call_session() -> None:
    session = _make_write_session()
    write_chunk_nodes(session, [])
    assert session.run.call_count == 0


# ---------------------------------------------------------------------------
# Unit test 2: write_chunk_nodes — single chunk passes correct row
# ---------------------------------------------------------------------------


def test_write_chunk_nodes_single_chunk_passes_correct_row_to_session() -> None:
    session = _make_write_session()
    write_chunk_nodes(session, [_EMBEDDED_CHUNK])

    session.run.assert_called_once()
    _, params = session.run.call_args[0]
    assert len(params["rows"]) == 1
    row = params["rows"][0]
    assert row["chunk_id"] == "K001_0"
    assert row["contract_id"] == "K001"
    assert row["text"] == "Indemnity clause text."
    assert row["embedding"] == [0.1, 0.2, 0.3, 0.4]


# ---------------------------------------------------------------------------
# Unit test 3: write_chunk_nodes — multiple chunks batched in one UNWIND call
# ---------------------------------------------------------------------------


def test_write_chunk_nodes_multiple_chunks_batched_in_single_unwind() -> None:
    chunks = [
        {**_EMBEDDED_CHUNK, "chunk_id": f"K001_{i}", "embedding": [float(i)] * 4}
        for i in range(3)
    ]
    session = _make_write_session()
    write_chunk_nodes(session, chunks)

    # Only one session.run call regardless of chunk count.
    assert session.run.call_count == 1
    _, params = session.run.call_args[0]
    assert len(params["rows"]) == 3
    assert [r["chunk_id"] for r in params["rows"]] == ["K001_0", "K001_1", "K001_2"]


# ---------------------------------------------------------------------------
# Unit test 4: write_chunk_graph — empty input returns zero result, no embed
# ---------------------------------------------------------------------------


def test_write_chunk_graph_empty_input_returns_zero_result_no_embed() -> None:
    session = MagicMock()
    embedder = _stub_embedder()

    result = write_chunk_graph(session, [], embedder)

    assert result == ChunkWriteResult(0, 0, 0, 0)
    embedder.embed.assert_not_called()
    session.run.assert_not_called()


# ---------------------------------------------------------------------------
# Unit test 5: chunk with no company_ids — ABOUT_COMPANY query not issued
# ---------------------------------------------------------------------------


def test_write_chunk_graph_no_company_ids_skips_about_company_query() -> None:
    chunk = {**_VALID_CHUNK, "company_ids": [], "related_entity_ids": []}
    session = _make_graph_session(found_contract_ids=["K001"])
    embedder = _stub_embedder()

    write_chunk_graph(session, [chunk], embedder)

    # Calls: MATCH_CONTRACT + MERGE_CHUNK_NODES + MERGE_FROM_CONTRACT = 3
    # No ABOUT_COMPANY or RELATED_TO because both lists are empty.
    assert session.run.call_count == 3


# ---------------------------------------------------------------------------
# Unit test 6: multiple companies — one row per company sent to ABOUT_COMPANY
# ---------------------------------------------------------------------------


def test_write_chunk_graph_multiple_companies_sends_one_row_per_company() -> None:
    chunk = {**_VALID_CHUNK, "company_ids": ["C001", "C002"], "related_entity_ids": []}
    session = _make_graph_session(found_contract_ids=["K001"])
    embedder = _stub_embedder()

    write_chunk_graph(session, [chunk], embedder)

    # Calls: MATCH_CONTRACT, MERGE_CHUNK_NODES, MERGE_FROM_CONTRACT, MERGE_ABOUT_COMPANY = 4
    assert session.run.call_count == 4
    ac_params = session.run.call_args_list[3][0][1]
    company_ids_sent = sorted(r["company_id"] for r in ac_params["rows"])
    assert company_ids_sent == ["C001", "C002"]


# ---------------------------------------------------------------------------
# Unit test 7: mixed related_entity_ids — one row per entity in RELATED_TO
# ---------------------------------------------------------------------------


def test_write_chunk_graph_mixed_related_entities_sends_one_row_per_entity() -> None:
    chunk = {
        **_VALID_CHUNK,
        "company_ids": [],
        "related_entity_ids": ["P001", "A001", "PRD001"],
    }
    session = _make_graph_session(found_contract_ids=["K001"])
    embedder = _stub_embedder()

    write_chunk_graph(session, [chunk], embedder)

    # Calls: MATCH_CONTRACT, MERGE_CHUNK_NODES, MERGE_FROM_CONTRACT, MERGE_RELATED_TO = 4
    assert session.run.call_count == 4
    rt_params = session.run.call_args_list[3][0][1]
    entity_ids_sent = sorted(r["entity_id"] for r in rt_params["rows"])
    assert entity_ids_sent == ["A001", "P001", "PRD001"]


# ---------------------------------------------------------------------------
# Unit test 8: missing contract_id — warning logged, chunk skipped
# ---------------------------------------------------------------------------


def test_write_chunk_graph_missing_contract_logs_warning_and_skips_chunk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chunk = {**_VALID_CHUNK, "contract_id": "MISSING_CTR"}
    session = _make_graph_session(found_contract_ids=[])
    embedder = _stub_embedder()

    with caplog.at_level(
        logging.WARNING, logger="graphrag_assistant.loaders.chunk_node_writer"
    ):
        result = write_chunk_graph(session, [chunk], embedder)

    assert result == ChunkWriteResult(0, 0, 0, 0)
    assert any("MISSING_CTR" in msg for msg in caplog.messages), (
        "Expected a WARNING mentioning the missing contract_id 'MISSING_CTR'"
    )
    # Only the MATCH_CONTRACT read call; no writes because chunk was skipped.
    assert session.run.call_count == 1


# ---------------------------------------------------------------------------
# Unit test 9: embed() called exactly once for a single chunk per run
# ---------------------------------------------------------------------------


def test_write_chunk_graph_embedding_called_exactly_once_per_run() -> None:
    chunk = {**_VALID_CHUNK, "company_ids": [], "related_entity_ids": []}
    session = _make_graph_session(found_contract_ids=["K001"])
    embedder = _stub_embedder()

    write_chunk_graph(session, [chunk], embedder)

    embedder.embed.assert_called_once()
    texts_passed = embedder.embed.call_args[0][0]
    assert texts_passed == [chunk["text"]]


# ---------------------------------------------------------------------------
# Unit test 10: embed() called once even when multiple chunks are present
# ---------------------------------------------------------------------------


def test_write_chunk_graph_embedding_called_once_for_multiple_chunks() -> None:
    chunks = [
        {**_VALID_CHUNK, "chunk_id": f"K001_{i}", "company_ids": [], "related_entity_ids": []}
        for i in range(3)
    ]
    session = _make_graph_session(found_contract_ids=["K001"])
    embedder = _stub_embedder()

    write_chunk_graph(session, chunks, embedder)

    embedder.embed.assert_called_once()
    texts_passed = embedder.embed.call_args[0][0]
    assert len(texts_passed) == 3


# ---------------------------------------------------------------------------
# Unit test 11: idempotency — second run produces same call structure
# ---------------------------------------------------------------------------


def test_write_chunk_graph_idempotency_same_call_structure_on_second_run() -> None:
    chunk = {**_VALID_CHUNK, "company_ids": ["C001"], "related_entity_ids": []}

    session_a = _make_graph_session(found_contract_ids=["K001"])
    session_b = _make_graph_session(found_contract_ids=["K001"])
    embedder = _stub_embedder()

    write_chunk_graph(session_a, [chunk], embedder)
    write_chunk_graph(session_b, [chunk], embedder)

    # Both runs issue the same number of session.run calls.
    assert session_a.run.call_count == session_b.run.call_count
    # embed() called exactly once per invocation (twice total across two runs).
    assert embedder.embed.call_count == 2


# ---------------------------------------------------------------------------
# Unit test 12: no related_entity_ids — RELATED_TO query is not issued
# ---------------------------------------------------------------------------


def test_write_chunk_graph_no_related_entities_skips_related_to_query() -> None:
    chunk = {**_VALID_CHUNK, "company_ids": ["C001"], "related_entity_ids": []}
    session = _make_graph_session(found_contract_ids=["K001"])
    embedder = _stub_embedder()

    write_chunk_graph(session, [chunk], embedder)

    # Calls: MATCH_CONTRACT, MERGE_CHUNK_NODES, MERGE_FROM_CONTRACT, MERGE_ABOUT_COMPANY = 4
    # No RELATED_TO call because related_entity_ids is empty.
    assert session.run.call_count == 4


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


def _seed_contract_and_companies(driver) -> None:
    """Seed one Contract and two Company nodes for integration tests."""
    with driver.session() as s:
        s.run(
            "MERGE (c:Contract {contract_id: 'CTR_WG'}) "
            "SET c.title = 'Writer Integration Test Contract', "
            "c.effective_date = '2024-01-01', c.expiry_date = '2025-01-01', "
            "c.status = 'active', c.value_usd = 50000.0"
        )
        s.run("MERGE (co:Company {id: 'CO_WG1'}) SET co.name = 'Gamma Corp'")
        s.run("MERGE (co:Company {id: 'CO_WG2'}) SET co.name = 'Delta Ltd'")


def _count_rel(driver, rel_type: str) -> int:
    with driver.session() as s:
        rows = list(s.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS cnt"))
        return rows[0]["cnt"]


class _FixedEmbedder:
    """Deterministic 4-dim embedding stub."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(i), 0.0, 0.0, 0.0] for i in range(len(texts))]


# ---------------------------------------------------------------------------
# Integration test 13: FROM_CONTRACT edge count equals chunk count
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_from_contract_count_equals_chunk_count(
    neo4j_driver, clean_graph
) -> None:
    _seed_contract_and_companies(neo4j_driver)

    chunks = [
        {
            "chunk_id": f"WG_{i}",
            "contract_id": "CTR_WG",
            "text": f"Writer integration chunk {i}.",
            "company_ids": ["CO_WG1"],
            "related_entity_ids": [],
        }
        for i in range(4)
    ]

    with neo4j_driver.session() as session:
        result = write_chunk_graph(session, chunks, _FixedEmbedder())

    assert result.chunks_written == 4
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
    _seed_contract_and_companies(neo4j_driver)

    chunks = [
        {
            "chunk_id": "WG_AC_0",
            "contract_id": "CTR_WG",
            "text": "Chunk referencing two companies.",
            "company_ids": ["CO_WG1", "CO_WG2"],
            "related_entity_ids": [],
        }
    ]

    with neo4j_driver.session() as session:
        result = write_chunk_graph(session, chunks, _FixedEmbedder())

    assert result.about_company_edges >= 1, (
        "Expected at least one ABOUT_COMPANY edge after loading a chunk with company_ids"
    )
    graph_count = _count_rel(neo4j_driver, "ABOUT_COMPANY")
    assert graph_count >= 1, (
        f"MATCH ()-[:ABOUT_COMPANY]->() returned {graph_count}, expected >= 1"
    )
