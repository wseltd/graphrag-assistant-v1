"""Tests for graphrag_assistant.loaders.chunk_reader (T018.a).

Unit tests (12)
---------------
 1. test_load_chunks_returns_records_for_valid_jsonl
 2. test_load_chunks_skips_blank_lines
 3. test_load_chunks_empty_file_returns_empty_list
 4. test_load_chunks_raises_value_error_on_missing_field
 5. test_load_chunks_raises_value_error_on_invalid_json
 6. test_load_chunks_to_neo4j_empty_input_returns_zero_result
 7. test_load_chunks_to_neo4j_no_company_ids_no_about_company_rows
 8. test_load_chunks_to_neo4j_multiple_companies_sends_one_row_per_company
 9. test_load_chunks_to_neo4j_mixed_related_entity_types_sends_one_row_per_entity
10. test_load_chunks_to_neo4j_missing_contract_logs_warning_and_skips_chunk
11. test_load_chunks_to_neo4j_embedding_called_exactly_once_per_run
12. test_load_chunks_to_neo4j_embedding_called_once_for_multiple_chunks

Integration tests (2) — require a reachable Neo4j instance (pytest -m integration)
------------------------------------------------------------------------------------
13. test_integration_from_contract_count_equals_chunk_count
14. test_integration_at_least_one_about_company_edge_exists
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any
from unittest.mock import MagicMock

import pytest

from graphrag_assistant.loaders.chunk_reader import (
    ChunkLoadResult,
    load_chunks,
    load_chunks_to_neo4j,
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


def _write_jsonl(tmp_path_str: str, records: list[dict]) -> str:
    """Write *records* as JSONL to *tmp_path_str* and return the path."""
    with open(tmp_path_str, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return tmp_path_str


def _make_driver(session_mock: MagicMock) -> MagicMock:
    """Return a driver mock whose session() context manager yields *session_mock*."""
    driver = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = session_mock
    cm.__exit__.return_value = False
    driver.session.return_value = cm
    return driver


def _make_session(found_contract_ids: list[str] | None = None) -> MagicMock:
    """Build a session mock.

    Call 1  (MATCH_CONTRACT) returns rows [{cid: id} …] for *found_contract_ids*.
    Subsequent write calls return a mock whose .consume().counters.relationships_created
    equals 0 by default (unit tests verify call structure, not Neo4j counters).
    """
    session = MagicMock()
    found = found_contract_ids or []

    _write_result = MagicMock()
    _write_result.consume.return_value.counters.relationships_created = 0

    call_count: list[int] = [0]

    def _side_effect(cypher: str, params: Any = None) -> Any:
        call_count[0] += 1
        if call_count[0] == 1:
            # First call is always the contract-lookup read query.
            return [{"cid": cid} for cid in found]
        return _write_result

    session.run.side_effect = _side_effect
    return session


def _stub_embedder(dim: int = 4) -> MagicMock:
    """Return an embedding provider stub that returns deterministic vectors."""
    embedder = MagicMock()
    embedder.embed.side_effect = lambda texts: [[0.1 * i] * dim for i in range(len(texts))]
    return embedder


# ---------------------------------------------------------------------------
# Unit test 1: load_chunks returns records from a valid JSONL file
# ---------------------------------------------------------------------------


def test_load_chunks_returns_records_for_valid_jsonl(tmp_path) -> None:
    path = _write_jsonl(str(tmp_path / "chunks.jsonl"), [_VALID_CHUNK])
    records = load_chunks(path)
    assert len(records) == 1
    assert records[0]["chunk_id"] == "K001_0"
    assert records[0]["company_ids"] == ["C001"]


# ---------------------------------------------------------------------------
# Unit test 2: load_chunks skips blank lines without raising
# ---------------------------------------------------------------------------


def test_load_chunks_skips_blank_lines(tmp_path) -> None:
    path = str(tmp_path / "chunks.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write(json.dumps(_VALID_CHUNK) + "\n")
        fh.write("   \n")
    records = load_chunks(path)
    assert len(records) == 1


# ---------------------------------------------------------------------------
# Unit test 3: load_chunks returns empty list for an empty file
# ---------------------------------------------------------------------------


def test_load_chunks_empty_file_returns_empty_list(tmp_path) -> None:
    path = str(tmp_path / "empty.jsonl")
    open(path, "w").close()
    assert load_chunks(path) == []


# ---------------------------------------------------------------------------
# Unit test 4: load_chunks raises ValueError when a required field is absent
# ---------------------------------------------------------------------------


def test_load_chunks_raises_value_error_on_missing_field(tmp_path) -> None:
    bad = {k: v for k, v in _VALID_CHUNK.items() if k != "contract_id"}
    path = _write_jsonl(str(tmp_path / "chunks.jsonl"), [bad])
    with pytest.raises(ValueError, match="contract_id"):
        load_chunks(path)


# ---------------------------------------------------------------------------
# Unit test 5: load_chunks raises ValueError on malformed JSON
# ---------------------------------------------------------------------------


def test_load_chunks_raises_value_error_on_invalid_json(tmp_path) -> None:
    path = str(tmp_path / "chunks.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{not valid json}\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_chunks(path)


# ---------------------------------------------------------------------------
# Unit test 6: empty chunks list → zero ChunkLoadResult, no embed call
# ---------------------------------------------------------------------------


def test_load_chunks_to_neo4j_empty_input_returns_zero_result() -> None:
    embedder = _stub_embedder()
    driver = MagicMock()
    result = load_chunks_to_neo4j([], driver, embedder)
    assert result == ChunkLoadResult(0, 0, 0, 0)
    embedder.embed.assert_not_called()
    driver.session.assert_not_called()


# ---------------------------------------------------------------------------
# Unit test 7: chunk with no company_ids — ABOUT_COMPANY query not called
# ---------------------------------------------------------------------------


def test_load_chunks_to_neo4j_no_company_ids_no_about_company_rows() -> None:
    chunk = {**_VALID_CHUNK, "company_ids": [], "related_entity_ids": []}
    session = _make_session(found_contract_ids=["K001"])
    driver = _make_driver(session)
    embedder = _stub_embedder()

    load_chunks_to_neo4j([chunk], driver, embedder)

    # Calls: 1 MATCH_CONTRACT + 1 MERGE_CHUNKS + 1 MERGE_FROM_CONTRACT = 3 total.
    # No ABOUT_COMPANY or RELATED_TO call because both lists are empty.
    assert session.run.call_count == 3


# ---------------------------------------------------------------------------
# Unit test 8: chunk with multiple companies → one row per company in ABOUT_COMPANY
# ---------------------------------------------------------------------------


def test_load_chunks_to_neo4j_multiple_companies_sends_one_row_per_company() -> None:
    chunk = {**_VALID_CHUNK, "company_ids": ["C001", "C002"], "related_entity_ids": []}
    session = _make_session(found_contract_ids=["K001"])
    driver = _make_driver(session)
    embedder = _stub_embedder()

    load_chunks_to_neo4j([chunk], driver, embedder)

    # Calls: MATCH_CONTRACT, MERGE_CHUNKS, MERGE_FROM_CONTRACT, MERGE_ABOUT_COMPANY = 4
    assert session.run.call_count == 4
    # The ABOUT_COMPANY call (4th) must pass rows with two company entries.
    about_call_params = session.run.call_args_list[3][0][1]
    company_ids_sent = [r["company_id"] for r in about_call_params["rows"]]
    assert sorted(company_ids_sent) == ["C001", "C002"]


# ---------------------------------------------------------------------------
# Unit test 9: mixed related_entity_ids → one row per entity in RELATED_TO
# ---------------------------------------------------------------------------


def test_load_chunks_to_neo4j_mixed_related_entity_types_sends_one_row_per_entity() -> None:
    chunk = {
        **_VALID_CHUNK,
        "company_ids": [],
        "related_entity_ids": ["P001", "A001", "PRD001"],
    }
    session = _make_session(found_contract_ids=["K001"])
    driver = _make_driver(session)
    embedder = _stub_embedder()

    load_chunks_to_neo4j([chunk], driver, embedder)

    # Calls: MATCH_CONTRACT, MERGE_CHUNKS, MERGE_FROM_CONTRACT, MERGE_RELATED_TO = 4
    assert session.run.call_count == 4
    rt_call_params = session.run.call_args_list[3][0][1]
    entity_ids_sent = [r["entity_id"] for r in rt_call_params["rows"]]
    assert sorted(entity_ids_sent) == ["A001", "P001", "PRD001"]


# ---------------------------------------------------------------------------
# Unit test 10: missing contract → warning logged, chunk skipped, 0 result
# ---------------------------------------------------------------------------


def test_load_chunks_to_neo4j_missing_contract_logs_warning_and_skips_chunk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chunk = {**_VALID_CHUNK, "contract_id": "MISSING_CTR"}
    # Session returns empty found_contracts so no contract matches.
    session = _make_session(found_contract_ids=[])
    driver = _make_driver(session)
    embedder = _stub_embedder()

    with caplog.at_level(logging.WARNING, logger="graphrag_assistant.loaders.chunk_reader"):
        result = load_chunks_to_neo4j([chunk], driver, embedder)

    assert result == ChunkLoadResult(0, 0, 0, 0)
    assert any("MISSING_CTR" in msg for msg in caplog.messages), (
        "Expected a WARNING mentioning the missing contract_id 'MISSING_CTR'"
    )
    # Only the MATCH_CONTRACT read call was made; no writes.
    assert session.run.call_count == 1


# ---------------------------------------------------------------------------
# Unit test 11: embed() called exactly once for a single chunk
# ---------------------------------------------------------------------------


def test_load_chunks_to_neo4j_embedding_called_exactly_once_per_run() -> None:
    chunk = {**_VALID_CHUNK, "company_ids": [], "related_entity_ids": []}
    session = _make_session(found_contract_ids=["K001"])
    driver = _make_driver(session)
    embedder = _stub_embedder()

    load_chunks_to_neo4j([chunk], driver, embedder)

    embedder.embed.assert_called_once()
    texts_passed = embedder.embed.call_args[0][0]
    assert texts_passed == [chunk["text"]]


# ---------------------------------------------------------------------------
# Unit test 12: embed() called once even when multiple chunks are present
# ---------------------------------------------------------------------------


def test_load_chunks_to_neo4j_embedding_called_once_for_multiple_chunks() -> None:
    chunks = [
        {**_VALID_CHUNK, "chunk_id": f"K001_{i}", "company_ids": [], "related_entity_ids": []}
        for i in range(3)
    ]
    session = _make_session(found_contract_ids=["K001"])
    driver = _make_driver(session)
    embedder = _stub_embedder()

    load_chunks_to_neo4j(chunks, driver, embedder)

    # embed() called exactly once with all three texts.
    embedder.embed.assert_called_once()
    texts_passed = embedder.embed.call_args[0][0]
    assert len(texts_passed) == 3


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
            "MERGE (c:Contract {contract_id: 'CTR_IT'}) "
            "SET c.title = 'Integration Test Contract', "
            "c.effective_date = '2024-01-01', c.expiry_date = '2025-01-01', "
            "c.status = 'active', c.value_usd = 100000.0"
        )
        s.run("MERGE (co:Company {id: 'CO_IT1'}) SET co.name = 'Alpha Corp'")
        s.run("MERGE (co:Company {id: 'CO_IT2'}) SET co.name = 'Beta Ltd'")


def _count_rel(driver, rel_type: str) -> int:
    with driver.session() as s:
        rows = list(s.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS cnt"))
        return rows[0]["cnt"]


class _FixedEmbedder:
    """Deterministic embedding stub: returns 4-dim vectors."""

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
            "chunk_id": f"IT_{i}",
            "contract_id": "CTR_IT",
            "text": f"Integration test chunk {i}.",
            "company_ids": ["CO_IT1"],
            "related_entity_ids": [],
        }
        for i in range(4)
    ]

    result = load_chunks_to_neo4j(chunks, neo4j_driver, _FixedEmbedder())

    assert result.chunks_loaded == 4
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
            "chunk_id": "IT_AC_0",
            "contract_id": "CTR_IT",
            "text": "Chunk referencing two companies.",
            "company_ids": ["CO_IT1", "CO_IT2"],
            "related_entity_ids": [],
        }
    ]

    result = load_chunks_to_neo4j(chunks, neo4j_driver, _FixedEmbedder())

    assert result.about_company_edges >= 1, (
        "Expected at least one ABOUT_COMPANY edge after loading a chunk with company_ids"
    )
    graph_count = _count_rel(neo4j_driver, "ABOUT_COMPANY")
    assert graph_count >= 1, (
        f"MATCH ()-[:ABOUT_COMPANY]->() returned {graph_count}, expected >= 1"
    )
