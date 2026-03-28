"""Tests for graphrag_assistant.loaders.chunk_related_writer (T018.d).

Unit tests (12)
---------------
 1. test_empty_input_does_not_call_session
 2. test_no_related_entity_ids_does_not_call_session
 3. test_missing_related_entity_ids_key_does_not_call_session
 4. test_none_related_entity_ids_does_not_call_session
 5. test_single_entity_issues_four_calls_one_per_label
 6. test_each_call_uses_correct_label_param
 7. test_rows_contain_correct_chunk_id_and_entity_id
 8. test_multiple_entities_all_rows_sent_to_each_label_call
 9. test_multiple_chunks_rows_aggregated_across_chunks
10. test_mixed_entity_types_all_ids_sent_to_each_label_call
11. test_idempotency_same_call_count_and_params_on_second_run
12. test_company_person_product_contract_labels_all_covered

Integration tests (2) — require a reachable Neo4j instance (pytest -m integration)
------------------------------------------------------------------------------------
13. test_integration_at_least_one_related_to_edge_created
14. test_integration_idempotency_same_count_on_second_run
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from graphrag_assistant.loaders.chunk_related_writer import (
    _ENTITY_LABELS,
    write_related_to_edges,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHUNK_COMPANY = {
    "chunk_id": "K001_0",
    "contract_id": "K001",
    "text": "Indemnity clause.",
    "related_entity_ids": ["C001"],
}

_CHUNK_MIXED = {
    "chunk_id": "K001_1",
    "contract_id": "K001",
    "text": "Multi-entity clause.",
    "related_entity_ids": ["C001", "P001", "PR001"],
}


def _make_session() -> MagicMock:
    session = MagicMock()
    session.run.return_value = MagicMock()
    return session


# ---------------------------------------------------------------------------
# Unit test 1: empty input — no session calls
# ---------------------------------------------------------------------------


def test_empty_input_does_not_call_session() -> None:
    session = _make_session()
    write_related_to_edges(session, [])
    assert session.run.call_count == 0


# ---------------------------------------------------------------------------
# Unit test 2: chunk with empty related_entity_ids — no session calls
# ---------------------------------------------------------------------------


def test_no_related_entity_ids_does_not_call_session() -> None:
    chunk = {"chunk_id": "K001_0", "contract_id": "K001", "text": "A.", "related_entity_ids": []}
    session = _make_session()
    write_related_to_edges(session, [chunk])
    assert session.run.call_count == 0


# ---------------------------------------------------------------------------
# Unit test 3: chunk without the related_entity_ids key — no session calls
# ---------------------------------------------------------------------------


def test_missing_related_entity_ids_key_does_not_call_session() -> None:
    chunk = {"chunk_id": "K001_0", "contract_id": "K001", "text": "No key."}
    session = _make_session()
    write_related_to_edges(session, [chunk])
    assert session.run.call_count == 0


# ---------------------------------------------------------------------------
# Unit test 4: chunk with related_entity_ids=None — no session calls
# ---------------------------------------------------------------------------


def test_none_related_entity_ids_does_not_call_session() -> None:
    chunk = {
        "chunk_id": "K001_0", "contract_id": "K001",
        "text": "None.", "related_entity_ids": None,
    }
    session = _make_session()
    write_related_to_edges(session, [chunk])
    assert session.run.call_count == 0


# ---------------------------------------------------------------------------
# Unit test 5: single entity → exactly 4 session.run calls (one per label)
# ---------------------------------------------------------------------------


def test_single_entity_issues_four_calls_one_per_label() -> None:
    session = _make_session()
    write_related_to_edges(session, [_CHUNK_COMPANY])
    assert session.run.call_count == len(_ENTITY_LABELS)


# ---------------------------------------------------------------------------
# Unit test 6: each call receives the correct label param
# ---------------------------------------------------------------------------


def test_each_call_uses_correct_label_param() -> None:
    session = _make_session()
    write_related_to_edges(session, [_CHUNK_COMPANY])

    called_labels = [call_args[0][1]["label"] for call_args in session.run.call_args_list]
    assert sorted(called_labels) == sorted(_ENTITY_LABELS)


# ---------------------------------------------------------------------------
# Unit test 7: rows contain correct chunk_id and entity_id in every call
# ---------------------------------------------------------------------------


def test_rows_contain_correct_chunk_id_and_entity_id() -> None:
    session = _make_session()
    write_related_to_edges(session, [_CHUNK_COMPANY])

    for call_args in session.run.call_args_list:
        params = call_args[0][1]
        assert len(params["rows"]) == 1
        assert params["rows"][0] == {"chunk_id": "K001_0", "entity_id": "C001"}


# ---------------------------------------------------------------------------
# Unit test 8: multiple entities — all rows sent to each label call
# ---------------------------------------------------------------------------


def test_multiple_entities_all_rows_sent_to_each_label_call() -> None:
    chunk = {
        "chunk_id": "K001_0",
        "contract_id": "K001",
        "text": "Two entities.",
        "related_entity_ids": ["C001", "P001"],
    }
    session = _make_session()
    write_related_to_edges(session, [chunk])

    assert session.run.call_count == len(_ENTITY_LABELS)
    for call_args in session.run.call_args_list:
        params = call_args[0][1]
        assert len(params["rows"]) == 2
        entity_ids = sorted(r["entity_id"] for r in params["rows"])
        assert entity_ids == ["C001", "P001"]


# ---------------------------------------------------------------------------
# Unit test 9: multiple chunks — rows from all chunks aggregated
# ---------------------------------------------------------------------------


def test_multiple_chunks_rows_aggregated_across_chunks() -> None:
    chunk_a = {
        "chunk_id": "K001_0", "contract_id": "K001",
        "text": "A.", "related_entity_ids": ["C001"],
    }
    chunk_b = {
        "chunk_id": "K001_1", "contract_id": "K001",
        "text": "B.", "related_entity_ids": ["P001"],
    }
    session = _make_session()
    write_related_to_edges(session, [chunk_a, chunk_b])

    assert session.run.call_count == len(_ENTITY_LABELS)
    for call_args in session.run.call_args_list:
        params = call_args[0][1]
        assert len(params["rows"]) == 2
        chunk_ids = sorted(r["chunk_id"] for r in params["rows"])
        assert chunk_ids == ["K001_0", "K001_1"]


# ---------------------------------------------------------------------------
# Unit test 10: mixed entity types — all entity_ids sent to each label call
# ---------------------------------------------------------------------------


def test_mixed_entity_types_all_ids_sent_to_each_label_call() -> None:
    session = _make_session()
    write_related_to_edges(session, [_CHUNK_MIXED])

    assert session.run.call_count == len(_ENTITY_LABELS)
    for call_args in session.run.call_args_list:
        params = call_args[0][1]
        entity_ids = sorted(r["entity_id"] for r in params["rows"])
        assert entity_ids == sorted(_CHUNK_MIXED["related_entity_ids"])


# ---------------------------------------------------------------------------
# Unit test 11: idempotency — two calls produce the same session call structure
# ---------------------------------------------------------------------------


def test_idempotency_same_call_count_and_params_on_second_run() -> None:
    session_a = _make_session()
    session_b = _make_session()

    write_related_to_edges(session_a, [_CHUNK_COMPANY])
    write_related_to_edges(session_b, [_CHUNK_COMPANY])

    assert session_a.run.call_count == session_b.run.call_count == len(_ENTITY_LABELS)
    pairs = zip(
        session_a.run.call_args_list, session_b.run.call_args_list, strict=True
    )
    for args_a, args_b in pairs:
        assert args_a[0][1]["rows"] == args_b[0][1]["rows"]
        assert args_a[0][1]["label"] == args_b[0][1]["label"]


# ---------------------------------------------------------------------------
# Unit test 12: all four label strings (Company, Person, Product, Contract) used
# ---------------------------------------------------------------------------


def test_company_person_product_contract_labels_all_covered() -> None:
    session = _make_session()
    write_related_to_edges(session, [_CHUNK_COMPANY])

    called_labels = {call_args[0][1]["label"] for call_args in session.run.call_args_list}
    assert called_labels == {"Company", "Person", "Product", "Contract"}


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


def _seed_related_graph(driver) -> None:
    """Seed Chunk and Company nodes for RELATED_TO integration tests."""
    with driver.session() as s:
        s.run(
            "MERGE (co:Company {id: 'CO_RT1'}) SET co.name = 'RelatedCorp'"
        )
        s.run(
            "MERGE (p:Person {id: 'PE_RT1'}) SET p.name = 'Jane Related'"
        )
        for i in range(2):
            s.run(
                "MERGE (ch:Chunk {chunk_id: $cid}) "
                "SET ch.contract_id = 'CTR_RT', ch.text = $text, ch.embedding = [0.1, 0.2]",
                {"cid": f"RT_{i}", "text": f"Related chunk {i}."},
            )


def _count_rel(driver, rel_type: str) -> int:
    with driver.session() as s:
        rows = list(s.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS cnt"))
        return rows[0]["cnt"]


# ---------------------------------------------------------------------------
# Integration test 13: at least one RELATED_TO edge created after loading
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_at_least_one_related_to_edge_created(
    neo4j_driver, clean_graph
) -> None:
    _seed_related_graph(neo4j_driver)

    chunks = [
        {
            "chunk_id": "RT_0",
            "contract_id": "CTR_RT",
            "text": "Related to a company.",
            "related_entity_ids": ["CO_RT1"],
        },
        {
            "chunk_id": "RT_1",
            "contract_id": "CTR_RT",
            "text": "Related to a person.",
            "related_entity_ids": ["PE_RT1"],
        },
    ]

    with neo4j_driver.session() as session:
        write_related_to_edges(session, chunks)

    count = _count_rel(neo4j_driver, "RELATED_TO")
    assert count >= 1, (
        f"MATCH ()-[:RELATED_TO]->() returned {count}, expected >= 1"
    )


# ---------------------------------------------------------------------------
# Integration test 14: idempotency — running twice produces no duplicate edges
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_idempotency_same_count_on_second_run(
    neo4j_driver, clean_graph
) -> None:
    _seed_related_graph(neo4j_driver)

    chunks = [
        {
            "chunk_id": "RT_0",
            "contract_id": "CTR_RT",
            "text": "Related chunk.",
            "related_entity_ids": ["CO_RT1", "PE_RT1"],
        }
    ]

    with neo4j_driver.session() as session:
        write_related_to_edges(session, chunks)
    count_first = _count_rel(neo4j_driver, "RELATED_TO")

    with neo4j_driver.session() as session:
        write_related_to_edges(session, chunks)
    count_second = _count_rel(neo4j_driver, "RELATED_TO")

    assert count_first == count_second, (
        f"RELATED_TO count changed from {count_first} to {count_second} on second run"
    )
