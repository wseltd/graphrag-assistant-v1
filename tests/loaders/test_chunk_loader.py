"""Tests for graphrag_assistant.loaders.chunk_loader (T018.e).

Unit tests (12)
---------------
 1. test_embed_fn_called_exactly_once_single_chunk
 2. test_embed_fn_called_once_for_multiple_chunks
 3. test_embed_fn_receives_all_texts_in_order
 4. test_embeddings_attached_to_write_chunk_nodes_call
 5. test_no_company_ids_chunk_still_writes_node
 6. test_multiple_companies_per_chunk_forwarded_to_company_writer
 7. test_mixed_related_entity_types_forwarded_to_related_writer
 8. test_missing_contract_id_logs_warning_via_contract_edge_writer
 9. test_call_order_nodes_before_edges
10. test_empty_file_makes_no_writer_calls
11. test_idempotency_same_writer_calls_on_second_run
12. test_writer_functions_each_called_exactly_once

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
from unittest.mock import MagicMock, patch

import pytest

from graphrag_assistant.loaders.chunk_loader import load_chunks_to_graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHUNK_SIMPLE = {
    "chunk_id": "K001_0",
    "contract_id": "K001",
    "text": "Indemnity clause.",
    "company_ids": ["C001"],
    "related_entity_ids": ["P001"],
}

_CHUNK_NO_COMPANY = {
    "chunk_id": "K001_1",
    "contract_id": "K001",
    "text": "Governing law clause.",
    "company_ids": [],
    "related_entity_ids": ["P002"],
}

_CHUNK_MULTI_COMPANY = {
    "chunk_id": "K001_2",
    "contract_id": "K001",
    "text": "Multi-party clause.",
    "company_ids": ["C001", "C002", "C003"],
    "related_entity_ids": [],
}

_CHUNK_MIXED_RELATED = {
    "chunk_id": "K002_0",
    "contract_id": "K002",
    "text": "Cross-entity clause.",
    "company_ids": ["C001"],
    "related_entity_ids": ["C001", "P001", "PR001"],
}

_CHUNK_NO_CONTRACT = {
    "chunk_id": "K999_0",
    "contract_id": "",
    "text": "Orphaned chunk.",
    "company_ids": [],
    "related_entity_ids": [],
}


def _embed_stub(texts: list[str]) -> list[list[float]]:
    """Return a deterministic 4-float embedding per text."""
    return [[float(i), 0.2, 0.3, 0.4] for i, _ in enumerate(texts)]


def _write_jsonl(tmp_path, chunks: list[dict], filename: str = "chunks.jsonl") -> str:
    p = tmp_path / filename
    p.write_text("\n".join(json.dumps(c) for c in chunks), encoding="utf-8")
    return str(p)


def _make_session() -> MagicMock:
    session = MagicMock()
    session.run.return_value = MagicMock()
    return session


# Module paths for patching the writers where they are imported in chunk_loader.
_MOD = "graphrag_assistant.loaders.chunk_loader"


# ---------------------------------------------------------------------------
# Unit test 1: embed_fn called exactly once for a single chunk
# ---------------------------------------------------------------------------


def test_embed_fn_called_exactly_once_single_chunk(tmp_path) -> None:
    path = _write_jsonl(tmp_path, [_CHUNK_SIMPLE])
    embed_calls: list[list[str]] = []

    def counting_embed(texts: list[str]) -> list[list[float]]:
        embed_calls.append(texts)
        return _embed_stub(texts)

    with patch(f"{_MOD}.write_chunk_nodes"), \
         patch(f"{_MOD}.write_chunk_contract_edges"), \
         patch(f"{_MOD}.write_chunk_company_edges"), \
         patch(f"{_MOD}.write_related_to_edges"):
        load_chunks_to_graph(path, counting_embed, _make_session())

    assert len(embed_calls) == 1


# ---------------------------------------------------------------------------
# Unit test 2: embed_fn called exactly once for multiple chunks
# ---------------------------------------------------------------------------


def test_embed_fn_called_once_for_multiple_chunks(tmp_path) -> None:
    chunks = [_CHUNK_SIMPLE, _CHUNK_NO_COMPANY, _CHUNK_MULTI_COMPANY]
    path = _write_jsonl(tmp_path, chunks)
    embed_calls: list[list[str]] = []

    def counting_embed(texts: list[str]) -> list[list[float]]:
        embed_calls.append(texts)
        return _embed_stub(texts)

    with patch(f"{_MOD}.write_chunk_nodes"), \
         patch(f"{_MOD}.write_chunk_contract_edges"), \
         patch(f"{_MOD}.write_chunk_company_edges"), \
         patch(f"{_MOD}.write_related_to_edges"):
        load_chunks_to_graph(path, counting_embed, _make_session())

    assert len(embed_calls) == 1
    assert len(embed_calls[0]) == 3


# ---------------------------------------------------------------------------
# Unit test 3: embed_fn receives all texts in file order
# ---------------------------------------------------------------------------


def test_embed_fn_receives_all_texts_in_order(tmp_path) -> None:
    chunks = [_CHUNK_SIMPLE, _CHUNK_NO_COMPANY]
    path = _write_jsonl(tmp_path, chunks)
    received_texts: list[list[str]] = []

    def capturing_embed(texts: list[str]) -> list[list[float]]:
        received_texts.append(list(texts))
        return _embed_stub(texts)

    with patch(f"{_MOD}.write_chunk_nodes"), \
         patch(f"{_MOD}.write_chunk_contract_edges"), \
         patch(f"{_MOD}.write_chunk_company_edges"), \
         patch(f"{_MOD}.write_related_to_edges"):
        load_chunks_to_graph(path, capturing_embed, _make_session())

    assert received_texts[0] == [_CHUNK_SIMPLE["text"], _CHUNK_NO_COMPANY["text"]]


# ---------------------------------------------------------------------------
# Unit test 4: embeddings are attached to the chunks passed to write_chunk_nodes
# ---------------------------------------------------------------------------


def test_embeddings_attached_to_write_chunk_nodes_call(tmp_path) -> None:
    path = _write_jsonl(tmp_path, [_CHUNK_SIMPLE])

    with patch(f"{_MOD}.write_chunk_nodes") as mock_nodes, \
         patch(f"{_MOD}.write_chunk_contract_edges"), \
         patch(f"{_MOD}.write_chunk_company_edges"), \
         patch(f"{_MOD}.write_related_to_edges"):
        load_chunks_to_graph(path, _embed_stub, _make_session())

    _, chunks_arg = mock_nodes.call_args.args
    assert len(chunks_arg) == 1
    assert "embedding" in chunks_arg[0]
    assert chunks_arg[0]["embedding"] == [0.0, 0.2, 0.3, 0.4]
    assert chunks_arg[0]["chunk_id"] == "K001_0"


# ---------------------------------------------------------------------------
# Unit test 5: chunk with no company_ids still writes node
# ---------------------------------------------------------------------------


def test_no_company_ids_chunk_still_writes_node(tmp_path) -> None:
    path = _write_jsonl(tmp_path, [_CHUNK_NO_COMPANY])

    with patch(f"{_MOD}.write_chunk_nodes") as mock_nodes, \
         patch(f"{_MOD}.write_chunk_contract_edges"), \
         patch(f"{_MOD}.write_chunk_company_edges") as mock_company, \
         patch(f"{_MOD}.write_related_to_edges"):
        load_chunks_to_graph(path, _embed_stub, _make_session())

    # Node writer must be called even when company_ids is empty.
    mock_nodes.assert_called_once()
    _, chunks_arg = mock_nodes.call_args.args
    assert chunks_arg[0]["company_ids"] == []

    # Company edge writer is still called (it handles the empty list internally).
    mock_company.assert_called_once()


# ---------------------------------------------------------------------------
# Unit test 6: chunk with multiple companies forwarded to company edge writer
# ---------------------------------------------------------------------------


def test_multiple_companies_per_chunk_forwarded_to_company_writer(tmp_path) -> None:
    path = _write_jsonl(tmp_path, [_CHUNK_MULTI_COMPANY])

    with patch(f"{_MOD}.write_chunk_nodes"), \
         patch(f"{_MOD}.write_chunk_contract_edges"), \
         patch(f"{_MOD}.write_chunk_company_edges") as mock_company, \
         patch(f"{_MOD}.write_related_to_edges"):
        load_chunks_to_graph(path, _embed_stub, _make_session())

    _, chunks_arg = mock_company.call_args.args
    assert chunks_arg[0]["company_ids"] == ["C001", "C002", "C003"]


# ---------------------------------------------------------------------------
# Unit test 7: chunk with mixed related entity types forwarded to related writer
# ---------------------------------------------------------------------------


def test_mixed_related_entity_types_forwarded_to_related_writer(tmp_path) -> None:
    path = _write_jsonl(tmp_path, [_CHUNK_MIXED_RELATED])

    with patch(f"{_MOD}.write_chunk_nodes"), \
         patch(f"{_MOD}.write_chunk_contract_edges"), \
         patch(f"{_MOD}.write_chunk_company_edges"), \
         patch(f"{_MOD}.write_related_to_edges") as mock_related:
        load_chunks_to_graph(path, _embed_stub, _make_session())

    _, chunks_arg = mock_related.call_args.args
    assert chunks_arg[0]["related_entity_ids"] == ["C001", "P001", "PR001"]


# ---------------------------------------------------------------------------
# Unit test 8: chunk with falsy contract_id reaches contract edge writer (warns)
# ---------------------------------------------------------------------------


def test_missing_contract_id_logs_warning_via_contract_edge_writer(
    tmp_path, caplog
) -> None:
    path = _write_jsonl(tmp_path, [_CHUNK_NO_CONTRACT])

    # Use real contract-edge writer so the warning path actually fires.
    with patch(f"{_MOD}.write_chunk_nodes"), \
         patch(f"{_MOD}.write_chunk_company_edges"), \
         patch(f"{_MOD}.write_related_to_edges"):
        with caplog.at_level(logging.WARNING):
            load_chunks_to_graph(path, _embed_stub, _make_session())

    warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("K999_0" in m for m in warning_msgs), (
        f"Expected a WARNING for chunk K999_0, got: {warning_msgs!r}"
    )


# ---------------------------------------------------------------------------
# Unit test 9: write_chunk_nodes is called before all edge writers
# ---------------------------------------------------------------------------


def test_call_order_nodes_before_edges(tmp_path) -> None:
    path = _write_jsonl(tmp_path, [_CHUNK_SIMPLE])
    call_log: list[str] = []

    def recording(name):
        def inner(*_a, **_kw):
            call_log.append(name)
        return inner

    with patch(f"{_MOD}.write_chunk_nodes", side_effect=recording("nodes")), \
         patch(f"{_MOD}.write_chunk_contract_edges", side_effect=recording("contract")), \
         patch(f"{_MOD}.write_chunk_company_edges", side_effect=recording("company")), \
         patch(f"{_MOD}.write_related_to_edges", side_effect=recording("related")):
        load_chunks_to_graph(path, _embed_stub, _make_session())

    assert call_log == ["nodes", "contract", "company", "related"], (
        f"Unexpected call order: {call_log!r}"
    )


# ---------------------------------------------------------------------------
# Unit test 10: empty JSONL file makes no writer calls
# ---------------------------------------------------------------------------


def test_empty_file_makes_no_writer_calls(tmp_path) -> None:
    path = str(tmp_path / "empty.jsonl")
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    embed_calls: list = []

    with patch(f"{_MOD}.write_chunk_nodes") as mock_nodes, \
         patch(f"{_MOD}.write_chunk_contract_edges") as mock_contract, \
         patch(f"{_MOD}.write_chunk_company_edges") as mock_company, \
         patch(f"{_MOD}.write_related_to_edges") as mock_related:
        load_chunks_to_graph(path, lambda t: (embed_calls.append(t) or []), _make_session())

    assert embed_calls == [], "embed_fn must not be called for an empty file"
    mock_nodes.assert_not_called()
    mock_contract.assert_not_called()
    mock_company.assert_not_called()
    mock_related.assert_not_called()


# ---------------------------------------------------------------------------
# Unit test 11: idempotency — second run produces identical call structure
# ---------------------------------------------------------------------------


def test_idempotency_same_writer_calls_on_second_run(tmp_path) -> None:
    chunks = [_CHUNK_SIMPLE, _CHUNK_NO_COMPANY]
    path = _write_jsonl(tmp_path, chunks)
    session = _make_session()

    first_args: list[Any] = []
    second_args: list[Any] = []

    def capture_first(*args, **kwargs):
        first_args.append(args[1])  # chunks argument

    def capture_second(*args, **kwargs):
        second_args.append(args[1])

    with patch(f"{_MOD}.write_chunk_nodes", side_effect=capture_first), \
         patch(f"{_MOD}.write_chunk_contract_edges"), \
         patch(f"{_MOD}.write_chunk_company_edges"), \
         patch(f"{_MOD}.write_related_to_edges"):
        load_chunks_to_graph(path, _embed_stub, session)

    with patch(f"{_MOD}.write_chunk_nodes", side_effect=capture_second), \
         patch(f"{_MOD}.write_chunk_contract_edges"), \
         patch(f"{_MOD}.write_chunk_company_edges"), \
         patch(f"{_MOD}.write_related_to_edges"):
        load_chunks_to_graph(path, _embed_stub, session)

    # Both runs pass the same chunk_ids and embeddings to write_chunk_nodes.
    first_ids = [c["chunk_id"] for c in first_args[0]]
    second_ids = [c["chunk_id"] for c in second_args[0]]
    assert first_ids == second_ids

    first_embs = [c["embedding"] for c in first_args[0]]
    second_embs = [c["embedding"] for c in second_args[0]]
    assert first_embs == second_embs


# ---------------------------------------------------------------------------
# Unit test 12: all four writer functions called exactly once per run
# ---------------------------------------------------------------------------


def test_writer_functions_each_called_exactly_once(tmp_path) -> None:
    path = _write_jsonl(tmp_path, [_CHUNK_SIMPLE, _CHUNK_MIXED_RELATED])

    with patch(f"{_MOD}.write_chunk_nodes") as mock_nodes, \
         patch(f"{_MOD}.write_chunk_contract_edges") as mock_contract, \
         patch(f"{_MOD}.write_chunk_company_edges") as mock_company, \
         patch(f"{_MOD}.write_related_to_edges") as mock_related:
        load_chunks_to_graph(path, _embed_stub, _make_session())

    assert mock_nodes.call_count == 1, (
        f"write_chunk_nodes called {mock_nodes.call_count} times"
    )
    assert mock_contract.call_count == 1, (
        f"write_chunk_contract_edges called {mock_contract.call_count} times"
    )
    assert mock_company.call_count == 1, (
        f"write_chunk_company_edges called {mock_company.call_count} times"
    )
    assert mock_related.call_count == 1, (
        f"write_related_to_edges called {mock_related.call_count} times"
    )


# ---------------------------------------------------------------------------
# Integration tests — require a reachable Neo4j instance (pytest -m integration)
# ---------------------------------------------------------------------------


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


@pytest.fixture()
def clean_graph(neo4j_driver):
    _delete_all(neo4j_driver)
    yield
    _delete_all(neo4j_driver)


def _delete_all(driver) -> None:
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")


def _count_nodes(driver, label: str) -> int:
    with driver.session() as s:
        rows = list(s.run(f"MATCH (n:{label}) RETURN count(n) AS cnt"))
        return rows[0]["cnt"]


def _count_rel(driver, rel_type: str) -> int:
    with driver.session() as s:
        rows = list(s.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS cnt"))
        return rows[0]["cnt"]


def _seed_graph_and_chunks(driver, tmp_path):
    """Seed minimal Contract and Company nodes; return path to matching chunks.jsonl."""
    with driver.session() as s:
        s.run(
            "UNWIND $rows AS row "
            "MERGE (c:Contract {contract_id: row.contract_id})",
            {"rows": [{"contract_id": "K001"}, {"contract_id": "K002"}]},
        )
        s.run(
            "UNWIND $rows AS row "
            "MERGE (c:Company {id: row.id})",
            {"rows": [{"id": "C001"}, {"id": "C002"}]},
        )

    chunks = [
        {
            "chunk_id": "K001_0",
            "contract_id": "K001",
            "text": "Indemnity clause text.",
            "company_ids": ["C001"],
            "related_entity_ids": [],
        },
        {
            "chunk_id": "K001_1",
            "contract_id": "K001",
            "text": "Governing law clause text.",
            "company_ids": ["C001", "C002"],
            "related_entity_ids": [],
        },
        {
            "chunk_id": "K002_0",
            "contract_id": "K002",
            "text": "Payment terms clause.",
            "company_ids": [],
            "related_entity_ids": [],
        },
    ]
    path = str(tmp_path / "chunks.jsonl")
    (tmp_path / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c) for c in chunks), encoding="utf-8"
    )
    return path, len(chunks)


@pytest.mark.integration
def test_integration_from_contract_count_equals_chunk_count(
    neo4j_driver, clean_graph, tmp_path
) -> None:
    """FROM_CONTRACT edge count must equal the number of chunks in the file."""
    path, chunk_count = _seed_graph_and_chunks(neo4j_driver, tmp_path)

    with neo4j_driver.session() as session:
        load_chunks_to_graph(path, _embed_stub, session)

    chunk_node_count = _count_nodes(neo4j_driver, "Chunk")
    assert chunk_node_count == chunk_count, (
        f"Expected {chunk_count} Chunk nodes, got {chunk_node_count}"
    )

    from_contract_count = _count_rel(neo4j_driver, "FROM_CONTRACT")
    assert from_contract_count == chunk_count, (
        f"Expected {chunk_count} FROM_CONTRACT edges, got {from_contract_count}"
    )


@pytest.mark.integration
def test_integration_at_least_one_about_company_edge_exists(
    neo4j_driver, clean_graph, tmp_path
) -> None:
    """At least one ABOUT_COMPANY edge must exist after loading chunks with company_ids."""
    path, _ = _seed_graph_and_chunks(neo4j_driver, tmp_path)

    with neo4j_driver.session() as session:
        load_chunks_to_graph(path, _embed_stub, session)

    about_company_count = _count_rel(neo4j_driver, "ABOUT_COMPANY")
    assert about_company_count >= 1, (
        f"Expected at least 1 ABOUT_COMPANY edge, got {about_company_count}"
    )
