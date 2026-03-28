"""Tests for app.ingest.pipeline.ingest_contract (T029.b).

Unit tests (no Neo4j):
  test_happy_path_returns_correct_dict
  test_happy_path_embedding_called_once_with_all_texts
  test_empty_raw_text_raises_value_error
  test_whitespace_only_raises_value_error

Integration tests (require Neo4j — marked with @pytest.mark.integration):
  TestIntegration::test_first_ingest_creates_contract_and_chunks
  TestIntegration::test_second_ingest_is_noop_all_counts_zero
  TestIntegration::test_delete_and_reingest_recreates_nodes
  TestIntegration::test_multi_contract_distinct_ids_in_neo4j

Run integration tests with:
  pytest -m integration
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from neo4j import GraphDatabase

from app.ingest.pipeline import ingest_contract

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_SHORT_TEXT = "The supplier shall deliver goods within thirty days of the purchase order."

# 520 tokens — produces two chunks at stride=448 (512-64).
_LONG_TEXT = " ".join(["word"] * 520)


# ---------------------------------------------------------------------------
# Helpers for unit test mocks
# ---------------------------------------------------------------------------


def _make_mock_result(
    nodes_created: int = 0,
    relationships_created: int = 0,
) -> MagicMock:
    result = MagicMock()
    summary = MagicMock()
    summary.counters.nodes_created = nodes_created
    summary.counters.relationships_created = relationships_created
    result.consume.return_value = summary
    return result


def _make_mock_session(
    contract_nodes: int = 1,
    chunk_nodes: int = 1,
    edges: int = 1,
) -> MagicMock:
    """Build a mock session returning three sequential results per the MERGE order."""
    session = MagicMock()
    session.run.side_effect = [
        _make_mock_result(nodes_created=contract_nodes),   # Contract MERGE
        _make_mock_result(nodes_created=chunk_nodes),      # Chunk MERGE
        _make_mock_result(relationships_created=edges),    # Edge MERGE
    ]
    return session


def _make_mock_embedder(n_vectors: int = 1, dims: int = 4) -> MagicMock:
    provider = MagicMock()
    provider.embed.return_value = [[0.1] * dims for _ in range(n_vectors)]
    return provider


# ---------------------------------------------------------------------------
# Unit tests — no Neo4j required
# ---------------------------------------------------------------------------


def test_happy_path_returns_correct_dict() -> None:
    """Single-chunk contract returns a dict with all four required keys and counts."""
    session = _make_mock_session(contract_nodes=1, chunk_nodes=1, edges=1)
    embedder = _make_mock_embedder(n_vectors=1)

    result = ingest_contract("ct_001", _SHORT_TEXT, session, embedder)

    assert result["contract_id"] == "ct_001"
    assert result["chunks_merged"] == 1
    assert result["nodes_merged"] == 2  # 1 contract + 1 chunk
    assert result["edges_merged"] == 1


def test_happy_path_embedding_called_once_with_all_texts() -> None:
    """_LONG_TEXT produces two chunks; embed() is called exactly once with both texts."""
    n_chunks = 2
    session = _make_mock_session(contract_nodes=1, chunk_nodes=n_chunks, edges=n_chunks)
    embedder = _make_mock_embedder(n_vectors=n_chunks)

    result = ingest_contract("ct_long", _LONG_TEXT, session, embedder)

    assert embedder.embed.call_count == 1
    texts_passed = embedder.embed.call_args[0][0]
    assert len(texts_passed) == n_chunks

    assert result["chunks_merged"] == n_chunks
    assert result["nodes_merged"] == 1 + n_chunks


def test_empty_raw_text_raises_value_error() -> None:
    """Empty string must raise ValueError before any session or embed call."""
    session = MagicMock()
    embedder = MagicMock()

    with pytest.raises(ValueError, match="empty"):
        ingest_contract("ct_empty", "", session, embedder)

    session.run.assert_not_called()
    embedder.embed.assert_not_called()


def test_whitespace_only_raises_value_error() -> None:
    """Whitespace-only text must raise ValueError before any session or embed call."""
    session = MagicMock()
    embedder = MagicMock()

    with pytest.raises(ValueError, match="empty"):
        ingest_contract("ct_ws", "   \n\t  ", session, embedder)

    session.run.assert_not_called()
    embedder.embed.assert_not_called()


# ---------------------------------------------------------------------------
# Integration tests — require a reachable Neo4j instance
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
    driver = GraphDatabase.driver(_neo4j_uri(), auth=_neo4j_auth())
    try:
        driver.verify_connectivity()
    except Exception as exc:
        pytest.skip(f"Neo4j not reachable: {exc}")
    yield driver
    driver.close()


class _StubEmbedder:
    """Returns 4-dim zero vectors — sufficient for MERGE without a vector index."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


def _count_nodes(driver, label: str) -> int:
    with driver.session() as s:
        row = list(s.run(f"MATCH (n:{label}) RETURN count(n) AS cnt"))[0]
        return row["cnt"]


class TestIntegration:
    """Integration tests — autouse wipe fixture scoped to this class only."""

    @pytest.fixture(autouse=True)
    def clean_graph(self, neo4j_driver):
        _wipe(neo4j_driver)
        yield
        _wipe(neo4j_driver)

    @pytest.mark.integration
    def test_first_ingest_creates_contract_and_chunks(self, neo4j_driver) -> None:
        """First call must create at least one Contract node and its Chunk nodes."""
        embedder = _StubEmbedder()
        with neo4j_driver.session() as session:
            result = ingest_contract("supply_ct_001", _SHORT_TEXT, session, embedder)

        assert result["contract_id"] == "supply_ct_001"
        assert result["nodes_merged"] > 0
        assert result["chunks_merged"] > 0
        assert result["edges_merged"] > 0
        assert _count_nodes(neo4j_driver, "Contract") == 1
        assert _count_nodes(neo4j_driver, "Chunk") == result["chunks_merged"]

    @pytest.mark.integration
    def test_second_ingest_is_noop_all_counts_zero(self, neo4j_driver) -> None:
        """Second call with identical input must report 0 for all creation counts."""
        embedder = _StubEmbedder()
        with neo4j_driver.session() as session:
            first = ingest_contract("supply_ct_002", _SHORT_TEXT, session, embedder)

        with neo4j_driver.session() as session:
            second = ingest_contract("supply_ct_002", _SHORT_TEXT, session, embedder)

        assert second["chunks_merged"] == 0
        assert second["nodes_merged"] == 0
        assert second["edges_merged"] == 0
        assert _count_nodes(neo4j_driver, "Contract") == 1
        assert _count_nodes(neo4j_driver, "Chunk") == first["chunks_merged"]

    @pytest.mark.integration
    def test_delete_and_reingest_recreates_nodes(self, neo4j_driver) -> None:
        """Third ingest after manual deletion must recreate all nodes and edges."""
        embedder = _StubEmbedder()
        contract_id = "supply_ct_003"

        with neo4j_driver.session() as session:
            first = ingest_contract(contract_id, _SHORT_TEXT, session, embedder)

        assert first["nodes_merged"] > 0

        # Manually delete all nodes for this contract
        with neo4j_driver.session() as s:
            s.run(
                "MATCH (n) WHERE n.contract_id = $cid DETACH DELETE n",
                {"cid": contract_id},
            )

        assert _count_nodes(neo4j_driver, "Contract") == 0
        assert _count_nodes(neo4j_driver, "Chunk") == 0

        # Third ingest must recreate everything
        with neo4j_driver.session() as session:
            third = ingest_contract(contract_id, _SHORT_TEXT, session, embedder)

        assert third["nodes_merged"] == first["nodes_merged"]
        assert third["edges_merged"] == first["edges_merged"]
        assert _count_nodes(neo4j_driver, "Contract") == 1
        assert _count_nodes(neo4j_driver, "Chunk") == first["chunks_merged"]

    @pytest.mark.integration
    def test_multi_contract_distinct_ids_in_neo4j(self, neo4j_driver) -> None:
        """Five distinct contract IDs produce five distinct Contract nodes in Neo4j."""
        embedder = _StubEmbedder()
        contract_ids = [f"contract_{i:03d}" for i in range(1, 6)]
        texts = [
            f"Contract {i}: the buyer agrees to pay on delivery within thirty days. " * 3
            for i in range(1, 6)
        ]

        for cid, text in zip(contract_ids, texts, strict=True):
            with neo4j_driver.session() as session:
                result = ingest_contract(cid, text, session, embedder)
            assert result["contract_id"] == cid

        assert _count_nodes(neo4j_driver, "Contract") == 5

        with neo4j_driver.session() as s:
            found = {
                row["cid"]
                for row in s.run(
                    "UNWIND $ids AS cid MATCH (c:Contract {contract_id: cid}) RETURN cid",
                    {"ids": contract_ids},
                )
            }
        assert found == set(contract_ids)


def _wipe(driver) -> None:
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
