"""Tests for app.retrieval.constrained_retriever (T023).

Unit tests (15)
---------------
 1. test_constrained_cypher_contains_about_company_relationship_join
 2. test_constrained_cypher_contains_from_contract_relationship_join
 3. test_constrained_cypher_does_not_filter_by_chunk_id_directly
 4. test_constrained_cypher_uses_dollar_allowed_ids_param
 5. test_contract_ids_as_allowed_ids_triggers_constrained_path
 6. test_fallback_empty_allowed_ids_sets_graph_constrained_false
 7. test_fallback_uses_unconstrained_query_not_constrained
 8. test_default_top_k_is_five
 9. test_top_k_limits_returned_chunks
10. test_empty_session_result_returns_empty_chunks_list
11. test_retrieval_result_has_graph_constrained_field
12. test_retrieval_result_has_retrieved_chunk_ids_field
13. test_retrieval_result_has_vector_query_ms_field
14. test_no_fstring_in_module_source
15. test_no_percent_format_in_module_source

Integration tests (2) — require a reachable Neo4j instance (pytest -m integration)
------------------------------------------------------------------------------------
16. test_integration_constrained_returns_fewer_chunks_than_unconstrained
17. test_integration_all_returned_chunks_reachable_from_allowed_ids
"""
from __future__ import annotations

import inspect
import os
import re
from unittest.mock import MagicMock

import pytest

import app.retrieval.constrained_retriever as _mod
from app.retrieval.constrained_retriever import (
    _QUERY_CONSTRAINED,
    _QUERY_UNCONSTRAINED,
    ConstrainedRetrievalResult,
    retrieve_chunks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedding_provider(vector: list[float] | None = None) -> MagicMock:
    provider = MagicMock()
    provider.embed.return_value = [vector or [0.1, 0.2, 0.3]]
    return provider


def _make_session(rows: list[dict] | None = None) -> MagicMock:
    session = MagicMock()
    session.run.return_value = iter(rows or [])
    return session


def _make_row(chunk_id: str, score: float = 0.9) -> dict:
    return {"chunk_id": chunk_id, "score": score}


# ---------------------------------------------------------------------------
# Unit test 1: constrained Cypher contains ABOUT_COMPANY relationship join
# ---------------------------------------------------------------------------


def test_constrained_cypher_contains_about_company_relationship_join() -> None:
    assert "ABOUT_COMPANY" in _QUERY_CONSTRAINED
    # The join must reference a Company node property, not a raw id list
    assert "co.id IN $allowed_ids" in _QUERY_CONSTRAINED


# ---------------------------------------------------------------------------
# Unit test 2: constrained Cypher contains FROM_CONTRACT relationship join
# ---------------------------------------------------------------------------


def test_constrained_cypher_contains_from_contract_relationship_join() -> None:
    assert "FROM_CONTRACT" in _QUERY_CONSTRAINED
    # Contract node keyed on contract_id property, not a raw id list
    assert "ct.contract_id IN $allowed_ids" in _QUERY_CONSTRAINED


# ---------------------------------------------------------------------------
# Unit test 3: constrained Cypher does not filter by chunk_id directly
# ---------------------------------------------------------------------------


def test_constrained_cypher_does_not_filter_by_chunk_id_directly() -> None:
    # The WHERE clause must never compare chunk.chunk_id against allowed_ids;
    # that would bypass the relationship traversal and silently return wrong results.
    assert "chunk.chunk_id IN $allowed_ids" not in _QUERY_CONSTRAINED
    assert "chunk_id IN $allowed_ids" not in _QUERY_CONSTRAINED


# ---------------------------------------------------------------------------
# Unit test 4: constrained Cypher uses parameterised $allowed_ids
# ---------------------------------------------------------------------------


def test_constrained_cypher_uses_dollar_allowed_ids_param() -> None:
    assert "$allowed_ids" in _QUERY_CONSTRAINED
    # Unconstrained must NOT contain $allowed_ids — it has no filter
    assert "$allowed_ids" not in _QUERY_UNCONSTRAINED


# ---------------------------------------------------------------------------
# Unit test 5: passing contract node IDs still triggers constrained path
#   (contract IDs are not chunk IDs — retrieval must go through the graph)
# ---------------------------------------------------------------------------


def test_contract_ids_as_allowed_ids_triggers_constrained_path() -> None:
    provider = _make_embedding_provider()
    session = _make_session([_make_row("chunk-1")])

    result = retrieve_chunks("query", provider, session, allowed_ids=["CTR_001"])

    assert result.graph_constrained is True
    cypher_used = session.run.call_args[0][0]
    assert cypher_used is _QUERY_CONSTRAINED
    params_used = session.run.call_args[0][1]
    assert params_used["allowed_ids"] == ["CTR_001"]
    # FROM_CONTRACT join in the query confirms no direct chunk-ID filter
    assert "FROM_CONTRACT" in cypher_used


# ---------------------------------------------------------------------------
# Unit test 6: empty allowed_ids sets graph_constrained = False
# ---------------------------------------------------------------------------


def test_fallback_empty_allowed_ids_sets_graph_constrained_false() -> None:
    provider = _make_embedding_provider()
    session = _make_session([_make_row("chunk-1")])

    result = retrieve_chunks("query", provider, session, allowed_ids=[])

    assert result.graph_constrained is False


# ---------------------------------------------------------------------------
# Unit test 7: empty allowed_ids uses unconstrained query, not constrained
# ---------------------------------------------------------------------------


def test_fallback_uses_unconstrained_query_not_constrained() -> None:
    provider = _make_embedding_provider()
    session = _make_session([_make_row("chunk-1")])

    retrieve_chunks("query", provider, session, allowed_ids=[])

    cypher_used = session.run.call_args[0][0]
    assert cypher_used is _QUERY_UNCONSTRAINED
    params_used = session.run.call_args[0][1]
    assert "allowed_ids" not in params_used


# ---------------------------------------------------------------------------
# Unit test 8: default top_k is 5
# ---------------------------------------------------------------------------


def test_default_top_k_is_five() -> None:
    provider = _make_embedding_provider()
    rows = [_make_row(f"chunk-{i}", score=1.0 - i * 0.1) for i in range(10)]
    session = _make_session(rows)

    # Call without explicit top_k — mock returns 10 rows but LIMIT $top_k=5
    retrieve_chunks("query", provider, session, allowed_ids=[])

    params = session.run.call_args[0][1]
    assert params["top_k"] == 5


# ---------------------------------------------------------------------------
# Unit test 9: top_k limits returned chunks (mock returns more than top_k)
# ---------------------------------------------------------------------------


def test_top_k_limits_returned_chunks() -> None:
    provider = _make_embedding_provider()
    # Mock returns 3 rows; with top_k=2 the Cypher LIMIT param is 2
    rows = [_make_row(f"chunk-{i}") for i in range(3)]
    session = _make_session(rows)

    result = retrieve_chunks("query", provider, session, allowed_ids=[], top_k=2)

    params = session.run.call_args[0][1]
    assert params["top_k"] == 2
    # Actual chunk count equals what the mock returned (session enforces LIMIT)
    assert len(result.chunks) == 3


# ---------------------------------------------------------------------------
# Unit test 10: empty session result returns empty chunks list without error
# ---------------------------------------------------------------------------


def test_empty_session_result_returns_empty_chunks_list() -> None:
    provider = _make_embedding_provider()
    session = _make_session([])

    result = retrieve_chunks("query", provider, session, allowed_ids=["company-1"])

    assert result.chunks == []
    assert result.retrieved_chunk_ids == []
    assert result.graph_constrained is True


# ---------------------------------------------------------------------------
# Unit test 11: result carries graph_constrained field
# ---------------------------------------------------------------------------


def test_retrieval_result_has_graph_constrained_field() -> None:
    provider = _make_embedding_provider()
    session = _make_session([_make_row("chunk-1")])

    result = retrieve_chunks("query", provider, session, allowed_ids=["company-1"])

    assert isinstance(result, ConstrainedRetrievalResult)
    assert isinstance(result.graph_constrained, bool)


# ---------------------------------------------------------------------------
# Unit test 12: result carries retrieved_chunk_ids list
# ---------------------------------------------------------------------------


def test_retrieval_result_has_retrieved_chunk_ids_field() -> None:
    provider = _make_embedding_provider()
    session = _make_session([_make_row("chunk-42")])

    result = retrieve_chunks("query", provider, session, allowed_ids=["company-1"])

    assert isinstance(result.retrieved_chunk_ids, list)
    assert result.retrieved_chunk_ids == ["chunk-42"]


# ---------------------------------------------------------------------------
# Unit test 13: result carries vector_query_ms as float
# ---------------------------------------------------------------------------


def test_retrieval_result_has_vector_query_ms_field() -> None:
    provider = _make_embedding_provider()
    session = _make_session([])

    result = retrieve_chunks("query", provider, session, allowed_ids=[])

    assert isinstance(result.vector_query_ms, float)
    assert result.vector_query_ms >= 0.0


# ---------------------------------------------------------------------------
# Unit test 14: no f-string in module source
# ---------------------------------------------------------------------------


def test_no_fstring_in_module_source() -> None:
    source = inspect.getsource(_mod)
    assert not re.search(r'\bf"', source), "f-string (double-quote) found in module"
    assert not re.search(r"\bf'", source), "f-string (single-quote) found in module"


# ---------------------------------------------------------------------------
# Unit test 15: no %-format interpolation in module source
# ---------------------------------------------------------------------------


def test_no_percent_format_in_module_source() -> None:
    source = inspect.getsource(_mod)
    assert not re.search(r'["\']\s*%\s*[(\w]', source), (
        "%-format interpolation found in module"
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


# Embedding dimension must match the vector index (384 by default).
_EMBED_DIM = 384
_ZERO_VEC = [0.0] * _EMBED_DIM
_UNIT_VEC = [1.0] + [0.0] * (_EMBED_DIM - 1)


def _seed_integration_graph(driver) -> None:
    """Seed a minimal graph for constrained-retriever integration tests.

    Graph layout:
      Company {id: 'CO_CR1'} — linked to 2 chunks via ABOUT_COMPANY
      Contract {contract_id: 'CTR_CR1'} — linked to 1 chunk via FROM_CONTRACT
      3 additional chunks with NO relationship to CO_CR1 or CTR_CR1

    The vector index is created for the Chunk node with dimension 384.
    """
    with driver.session() as s:
        # Drop + recreate index with correct dimension for tests
        s.run("DROP INDEX chunk_embedding_idx IF EXISTS")
        s.run(
            "CREATE VECTOR INDEX chunk_embedding_idx IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.embedding) "
            "OPTIONS {indexConfig: {"
            "`vector.dimensions`: 384, "
            "`vector.similarityFunction`: 'cosine'"
            "}}"
        )

        # Domain nodes
        s.run("MERGE (co:Company {id: 'CO_CR1'}) SET co.name = 'ConstrainedCorp'")
        s.run(
            "MERGE (ct:Contract {contract_id: 'CTR_CR1'}) "
            "SET ct.title = 'Constrained Test Contract'"
        )

        # Chunks linked to CO_CR1 / CTR_CR1
        for i in range(2):
            s.run(
                "MERGE (ch:Chunk {chunk_id: $cid}) "
                "SET ch.text = $text, ch.embedding = $emb",
                {"cid": f"CR_linked_{i}", "text": f"Linked chunk {i}.", "emb": _UNIT_VEC},
            )
            s.run(
                "MATCH (ch:Chunk {chunk_id: $cid}), (co:Company {id: 'CO_CR1'}) "
                "MERGE (ch)-[:ABOUT_COMPANY]->(co)",
                {"cid": f"CR_linked_{i}"},
            )

        s.run(
            "MERGE (ch:Chunk {chunk_id: 'CR_linked_contract'}) "
            "SET ch.text = 'Contract chunk.', ch.embedding = $emb",
            {"emb": _UNIT_VEC},
        )
        s.run(
            "MATCH (ch:Chunk {chunk_id: 'CR_linked_contract'}), "
            "(ct:Contract {contract_id: 'CTR_CR1'}) "
            "MERGE (ch)-[:FROM_CONTRACT]->(ct)"
        )

        # Unlinked chunks — should NOT appear in constrained search
        for i in range(3):
            s.run(
                "MERGE (ch:Chunk {chunk_id: $cid}) "
                "SET ch.text = $text, ch.embedding = $emb",
                {
                    "cid": f"CR_unlinked_{i}",
                    "text": f"Unlinked chunk {i}.",
                    "emb": _UNIT_VEC,
                },
            )


def _make_mock_provider(vector: list[float]) -> MagicMock:
    provider = MagicMock()
    provider.embed.return_value = [vector]
    return provider


# ---------------------------------------------------------------------------
# Integration test 16: constrained returns fewer chunks than unconstrained
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_constrained_returns_fewer_chunks_than_unconstrained(
    neo4j_driver, clean_graph
) -> None:
    _seed_integration_graph(neo4j_driver)
    provider = _make_mock_provider(_UNIT_VEC)

    with neo4j_driver.session() as session:
        unconstrained = retrieve_chunks(
            "test query", provider, session, allowed_ids=[], top_k=10
        )

    with neo4j_driver.session() as session:
        constrained = retrieve_chunks(
            "test query", provider, session, allowed_ids=["CO_CR1"], top_k=10
        )

    assert unconstrained.graph_constrained is False
    assert constrained.graph_constrained is True
    assert len(constrained.chunks) < len(unconstrained.chunks), (
        f"Expected fewer constrained chunks ({len(constrained.chunks)}) "
        f"than unconstrained ({len(unconstrained.chunks)})"
    )


# ---------------------------------------------------------------------------
# Integration test 17: all returned chunks reachable from allowed_ids
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_all_returned_chunks_reachable_from_allowed_ids(
    neo4j_driver, clean_graph
) -> None:
    _seed_integration_graph(neo4j_driver)
    provider = _make_mock_provider(_UNIT_VEC)

    with neo4j_driver.session() as session:
        result = retrieve_chunks(
            "test query", provider, session, allowed_ids=["CO_CR1"], top_k=10
        )

    assert result.chunks, "Expected at least one chunk to be returned"

    # Verify each returned chunk is reachable via ABOUT_COMPANY or FROM_CONTRACT
    with neo4j_driver.session() as session:
        for chunk in result.chunks:
            rows = list(
                session.run(
                    "MATCH (ch:Chunk {chunk_id: $cid}) "
                    "WHERE ("
                    "EXISTS { MATCH (ch)-[:ABOUT_COMPANY]->(co:Company) WHERE co.id IN $ids } "
                    "OR EXISTS { "
                    "MATCH (ch)-[:FROM_CONTRACT]->(ct:Contract) WHERE ct.contract_id IN $ids "
                    "} "
                    ") "
                    "RETURN ch.chunk_id AS chunk_id",
                    {"cid": chunk.chunk_id, "ids": ["CO_CR1"]},
                )
            )
            assert rows, (
                f"Chunk {chunk.chunk_id!r} is not reachable from allowed_ids=['CO_CR1'] "
                "via ABOUT_COMPANY or FROM_CONTRACT"
            )
