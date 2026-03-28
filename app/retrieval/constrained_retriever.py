"""Constrained vector retriever for Chunk nodes (T023).

Fetches the top-k most similar Chunks using Neo4j's vector index, optionally
constrained to chunks reachable from a set of graph-resolved node IDs via
ABOUT_COMPANY (Chunk→Company) or FROM_CONTRACT (Chunk→Contract) relationships.

When *allowed_ids* is non-empty the WHERE filter runs entirely inside Neo4j —
no Python-side filtering, no pulling extra chunks into memory.  The
relationship join is the correct join point: *allowed_ids* contains domain-
level Company.id / Contract.contract_id values, not Chunk IDs.

When *allowed_ids* is empty (graph resolution returned nothing), the retriever
falls back to an unconstrained vector search and sets graph_constrained=False.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from graphrag_assistant.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)

# Vector index name must match neo4j_client._DDL_CHUNK_VECTOR_INDEX
_INDEX_NAME = "chunk_embedding_idx"

# Fetch this many candidates from the vector index before applying the
# relationship filter.  3× gives the filter room to work without loading
# an unbounded result set into the session.
_CANDIDATE_MULTIPLIER = 3

# ---------------------------------------------------------------------------
# Cypher queries — no f-strings, no %-format, all dynamic values via $params
# ---------------------------------------------------------------------------

# Constrained: restrict to chunks reachable from allowed_ids via either
#   ABOUT_COMPANY (Chunk→Company, filtered on Company.id) or
#   FROM_CONTRACT  (Chunk→Contract, filtered on Contract.contract_id).
# EXISTS subqueries keep the join inside Neo4j; no chunk IDs are passed
# directly — the filter traverses the graph relationships to reach the
# allowed domain nodes.
_QUERY_CONSTRAINED = (
    "CALL db.index.vector.queryNodes('chunk_embedding_idx', $candidates_k, $query_vector) "
    "YIELD node AS chunk, score "
    "WHERE ("
    "EXISTS { MATCH (chunk)-[:ABOUT_COMPANY]->(co:Company) WHERE co.id IN $allowed_ids } "
    "OR EXISTS { "
    "MATCH (chunk)-[:FROM_CONTRACT]->(ct:Contract) WHERE ct.contract_id IN $allowed_ids "
    "}) "
    "RETURN chunk.chunk_id AS chunk_id, score "
    "ORDER BY score DESC "
    "LIMIT $top_k"
)

# Unconstrained fallback: no relationship filter.
_QUERY_UNCONSTRAINED = (
    "CALL db.index.vector.queryNodes('chunk_embedding_idx', $candidates_k, $query_vector) "
    "YIELD node AS chunk, score "
    "RETURN chunk.chunk_id AS chunk_id, score "
    "ORDER BY score DESC "
    "LIMIT $top_k"
)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


@dataclass
class ChunkResult:
    """A single chunk returned by the vector retriever."""

    chunk_id: str
    score: float


@dataclass
class ConstrainedRetrievalResult:
    """Result from retrieve_chunks, carrying retrieval_debug-compatible fields."""

    chunks: list[ChunkResult]
    graph_constrained: bool
    retrieved_chunk_ids: list[str]
    vector_query_ms: float


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def retrieve_chunks(
    query: str,
    embedding_provider: EmbeddingProvider,
    session: Any,
    allowed_ids: list[str],
    *,
    top_k: int = 5,
) -> ConstrainedRetrievalResult:
    """Fetch top_k Chunks by vector similarity, optionally filtered to graph nodes.

    When *allowed_ids* is non-empty, only chunks reachable from those node IDs
    via ABOUT_COMPANY or FROM_CONTRACT relationships are candidates.  The
    filter evaluates entirely in Neo4j — *allowed_ids* contains Company.id or
    Contract.contract_id values, never Chunk IDs.

    When *allowed_ids* is empty, unconstrained vector search runs and
    ``result.graph_constrained`` is set to ``False``.

    Args:
        query:              Raw user question text to embed and search.
        embedding_provider: Provider whose ``embed()`` is called exactly once.
        session:            Open Neo4j session (caller manages lifecycle).
        allowed_ids:        Domain-level node IDs from the graph stage
                            (Company.id or Contract.contract_id values).
                            Empty list triggers unconstrained fallback.
        top_k:              Maximum number of results to return.  Default 5.

    Returns:
        ConstrainedRetrievalResult with chunks (chunk_id + score only),
        graph_constrained flag, retrieved_chunk_ids list, and vector_query_ms.
    """
    vector = embedding_provider.embed([query])[0]
    candidates_k = top_k * _CANDIDATE_MULTIPLIER

    start = time.monotonic()
    if allowed_ids:
        rows = session.run(
            _QUERY_CONSTRAINED,
            {
                "candidates_k": candidates_k,
                "query_vector": vector,
                "allowed_ids": allowed_ids,
                "top_k": top_k,
            },
        )
        graph_constrained = True
    else:
        rows = session.run(
            _QUERY_UNCONSTRAINED,
            {
                "candidates_k": candidates_k,
                "query_vector": vector,
                "top_k": top_k,
            },
        )
        graph_constrained = False

    elapsed_ms = (time.monotonic() - start) * 1000.0

    chunks = [
        ChunkResult(chunk_id=row["chunk_id"], score=row["score"])
        for row in rows
    ]
    chunk_ids = [c.chunk_id for c in chunks]

    logger.info(
        "constrained_retriever: constrained=%s chunk_ids=%s elapsed_ms=%.1f",
        graph_constrained,
        chunk_ids,
        elapsed_ms,
    )

    return ConstrainedRetrievalResult(
        chunks=chunks,
        graph_constrained=graph_constrained,
        retrieved_chunk_ids=chunk_ids,
        vector_query_ms=elapsed_ms,
    )
