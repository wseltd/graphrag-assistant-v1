"""Constrained vector retrieval pipeline module (T025.c).

retrieve_constrained fetches the top-k most relevant Chunks from a vector
store, filtered to those whose chunk_id appears in allowed_chunk_ids.

This is the join-point between graph traversal and text retrieval in the
GraphRAG pipeline:

    graph traversal → chunk_ids → retrieve_constrained → RankedChunk list

The vector_store parameter is any object with a
    search(query: str, top_k: int) -> list[dict]
method that returns dicts with at minimum {"chunk_id", "text", "score"}.

Empty allowed_chunk_ids
-----------------------
If allowed_chunk_ids is empty the function returns [] immediately without
querying the vector store.  The caller (GraphRAG pipeline) should detect this
and fall back to plain-RAG mode.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Multiplier applied to top_k when fetching candidates from the vector store.
# Ensures the filter has enough candidates to fill top_k after filtering out
# non-adjacent chunks, without pulling the whole corpus into memory.
_CANDIDATE_MULTIPLIER = 3


@dataclass
class RankedChunk:
    """A single chunk returned by constrained vector retrieval.

    Attributes:
        chunk_id: Unique identifier matching Chunk.chunk_id in the graph.
        text:     Raw text of the chunk.
        score:    Cosine similarity score from the vector store (higher = more
                  similar to the query).  Range depends on the underlying model.
    """

    chunk_id: str
    text: str
    score: float


def retrieve_constrained(
    query: str,
    allowed_chunk_ids: list[str],
    vector_store: Any,
    top_k: int = 5,
) -> list[RankedChunk]:
    """Fetch top-k most similar Chunks, filtered to graph-adjacent chunk IDs.

    Calls vector_store.search to get a candidate pool of size
    top_k * _CANDIDATE_MULTIPLIER, then filters results to those whose
    chunk_id appears in allowed_chunk_ids, and returns the top_k
    highest-scoring results.

    Args:
        query:             Raw user question text — passed directly to the
                           vector store (which handles embedding internally).
        allowed_chunk_ids: Chunk IDs from graph traversal.  Only chunks in
                           this set are returned.  If empty, returns []
                           without querying the vector store.
        vector_store:      Object with search(query: str, top_k: int) method
                           returning list[dict] with keys chunk_id, text,
                           score.
        top_k:             Maximum number of RankedChunk results to return.
                           Default 5.

    Returns:
        List of RankedChunk sorted by descending score, length <= top_k.
        Empty list when allowed_chunk_ids is empty.
    """
    if not allowed_chunk_ids:
        logger.warning(
            "retrieve_constrained: allowed_chunk_ids is empty — returning [] "
            "without querying vector store"
        )
        return []

    allowed: frozenset[str] = frozenset(allowed_chunk_ids)
    candidates_k = top_k * _CANDIDATE_MULTIPLIER

    raw: list[dict] = vector_store.search(query, candidates_k)

    ranked = [
        RankedChunk(
            chunk_id=r["chunk_id"],
            text=r["text"],
            score=r["score"],
        )
        for r in raw
        if r["chunk_id"] in allowed
    ]

    ranked.sort(key=lambda c: c.score, reverse=True)

    result = ranked[:top_k]
    logger.info(
        "retrieve_constrained: candidates=%d filtered=%d returned=%d",
        len(raw),
        len(ranked),
        len(result),
    )
    return result
