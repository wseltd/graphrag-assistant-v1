"""GraphRAG orchestrator pipeline (T025.e).

Wires the four pipeline stages in order:
  resolve_entities → traverse_from_anchors → retrieve_constrained → generate_answer

No business logic lives here — only imports and sequential calls.  Each stage
owns its own logic, logging, and error handling.

Fallback behaviour
------------------
When resolve_entities returns an empty list (no recognisable entity name in the
query), node_ids is also empty.  traverse_from_anchors logs a WARNING and
returns an empty GraphTraversalResult.  retrieve_constrained receives empty
allowed_chunk_ids, logs a WARNING, and returns [].  generate_answer then
produces the standard fallback line.  The caller (router) sets mode='plain_rag'
when entity_matches is empty.
"""
from __future__ import annotations

from typing import Any

from app.pipelines.citation_generator import GenerationResult, generate_answer
from app.pipelines.constrained_retrieval import retrieve_constrained
from app.pipelines.entity_resolver import EntityMatch, resolve_entities
from app.pipelines.graph_traversal import traverse_from_anchors


def run_graph_rag(
    query: str,
    session: Any,
    vector_store: Any,
    top_k_entities: int = 5,
    max_hops: int = 2,
    top_k_chunks: int = 5,
) -> GenerationResult:
    """Run the full GraphRAG pipeline and return a GenerationResult.

    Stages (sequential, each output feeds the next):
      1. resolve_entities      — query → list[EntityMatch]
      2. traverse_from_anchors — node_ids → GraphTraversalResult
      3. retrieve_constrained  — chunk_ids → list[RankedChunk]
      4. generate_answer       — chunks + triples → GenerationResult

    When resolve_entities returns an empty list the pipeline continues with
    empty inputs through every remaining stage, producing a GenerationResult
    that contains the standard fallback line and no citations.

    Args:
        query:           Raw user question string.
        session:         Open Neo4j session used by stages 1 and 2.
        vector_store:    Object with search(query: str, top_k: int) → list[dict]
                         used by stage 3.
        top_k_entities:  Maximum entity matches to resolve.  Default 5.
        max_hops:        Traversal depth for stage 2.  Default 2.
        top_k_chunks:    Maximum chunks to retrieve in stage 3.  Default 5.

    Returns:
        GenerationResult from generate_answer.
    """
    entity_matches: list[EntityMatch] = resolve_entities(
        query, session, top_k=top_k_entities
    )
    node_ids: list[str] = [m.node_id for m in entity_matches]

    traversal = traverse_from_anchors(node_ids, session, max_hops=max_hops)

    chunks = retrieve_constrained(
        query, traversal.chunk_ids, vector_store, top_k=top_k_chunks
    )

    return generate_answer(query, chunks, traversal.triples)
