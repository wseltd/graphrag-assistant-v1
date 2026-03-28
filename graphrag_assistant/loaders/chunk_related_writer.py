"""RELATED_TO polymorphic edge writer for Chunk nodes (T018.d).

Public API
----------
write_related_to_edges(session, chunks)  Batch-create RELATED_TO edges, one UNWIND per entity label.

Design
------
* Collects all (chunk_id, entity_id) pairs from chunks' related_entity_ids lists.
* Issues one UNWIND + OPTIONAL MATCH + MERGE Cypher call per supported entity label
  (Company, Person, Product, Contract) — no apoc dependency.
* Each label-specific query uses OPTIONAL MATCH (n:Label {id: ...}) so entity_ids that
  do not match that label produce no rows and no edges; other labels are handled by their
  respective queries.
* rel_label property on the edge is set to the entity label string via a Cypher param.
* MERGE on (Chunk)-[:RELATED_TO]->(target) is idempotent: running twice produces no
  duplicate edges.
* Chunks with no related_entity_ids (key absent, None, or empty list) produce no rows
  and trigger no Cypher calls.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Supported entity labels for polymorphic RELATED_TO edges.
_ENTITY_LABELS: tuple[str, ...] = ("Company", "Person", "Product", "Contract")


def _related_to_cypher(label: str) -> str:
    """Return the MERGE RELATED_TO Cypher statement for a specific entity label."""
    return (
        "UNWIND $rows AS row "
        f"MATCH (ch:Chunk {{chunk_id: row.chunk_id}}) "
        f"OPTIONAL MATCH (n:{label} {{id: row.entity_id}}) "
        "WITH ch, n "
        "WHERE n IS NOT NULL "
        "MERGE (ch)-[r:RELATED_TO]->(n) "
        "SET r.rel_label = $label"
    )


def write_related_to_edges(session: Any, chunks: list[dict]) -> None:
    """Batch-create RELATED_TO edges from Chunk nodes to entity nodes.

    Groups all (chunk_id, entity_id) pairs from chunks and issues one UNWIND +
    MERGE Cypher call per supported entity label (Company, Person, Product,
    Contract).  Each label-specific query uses OPTIONAL MATCH so entity_ids
    that do not resolve to that label are silently skipped.  Running this
    function twice on the same chunks is idempotent: MERGE produces no
    duplicate edges.

    Args:
        session: An open Neo4j session.
        chunks:  Chunk dicts; each may have a related_entity_ids list[str].
    """
    if not chunks:
        return

    rows = [
        {"chunk_id": c["chunk_id"], "entity_id": eid}
        for c in chunks
        for eid in (c.get("related_entity_ids") or [])
    ]

    if not rows:
        return

    for label in _ENTITY_LABELS:
        session.run(_related_to_cypher(label), {"rows": rows, "label": label})
