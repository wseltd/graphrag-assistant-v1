"""Graph traversal pipeline module (T025.b).

traverse_from_anchors resolves domain-level node IDs to Neo4j graph nodes and
walks outbound relationships up to *max_hops* deep, collecting:
  - chunk_ids: all Chunk.chunk_id values reachable from the anchors.
  - triples:   deduplicated (src, rel, dst) relationship tuples, where src and
               dst are canonical entity names or IDs (not Neo4j internal IDs).

Cypher safety
-------------
Domain node IDs are always passed as $node_ids — never interpolated into the
Cypher string.  Hop depth selects from a module-level constant list of query
strings; no runtime string construction occurs.

Design
------
Two module-level Cypher strings cover depths 1 and 2.  The 2-hop query uses
``[*1..2]``, which is a Cypher literal — not a parameter, not user-supplied.
The 1-hop query uses a single direct MATCH for efficiency on shallow lookups.

Triple.src and Triple.dst are projected using CASE expressions that select the
canonical name property per node label rather than toString(id(...)). This
makes triples human-readable when the citation_generator outputs them verbatim.
Nodes whose labels do not match any branch fall back to toString(id(...)) so no
edge is silently dropped.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Triple:
    """A directed relationship triple collected during graph traversal.

    Attributes:
        src: Canonical name of the source node (e.g. company name, person name,
             contract_id).  Falls back to toString(id(...)) for unknown labels.
        rel: Relationship type string (e.g. ``"PARTY_TO"``).
        dst: Canonical name of the target node.  Falls back to toString(id(...))
             for unknown labels.
    """

    src: str
    rel: str
    dst: str


@dataclass
class GraphTraversalResult:
    """Output of traverse_from_anchors.

    Attributes:
        chunk_ids: Deduplicated Chunk.chunk_id values reachable within the
                   requested hop depth.
        triples:   Deduplicated (src, rel, dst) triples along all traversal
                   paths.
    """

    chunk_ids: list[str]
    triples: list[Triple]


# ---------------------------------------------------------------------------
# Cypher queries (module-level constants, no runtime interpolation)
# ---------------------------------------------------------------------------

# CASE expression shared structure (explained once here):
#   Contract → contract_id  (domain key, unique across the graph)
#   Chunk    → chunk_id     (domain key, unique across the graph)
#   Company / Person / Product → name  (human-readable canonical label)
#   fallback → toString(id(...))  so no edge is silently dropped for unknown labels
#
# Chunk is never an anchor, so the src CASE has no Chunk branch.

# 1-hop: direct outgoing neighbours of each anchor node.
# anchor is always Company/Person/Product/Contract; b can be any label.
_Q_1HOP: str = (
    "UNWIND $node_ids AS nid "
    "MATCH (anchor) WHERE anchor.id = nid OR anchor.contract_id = nid "
    "MATCH (anchor)-[r]->(b) "
    "RETURN "
    "CASE "
    "WHEN 'Contract' IN labels(anchor) THEN anchor.contract_id "
    "WHEN 'Company' IN labels(anchor) THEN anchor.name "
    "WHEN 'Person' IN labels(anchor) THEN anchor.name "
    "WHEN 'Product' IN labels(anchor) THEN anchor.name "
    "ELSE toString(id(anchor)) "
    "END AS src, "
    "type(r) AS rel, "
    "CASE "
    "WHEN 'Contract' IN labels(b) THEN b.contract_id "
    "WHEN 'Chunk' IN labels(b) THEN b.chunk_id "
    "WHEN 'Company' IN labels(b) THEN b.name "
    "WHEN 'Person' IN labels(b) THEN b.name "
    "WHEN 'Product' IN labels(b) THEN b.name "
    "ELSE toString(id(b)) "
    "END AS dst, "
    "CASE WHEN 'Chunk' IN labels(b) THEN b.chunk_id ELSE null END AS chunk_id"
)

# 2-hop: all paths of length 1 or 2 from each anchor.
# The range literal [*1..2] is embedded directly — it is a Cypher constant,
# not user input.
# Because the path is unwound into individual relationship edges, there are no
# named node variables — startNode(r) and endNode(r) must be used for the CASE.
_Q_2HOP: str = (
    "UNWIND $node_ids AS nid "
    "MATCH (anchor) WHERE anchor.id = nid OR anchor.contract_id = nid "
    "MATCH p = (anchor)-[*1..2]->(b) "
    "UNWIND relationships(p) AS r "
    "RETURN "
    "CASE "
    "WHEN 'Contract' IN labels(startNode(r)) THEN startNode(r).contract_id "
    "WHEN 'Company' IN labels(startNode(r)) THEN startNode(r).name "
    "WHEN 'Person' IN labels(startNode(r)) THEN startNode(r).name "
    "WHEN 'Product' IN labels(startNode(r)) THEN startNode(r).name "
    "ELSE toString(id(startNode(r))) "
    "END AS src, "
    "type(r) AS rel, "
    "CASE "
    "WHEN 'Contract' IN labels(endNode(r)) THEN endNode(r).contract_id "
    "WHEN 'Chunk' IN labels(endNode(r)) THEN endNode(r).chunk_id "
    "WHEN 'Company' IN labels(endNode(r)) THEN endNode(r).name "
    "WHEN 'Person' IN labels(endNode(r)) THEN endNode(r).name "
    "WHEN 'Product' IN labels(endNode(r)) THEN endNode(r).name "
    "ELSE toString(id(endNode(r))) "
    "END AS dst, "
    "CASE WHEN 'Chunk' IN labels(endNode(r)) "
    "THEN endNode(r).chunk_id ELSE null END AS chunk_id"
)

# Index: _QUERIES[0] = 1-hop, _QUERIES[1] = 2-hop.
_QUERIES: list[str] = [_Q_1HOP, _Q_2HOP]
_MAX_HOPS: int = len(_QUERIES)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def traverse_from_anchors(
    node_ids: list[str],
    session: Any,
    max_hops: int = 2,
) -> GraphTraversalResult:
    """Walk the graph from anchor nodes collecting Chunk IDs and triples.

    For each anchor node (resolved by domain ID), follows outbound edges up to
    *max_hops* and accumulates the chunk_id of every Chunk node encountered and
    a Triple for every relationship edge traversed.  Both result lists are
    deduplicated before return.

    Args:
        node_ids:  Domain-level identifiers (Company.id or
                   Contract.contract_id).  Passed via ``$node_ids`` parameter —
                   never interpolated into Cypher.
        session:   Open Neo4j session (caller manages lifecycle).
        max_hops:  Traversal depth.  Supported range: 1–2.  Values outside
                   this range are clamped.

    Returns:
        GraphTraversalResult.  Both lists are empty when *node_ids* is empty
        or no outbound graph paths exist from the anchors.
    """
    if not node_ids:
        logger.warning("traverse_from_anchors: called with empty node_ids")
        return GraphTraversalResult(chunk_ids=[], triples=[])

    hops = min(max(1, max_hops), _MAX_HOPS)
    cypher = _QUERIES[hops - 1]

    rows = session.run(cypher, {"node_ids": node_ids})

    seen_triples: set[Triple] = set()
    seen_chunk_ids: set[str] = set()
    triples: list[Triple] = []
    chunk_ids: list[str] = []

    for row in rows:
        triple = Triple(src=row["src"], rel=row["rel"], dst=row["dst"])
        if triple not in seen_triples:
            seen_triples.add(triple)
            triples.append(triple)
        cid = row["chunk_id"]
        if cid is not None and cid not in seen_chunk_ids:
            seen_chunk_ids.add(cid)
            chunk_ids.append(cid)

    logger.info(
        "traverse_from_anchors: anchors=%d hops=%d triples=%d chunks=%d",
        len(node_ids),
        hops,
        len(triples),
        len(chunk_ids),
    )
    return GraphTraversalResult(chunk_ids=chunk_ids, triples=triples)
