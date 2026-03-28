"""Cypher template library for fixed-depth graph traversal.

TEMPLATES maps (source_label, target_label) to a list of three CypherTemplate
strings: [1-hop, 2-hop, 3-hop].  Every template uses UNWIND $ids so the
caller passes a parameterised list of Neo4j internal node IDs — no user-
controlled value is ever interpolated into the query string.

Relationship directions (all arrows explicit, no wildcards):
  DIRECTOR_OF    (Person)   -[:DIRECTOR_OF]->    (Company)
  REGISTERED_AT  (Company)  -[:REGISTERED_AT]->  (Address)
  SUPPLIES       (Company)  -[:SUPPLIES]->        (Product)
  PARTY_TO       (Company)  -[:PARTY_TO]->        (Contract)
  HAS_CLAUSE     (Contract) -[:HAS_CLAUSE]->      (Clause)
  FROM_CONTRACT  (Chunk)    -[:FROM_CONTRACT]->   (Contract)
  ABOUT_COMPANY  (Chunk)    -[:ABOUT_COMPANY]->   (Company)
  RELATED_TO     (Chunk)    -[:RELATED_TO]->      (any node)

Return columns from every template
-----------------------------------
source_id     — toString(id(src))     string representation of Neo4j internal ID
source_label  — labels(src)[0]        first label of the source node
rel_type      — type(r)               relationship type string
target_id     — toString(id(tgt))     string representation of Neo4j internal ID
target_label  — labels(tgt)[0]        first label of the target node

lookup_facts
------------
Convenience wrapper: looks up the template for (source_label, target_label),
executes it at the requested hop depth, and returns a list[GraphFact].
Returns an empty list immediately when *ids* is empty — no DB call is made.
"""
from __future__ import annotations

from typing import Any

from app.retrieval.graph_traverser_types import CypherTemplate, GraphFact

# ---------------------------------------------------------------------------
# Shared RETURN clauses (string constants — no f-strings, no interpolation)
# ---------------------------------------------------------------------------

_RET_DIRECT = (
    "RETURN toString(id(src)) AS source_id, "
    "labels(src)[0] AS source_label, "
    "type(r) AS rel_type, "
    "toString(id(tgt)) AS target_id, "
    "labels(tgt)[0] AS target_label"
)

_RET_PATH = (
    "UNWIND relationships(p) AS r "
    "RETURN toString(id(startNode(r))) AS source_id, "
    "labels(startNode(r))[0] AS source_label, "
    "type(r) AS rel_type, "
    "toString(id(endNode(r))) AS target_id, "
    "labels(endNode(r))[0] AS target_label"
)

# ---------------------------------------------------------------------------
# TEMPLATES
# Key: (source_label, target_label)
# Value: [1-hop template, 2-hop template, 3-hop template]
#
# 1-hop  — direct MATCH (src)-[r:REL]->(tgt)
# 2-hop  — MATCH p = (src)-[:REL]->(m1)-[:REL]->(tgt), UNWIND relationships(p)
# 3-hop  — MATCH p = (src)-[:REL]->(m1)-[:REL]->(m2)-[:REL]->(tgt), UNWIND
#
# Fixed-depth path patterns are used for the multi-hop templates so that every
# hop is spelled out explicitly and no variable-length * patterns appear.
# ---------------------------------------------------------------------------

TEMPLATES: dict[tuple[str, str], list[CypherTemplate]] = {

    # ------------------------------------------------------------------
    # DIRECTOR_OF  (Person) -[:DIRECTOR_OF]-> (Company)
    # ------------------------------------------------------------------
    ("Person", "Company"): [
        # 1-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Person) WHERE id(src) = id "
            "MATCH (src)-[r:DIRECTOR_OF]->(tgt:Company) "
            + _RET_DIRECT
        ),
        # 2-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Person) WHERE id(src) = id "
            "MATCH p = (src)-[:DIRECTOR_OF]->(m1:Company)-[:DIRECTOR_OF]->(tgt:Company) "
            + _RET_PATH
        ),
        # 3-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Person) WHERE id(src) = id "
            "MATCH p = (src)-[:DIRECTOR_OF]->(m1:Company)"
            "-[:DIRECTOR_OF]->(m2:Company)-[:DIRECTOR_OF]->(tgt:Company) "
            + _RET_PATH
        ),
    ],

    # ------------------------------------------------------------------
    # REGISTERED_AT  (Company) -[:REGISTERED_AT]-> (Address)
    # ------------------------------------------------------------------
    ("Company", "Address"): [
        # 1-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Company) WHERE id(src) = id "
            "MATCH (src)-[r:REGISTERED_AT]->(tgt:Address) "
            + _RET_DIRECT
        ),
        # 2-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Company) WHERE id(src) = id "
            "MATCH p = (src)-[:REGISTERED_AT]->(m1:Address)-[:REGISTERED_AT]->(tgt:Address) "
            + _RET_PATH
        ),
        # 3-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Company) WHERE id(src) = id "
            "MATCH p = (src)-[:REGISTERED_AT]->(m1:Address)"
            "-[:REGISTERED_AT]->(m2:Address)-[:REGISTERED_AT]->(tgt:Address) "
            + _RET_PATH
        ),
    ],

    # ------------------------------------------------------------------
    # SUPPLIES  (Company) -[:SUPPLIES]-> (Product)
    # ------------------------------------------------------------------
    ("Company", "Product"): [
        # 1-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Company) WHERE id(src) = id "
            "MATCH (src)-[r:SUPPLIES]->(tgt:Product) "
            + _RET_DIRECT
        ),
        # 2-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Company) WHERE id(src) = id "
            "MATCH p = (src)-[:SUPPLIES]->(m1:Product)-[:SUPPLIES]->(tgt:Product) "
            + _RET_PATH
        ),
        # 3-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Company) WHERE id(src) = id "
            "MATCH p = (src)-[:SUPPLIES]->(m1:Product)"
            "-[:SUPPLIES]->(m2:Product)-[:SUPPLIES]->(tgt:Product) "
            + _RET_PATH
        ),
    ],

    # ------------------------------------------------------------------
    # PARTY_TO  (Company) -[:PARTY_TO]-> (Contract)
    # ------------------------------------------------------------------
    ("Company", "Contract"): [
        # 1-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Company) WHERE id(src) = id "
            "MATCH (src)-[r:PARTY_TO]->(tgt:Contract) "
            + _RET_DIRECT
        ),
        # 2-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Company) WHERE id(src) = id "
            "MATCH p = (src)-[:PARTY_TO]->(m1:Contract)-[:PARTY_TO]->(tgt:Contract) "
            + _RET_PATH
        ),
        # 3-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Company) WHERE id(src) = id "
            "MATCH p = (src)-[:PARTY_TO]->(m1:Contract)"
            "-[:PARTY_TO]->(m2:Contract)-[:PARTY_TO]->(tgt:Contract) "
            + _RET_PATH
        ),
    ],

    # ------------------------------------------------------------------
    # HAS_CLAUSE  (Contract) -[:HAS_CLAUSE]-> (Clause)
    # ------------------------------------------------------------------
    ("Contract", "Clause"): [
        # 1-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Contract) WHERE id(src) = id "
            "MATCH (src)-[r:HAS_CLAUSE]->(tgt:Clause) "
            + _RET_DIRECT
        ),
        # 2-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Contract) WHERE id(src) = id "
            "MATCH p = (src)-[:HAS_CLAUSE]->(m1:Clause)-[:HAS_CLAUSE]->(tgt:Clause) "
            + _RET_PATH
        ),
        # 3-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Contract) WHERE id(src) = id "
            "MATCH p = (src)-[:HAS_CLAUSE]->(m1:Clause)"
            "-[:HAS_CLAUSE]->(m2:Clause)-[:HAS_CLAUSE]->(tgt:Clause) "
            + _RET_PATH
        ),
    ],

    # ------------------------------------------------------------------
    # FROM_CONTRACT  (Chunk) -[:FROM_CONTRACT]-> (Contract)
    # ------------------------------------------------------------------
    ("Chunk", "Contract"): [
        # 1-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Chunk) WHERE id(src) = id "
            "MATCH (src)-[r:FROM_CONTRACT]->(tgt:Contract) "
            + _RET_DIRECT
        ),
        # 2-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Chunk) WHERE id(src) = id "
            "MATCH p = (src)-[:FROM_CONTRACT]->(m1:Contract)-[:FROM_CONTRACT]->(tgt:Contract) "
            + _RET_PATH
        ),
        # 3-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Chunk) WHERE id(src) = id "
            "MATCH p = (src)-[:FROM_CONTRACT]->(m1:Contract)"
            "-[:FROM_CONTRACT]->(m2:Contract)-[:FROM_CONTRACT]->(tgt:Contract) "
            + _RET_PATH
        ),
    ],

    # ------------------------------------------------------------------
    # ABOUT_COMPANY  (Chunk) -[:ABOUT_COMPANY]-> (Company)
    # ------------------------------------------------------------------
    ("Chunk", "Company"): [
        # 1-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Chunk) WHERE id(src) = id "
            "MATCH (src)-[r:ABOUT_COMPANY]->(tgt:Company) "
            + _RET_DIRECT
        ),
        # 2-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Chunk) WHERE id(src) = id "
            "MATCH p = (src)-[:ABOUT_COMPANY]->(m1:Company)-[:ABOUT_COMPANY]->(tgt:Company) "
            + _RET_PATH
        ),
        # 3-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Chunk) WHERE id(src) = id "
            "MATCH p = (src)-[:ABOUT_COMPANY]->(m1:Company)"
            "-[:ABOUT_COMPANY]->(m2:Company)-[:ABOUT_COMPANY]->(tgt:Company) "
            + _RET_PATH
        ),
    ],

    # ------------------------------------------------------------------
    # RELATED_TO  (Chunk) -[:RELATED_TO]-> (any node)
    # Target label is not constrained — RELATED_TO is polymorphic.
    # ------------------------------------------------------------------
    ("Chunk", "Entity"): [
        # 1-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Chunk) WHERE id(src) = id "
            "MATCH (src)-[r:RELATED_TO]->(tgt) "
            + _RET_DIRECT
        ),
        # 2-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Chunk) WHERE id(src) = id "
            "MATCH p = (src)-[:RELATED_TO]->()-[:RELATED_TO]->(tgt) "
            + _RET_PATH
        ),
        # 3-hop
        (
            "UNWIND $ids AS id "
            "MATCH (src:Chunk) WHERE id(src) = id "
            "MATCH p = (src)-[:RELATED_TO]->()-[:RELATED_TO]->()-[:RELATED_TO]->(tgt) "
            + _RET_PATH
        ),
    ],
}


# ---------------------------------------------------------------------------
# lookup_facts
# ---------------------------------------------------------------------------


def lookup_facts(
    session: Any,
    source_label: str,
    target_label: str,
    ids: list[int],
    *,
    hop: int = 1,
) -> list[GraphFact]:
    """Execute the template for (source_label, target_label) at *hop* depth.

    Parameters
    ----------
    session:
        An open Neo4j session exposing a ``.run(cypher, params)`` method.
    source_label:
        Label of the starting node (e.g. ``"Person"``).
    target_label:
        Label of the target node (e.g. ``"Company"``).
    ids:
        Neo4j internal node IDs for the source nodes.  When empty, returns
        ``[]`` immediately without issuing any database call.
    hop:
        Depth of traversal — 1, 2, or 3.  Defaults to 1.

    Returns
    -------
    list[GraphFact]
        One ``GraphFact`` per row returned by the Cypher template.

    Raises
    ------
    KeyError
        If ``(source_label, target_label)`` is not in ``TEMPLATES``.
    IndexError
        If ``hop`` is not in ``[1, 2, 3]``.
    """
    if not ids:
        return []
    template = TEMPLATES[(source_label, target_label)][hop - 1]
    rows = session.run(template, {"ids": ids})
    return [
        GraphFact(
            source_id=row["source_id"],
            source_label=row["source_label"],
            rel_type=row["rel_type"],
            target_id=row["target_id"],
            target_label=row["target_label"],
        )
        for row in rows
    ]
