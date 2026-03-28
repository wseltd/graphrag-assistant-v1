"""Entity router: GET /entities/{entity_type}/{entity_id} (T031).

Returns a single node with all its properties and all 1-hop edges in both
directions.  The ``entity_type`` path segment is validated against a fixed
whitelist of Neo4j labels; any unknown type returns 404 before the database
is consulted.

Edge direction
--------------
Outgoing edges  (n)-[r]->(m)  are tagged ``direction='out'``.
Incoming edges  (p)-[r]->(n)  are tagged ``direction='in'``.

Both are collected in a single Cypher statement via two sequential
OPTIONAL MATCHes separated by a WITH clause.  The intermediate WITH
prevents the Cartesian product that would arise if both patterns were
matched simultaneously (n out-edges × n in-edges rows).

The Neo4j label is injected as a string literal into the Cypher template
(not as a parameter, because Cypher does not support label parameterisation).
This is safe because the label has already been validated against
``_ALLOWED_LABELS`` — a frozenset of known-good values — before the query
is built.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.dependencies import get_neo4j_driver, require_api_key

_Neo4jDriver = Annotated[Any, Depends(get_neo4j_driver)]

logger = logging.getLogger(__name__)

router = APIRouter()


_ALLOWED_LABELS: frozenset[str] = frozenset(
    {"Company", "Person", "Address", "Product", "Contract", "Clause", "Chunk"}
)

# ------------------------------------------------------------------
# Cypher template — label placeholder filled after whitelist check.
# Double-brace {{ }} escapes literal braces in the f-string / format call.
# ------------------------------------------------------------------
_ENTITY_QUERY_TEMPLATE = """\
MATCH (n:{label} {{id: $node_id}})
OPTIONAL MATCH (n)-[r_out]->(m_out)
WITH n,
     collect(
       CASE WHEN r_out IS NOT NULL
         THEN {{rel_type: type(r_out), direction: 'out',
               neighbour_id: m_out.id, neighbour_label: labels(m_out)[0]}}
         ELSE null
       END
     ) AS out_edges
OPTIONAL MATCH (n_in)-[r_in]->(n)
WITH n, out_edges,
     collect(
       CASE WHEN r_in IS NOT NULL
         THEN {{rel_type: type(r_in), direction: 'in',
               neighbour_id: n_in.id, neighbour_label: labels(n_in)[0]}}
         ELSE null
       END
     ) AS in_edges
RETURN n,
       [e IN out_edges WHERE e IS NOT NULL] +
       [e IN in_edges  WHERE e IS NOT NULL] AS edges
"""


class EdgeEntry(BaseModel):
    rel_type: str
    direction: str  # 'out' or 'in'
    neighbour_id: str
    neighbour_label: str

    def __repr__(self) -> str:
        return super().__repr__()


class EntityResponse(BaseModel):
    id: str
    label: str
    properties: dict[str, Any]
    edges: list[EdgeEntry]

    def __repr__(self) -> str:
        return super().__repr__()


@router.get(
    "/entities/{entity_type}/{entity_id}",
    response_model=EntityResponse,
    status_code=status.HTTP_200_OK,
)
def get_entity(
    entity_type: str,
    entity_id: str,
    driver: _Neo4jDriver,
    _api_key: str = Depends(require_api_key),
) -> EntityResponse:
    """Fetch a node's properties and all 1-hop edges (both directions).

    Args:
        entity_type: Neo4j label — must be one of the whitelisted values.
        entity_id:   Value of the node's ``id`` property.
        driver:      Neo4j driver injected via :func:`app.dependencies.get_neo4j_driver`.

    Returns:
        :class:`EntityResponse` with ``id``, ``label``, ``properties``, and ``edges``.

    Raises:
        HTTP 404: ``entity_type`` is not in the whitelist, or no node with
                  the given id exists for that label.
    """
    if entity_type not in _ALLOWED_LABELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown entity type: {entity_type!r}",
        )

    cypher = _ENTITY_QUERY_TEMPLATE.format(label=entity_type)

    with driver.session() as session:
        row = next(iter(session.run(cypher, {"node_id": entity_id})), None)

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{entity_type} with id {entity_id!r} not found",
        )
    node = row["n"]
    raw_edges: list[dict[str, Any]] = row["edges"]

    # All node properties except the embedding vector — it is internal
    # infrastructure (vector search index) and not meaningful to callers.
    properties: dict[str, Any] = {
        k: v for k, v in dict(node).items() if k != "embedding"
    }

    edges = [
        EdgeEntry(
            rel_type=e["rel_type"],
            direction=e["direction"],
            neighbour_id=str(e["neighbour_id"]),
            neighbour_label=e["neighbour_label"],
        )
        for e in raw_edges
    ]

    logger.info(
        "get_entity: type=%r id=%r edges=%d",
        entity_type,
        entity_id,
        len(edges),
    )

    return EntityResponse(
        id=entity_id,
        label=entity_type,
        properties=properties,
        edges=edges,
    )
