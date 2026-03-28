"""Batched graph reset helper (T019.a).

Public API
----------
reset_graph(driver)
    Delete all nodes and relationships from the Neo4j graph using a
    batched CALL { ... } IN TRANSACTIONS approach.  Safe to call on an
    empty graph (exits after one no-op iteration).

Design
------
* The Cypher uses CALL { MATCH (n) WITH n LIMIT 1000 DETACH DELETE n }
  IN TRANSACTIONS OF 1000 ROWS so each outer call deletes at most 1000
  nodes in sub-transactions, avoiding memory and lock pressure on large
  graphs.
* The outer Python loop repeats until the server reports zero nodes
  deleted, guaranteeing the graph is empty on return.
* Each loop iteration opens and closes its own session so that driver
  errors (network drop, auth failure) surface immediately and propagate
  to the caller without suppression.
"""
from __future__ import annotations

import logging

import neo4j

logger = logging.getLogger(__name__)

_BATCH_DELETE_CYPHER = (
    "CALL { MATCH (n) WITH n LIMIT 1000 DETACH DELETE n }"
    " IN TRANSACTIONS OF 1000 ROWS"
)


def reset_graph(driver: neo4j.Driver) -> None:
    """Delete all nodes and relationships from the Neo4j graph.

    Issues CALL { MATCH (n) WITH n LIMIT 1000 DETACH DELETE n } IN
    TRANSACTIONS OF 1000 ROWS in a loop until no nodes remain.  Each
    iteration opens its own session (implicit autocommit transaction),
    which is required by the IN TRANSACTIONS clause.

    Returns only when the graph is empty.  Raises on any driver error.

    Args:
        driver: An open and reachable Neo4j driver instance.
    """
    iterations = 0
    while True:
        with driver.session() as session:
            result = session.run(_BATCH_DELETE_CYPHER)
            summary = result.consume()
            deleted = summary.counters.nodes_deleted
        iterations += 1
        logger.debug(
            "reset_graph: iteration %d deleted %d node(s)", iterations, deleted
        )
        if deleted == 0:
            break
    logger.info("reset_graph: graph empty after %d iteration(s)", iterations)
