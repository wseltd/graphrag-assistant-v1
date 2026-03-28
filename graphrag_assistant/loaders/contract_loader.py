"""Contract and Clause node loader for GraphRAG assistant (T017.a).

Public API
----------
load_contracts(tx, rows)            MERGE Contract nodes; no edges.
load_clauses(tx, rows)              MERGE Clause nodes; no edges.
load_has_clause_edges(tx, rows)     MERGE HAS_CLAUSE Contract→Clause edges.
load_party_to_edges(session, rows)  MERGE PARTY_TO edges; warns on miss, never raises.

Design
------
* load_contracts / load_clauses / load_has_clause_edges each accept a Neo4j
  transaction (tx) so they compose inside a single write transaction.
* load_party_to_edges accepts a session because it needs a read phase before
  writing: for each party_id we MATCH first; absent nodes produce a WARNING
  log entry and are silently skipped (all-or-nothing per edge would hide
  legitimate partial matches in multi-party contracts).
* MERGE on natural key guarantees idempotency; a second call produces
  identical node/edge counts.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cypher statements
# ---------------------------------------------------------------------------

_MERGE_CONTRACTS = (
    "UNWIND $rows AS row "
    "MERGE (c:Contract {contract_id: row.contract_id}) "
    "SET c.title = row.title, "
    "c.effective_date = row.effective_date, "
    "c.expiry_date = row.expiry_date, "
    "c.status = row.status, "
    "c.value_usd = toFloat(row.value_usd)"
)

_MERGE_CLAUSES = (
    "UNWIND $rows AS row "
    "MERGE (cl:Clause {clause_id: row.clause_id}) "
    "SET cl.contract_id = row.contract_id, "
    "cl.clause_type = row.clause_type, "
    "cl.clause_order = toInteger(row.clause_order), "
    "cl.text = row.text"
)

_MERGE_HAS_CLAUSE = (
    "UNWIND $rows AS row "
    "MATCH (c:Contract {contract_id: row.contract_id}) "
    "MATCH (cl:Clause {clause_id: row.clause_id}) "
    "MERGE (c)-[r:HAS_CLAUSE]->(cl) "
    "ON CREATE SET r.clause_order = toInteger(row.clause_order)"
)

_MATCH_PARTY = (
    "MATCH (n) "
    "WHERE (n:Company OR n:Person) AND n.id = $party_id "
    "RETURN n.id AS id"
)

_MERGE_PARTY_TO = (
    "MATCH (n) "
    "WHERE (n:Company OR n:Person) AND n.id = $party_id "
    "MATCH (c:Contract {contract_id: $contract_id}) "
    "MERGE (n)-[r:PARTY_TO]->(c)"
)

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def load_contracts(tx: Any, rows: list[dict]) -> None:
    """MERGE Contract nodes from *rows*.

    Each dict must contain: contract_id, title, effective_date, expiry_date,
    status, value_usd.  Exits immediately if *rows* is empty.  No edges are
    created.
    """
    if not rows:
        return
    tx.run(_MERGE_CONTRACTS, {"rows": rows})


def load_clauses(tx: Any, rows: list[dict]) -> None:
    """MERGE Clause nodes from *rows*.

    Each dict must contain: clause_id, contract_id, clause_type,
    clause_order, text.  Exits immediately if *rows* is empty.  No edges are
    created.
    """
    if not rows:
        return
    tx.run(_MERGE_CLAUSES, {"rows": rows})


def load_has_clause_edges(tx: Any, rows: list[dict]) -> None:
    """MERGE HAS_CLAUSE edges from Contract to Clause.

    Each dict must contain: contract_id, clause_id, clause_order.
    clause_order is stored as an integer property on the relationship.
    Exits immediately if *rows* is empty.
    """
    if not rows:
        return
    tx.run(_MERGE_HAS_CLAUSE, {"rows": rows})


def load_party_to_edges(session: Any, contract_rows: list[dict]) -> int:
    """MERGE PARTY_TO edges for all parties listed in *contract_rows*.

    Each row may supply a pipe-separated ``party_ids`` field.  For each
    party_id a MATCH is run against Company and Person nodes.  If no
    matching node is found, a WARNING is emitted and that edge is skipped;
    no exception is raised.

    Returns the number of PARTY_TO edges written (created or already existing).
    """
    edges_written = 0
    for row in contract_rows:
        contract_id = row.get("contract_id", "")
        raw = row.get("party_ids", "")
        party_ids = [p.strip() for p in raw.split("|") if p.strip()]
        for party_id in party_ids:
            result = session.run(_MATCH_PARTY, {"party_id": party_id})
            if not list(result):
                logger.warning(
                    "PARTY_TO: party_id %r not found for contract %r — skipping",
                    party_id,
                    contract_id,
                )
                continue
            session.run(
                _MERGE_PARTY_TO,
                {"party_id": party_id, "contract_id": contract_id},
            )
            edges_written += 1
    return edges_written
