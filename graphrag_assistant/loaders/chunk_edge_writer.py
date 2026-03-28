"""FROM_CONTRACT and ABOUT_COMPANY edge writer for Chunk nodes (T018.c).

Public API
----------
write_chunk_contract_edges(session, chunks)  Batch-create FROM_CONTRACT edges.
write_chunk_company_edges(session, chunks)   Batch-create ABOUT_COMPANY edges.

Design
------
* Both functions accept a list of chunk dicts and a Neo4j session.
* Each function issues a single UNWIND + MERGE Cypher statement.
* MERGE is on chunk_id + target-node id, so running either function twice is
  idempotent: no duplicate edges are created.
* Chunks whose contract_id field is absent or falsy are logged at WARNING and
  skipped without raising.  ABOUT_COMPANY silently skips absent company nodes
  (the OPTIONAL MATCH / WHERE IS NOT NULL pattern handles that in Cypher).
* No RELATED_TO edges are created here.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cypher statements
# ---------------------------------------------------------------------------

_MERGE_FROM_CONTRACT = (
    "UNWIND $rows AS row "
    "MATCH (ch:Chunk {chunk_id: row.chunk_id}) "
    "MATCH (ct:Contract {contract_id: row.contract_id}) "
    "MERGE (ch)-[:FROM_CONTRACT]->(ct)"
)

_MERGE_ABOUT_COMPANY = (
    "UNWIND $rows AS row "
    "MATCH (ch:Chunk {chunk_id: row.chunk_id}) "
    "OPTIONAL MATCH (co:Company {id: row.company_id}) "
    "WITH ch, co "
    "WHERE co IS NOT NULL "
    "MERGE (ch)-[:ABOUT_COMPANY]->(co)"
)

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def write_chunk_contract_edges(session: Any, chunks: list[dict]) -> None:
    """Batch-create FROM_CONTRACT edges (Chunk → Contract) using UNWIND MERGE.

    Chunks whose contract_id field is missing or falsy are logged at WARNING
    level and skipped; all others are batched into a single Cypher call.
    Running this function twice on the same chunks is idempotent.

    Args:
        session: An open Neo4j session.
        chunks:  Chunk dicts; each must have chunk_id and contract_id.
    """
    if not chunks:
        return

    rows: list[dict] = []
    for chunk in chunks:
        contract_id = chunk.get("contract_id")
        if not contract_id:
            logger.warning(
                "FROM_CONTRACT: chunk %r has no contract_id — skipping",
                chunk.get("chunk_id"),
            )
            continue
        rows.append({"chunk_id": chunk["chunk_id"], "contract_id": contract_id})

    if rows:
        session.run(_MERGE_FROM_CONTRACT, {"rows": rows})


def write_chunk_company_edges(session: Any, chunks: list[dict]) -> None:
    """Batch-create ABOUT_COMPANY edges (Chunk → Company) using UNWIND MERGE.

    Expands each chunk's company_ids list into individual rows.  Chunks with
    no company_ids produce no rows and do not trigger a Cypher call.
    Company nodes absent from the graph are silently skipped by the OPTIONAL
    MATCH in the Cypher query.  Running this function twice is idempotent.

    Args:
        session: An open Neo4j session.
        chunks:  Chunk dicts; each may have a company_ids list[str].
    """
    if not chunks:
        return

    rows = [
        {"chunk_id": c["chunk_id"], "company_id": cid}
        for c in chunks
        for cid in (c.get("company_ids") or [])
    ]

    if rows:
        session.run(_MERGE_ABOUT_COMPANY, {"rows": rows})
