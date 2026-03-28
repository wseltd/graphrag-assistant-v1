"""Chunk node writer: writes pre-embedded Chunk nodes to Neo4j and creates
FROM_CONTRACT, ABOUT_COMPANY, and RELATED_TO edges (T018.b).

Public API
----------
write_chunk_nodes(session, chunks)                     Low-level: writes Chunk nodes only.
write_chunk_graph(session, chunks, embedding_provider) High-level: embed + nodes + edges.

Design
------
* write_chunk_nodes accepts pre-embedded chunks (each must carry an 'embedding' key)
  and issues a single UNWIND + MERGE Cypher statement.  No edge creation here.
* write_chunk_graph batches all embed() calls into one invocation per run, checks
  which contract_ids have matching Contract nodes (warns + skips missing ones), then
  creates FROM_CONTRACT, ABOUT_COMPANY (silently skip absent companies), and RELATED_TO
  (silently skip absent entities; rel_label stored on the edge) relationships.
* MERGE on chunk_id is fully idempotent: a second run updates the embedding vector and
  produces the same edge counts without duplicates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from graphrag_assistant.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cypher statements
# ---------------------------------------------------------------------------

_MERGE_CHUNK_NODES = (
    "UNWIND $rows AS row "
    "MERGE (c:Chunk {chunk_id: row.chunk_id}) "
    "SET c.contract_id = row.contract_id, "
    "c.text = row.text, "
    "c.embedding = row.embedding"
)

_MATCH_CONTRACT = (
    "UNWIND $ids AS cid "
    "MATCH (c:Contract {contract_id: cid}) "
    "RETURN cid"
)

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

_MERGE_RELATED_TO = (
    "UNWIND $rows AS row "
    "MATCH (ch:Chunk {chunk_id: row.chunk_id}) "
    "OPTIONAL MATCH (n) WHERE n.id = row.entity_id "
    "WITH ch, n "
    "WHERE n IS NOT NULL "
    "MERGE (ch)-[r:RELATED_TO]->(n) "
    "SET r.rel_label = labels(n)[0]"
)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class ChunkWriteResult:
    chunks_written: int
    from_contract_edges: int
    about_company_edges: int
    related_to_edges: int


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def write_chunk_nodes(session: Any, chunks: list[dict]) -> None:
    """MERGE Chunk nodes from pre-embedded chunk dicts. No edge creation.

    Each dict in *chunks* must have: chunk_id, contract_id, text, embedding.
    The embedding must be a list[float] whose length matches the provider's
    output dimension.  Running this function twice on the same chunks is
    idempotent: MERGE on chunk_id and SET overwrite the embedding vector.

    Args:
        session: An open Neo4j session.
        chunks:  Pre-embedded chunk dicts, each with an 'embedding' key.
    """
    if not chunks:
        return
    rows = [
        {
            "chunk_id": c["chunk_id"],
            "contract_id": c["contract_id"],
            "text": c["text"],
            "embedding": c["embedding"],
        }
        for c in chunks
    ]
    session.run(_MERGE_CHUNK_NODES, {"rows": rows})


def write_chunk_graph(
    session: Any,
    chunks: list[dict],
    embedding_provider: EmbeddingProvider,
) -> ChunkWriteResult:
    """Embed chunk texts, write Chunk nodes, and create all relationship edges.

    Calls embedding_provider.embed() exactly once per invocation with all chunk
    texts batched together, then writes Chunk nodes via write_chunk_nodes and
    creates FROM_CONTRACT, ABOUT_COMPANY, and RELATED_TO edges.

    Chunks whose contract_id has no matching Contract node are logged at WARNING
    level and skipped entirely (no node, no edges for that chunk).  ABOUT_COMPANY
    and RELATED_TO edges silently skip absent target nodes without raising.

    Running this function twice on the same chunks is idempotent: MERGE on
    chunk_id produces no duplicates and SET overwrites the embedding vector.

    Args:
        session:            Open Neo4j session.
        chunks:             Chunk dicts without embeddings; each must have
                            chunk_id, contract_id, text, company_ids,
                            related_entity_ids.
        embedding_provider: Provider whose embed() is called once per run.

    Returns:
        ChunkWriteResult with counts of written chunks and each edge type.
    """
    if not chunks:
        return ChunkWriteResult(0, 0, 0, 0)

    # Single batched embed call — one embed() per invocation.
    embeddings = embedding_provider.embed([c["text"] for c in chunks])
    embedding_map = {
        c["chunk_id"]: emb
        for c, emb in zip(chunks, embeddings, strict=True)
    }

    # Identify which contract_ids have matching Contract nodes.
    contract_ids = list({c["contract_id"] for c in chunks})
    found_contracts = {
        row["cid"]
        for row in session.run(_MATCH_CONTRACT, {"ids": contract_ids})
    }

    # Warn and skip chunks whose contract_id has no Contract node.
    valid_chunks: list[dict] = []
    for chunk in chunks:
        if chunk["contract_id"] not in found_contracts:
            logger.warning(
                "FROM_CONTRACT: Contract %r not found for chunk %r — skipping",
                chunk["contract_id"],
                chunk["chunk_id"],
            )
        else:
            valid_chunks.append(chunk)

    if not valid_chunks:
        return ChunkWriteResult(0, 0, 0, 0)

    # Attach embeddings and write Chunk nodes (no edges in write_chunk_nodes).
    embedded = [{**c, "embedding": embedding_map[c["chunk_id"]]} for c in valid_chunks]
    write_chunk_nodes(session, embedded)

    # FROM_CONTRACT edges (Chunk → Contract).
    fc_result = session.run(
        _MERGE_FROM_CONTRACT,
        {
            "rows": [
                {"chunk_id": c["chunk_id"], "contract_id": c["contract_id"]}
                for c in valid_chunks
            ]
        },
    )
    from_contract_count = fc_result.consume().counters.relationships_created

    # ABOUT_COMPANY edges — one row per company_id per chunk.
    ac_rows = [
        {"chunk_id": c["chunk_id"], "company_id": cid}
        for c in valid_chunks
        for cid in (c.get("company_ids") or [])
    ]
    about_company_count = 0
    if ac_rows:
        ac_result = session.run(_MERGE_ABOUT_COMPANY, {"rows": ac_rows})
        about_company_count = ac_result.consume().counters.relationships_created

    # RELATED_TO edges — one row per entity_id per chunk.
    rt_rows = [
        {"chunk_id": c["chunk_id"], "entity_id": eid}
        for c in valid_chunks
        for eid in (c.get("related_entity_ids") or [])
    ]
    related_to_count = 0
    if rt_rows:
        rt_result = session.run(_MERGE_RELATED_TO, {"rows": rt_rows})
        related_to_count = rt_result.consume().counters.relationships_created

    logger.info(
        "ChunkNodeWriter: written=%d from_contract=%d about_company=%d related_to=%d",
        len(valid_chunks),
        from_contract_count,
        about_company_count,
        related_to_count,
    )
    return ChunkWriteResult(
        chunks_written=len(valid_chunks),
        from_contract_edges=from_contract_count,
        about_company_edges=about_company_count,
        related_to_edges=related_to_count,
    )
