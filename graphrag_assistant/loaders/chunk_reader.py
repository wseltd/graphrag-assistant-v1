"""Chunk JSONL reader and Neo4j loader for GraphRAG assistant (T018.a).

Public API
----------
load_chunks(path)                                 JSONL reader + field validator
load_chunks_to_neo4j(chunks, driver, embedder)    MERGE Chunk nodes and edges

Design
------
* load_chunks validates the five required fields and raises ValueError on the
  first malformed record so callers get actionable failure information.
* load_chunks_to_neo4j embeds all chunk texts in a single batch call (one
  embed() call per run, regardless of chunk count), then MERGEs Chunk nodes,
  FROM_CONTRACT edges (warn + skip on missing Contract node), ABOUT_COMPANY
  edges (silently skip missing Company nodes), and RELATED_TO edges (silently
  skip absent entities; rel_label is set from the matched node's Neo4j label).
* MERGE on chunk_id is idempotent: repeated runs update the embedding vector
  and produce the same edge counts without creating duplicates.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from neo4j import Driver

from graphrag_assistant.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {"chunk_id", "contract_id", "text", "company_ids", "related_entity_ids"}
)

# ---------------------------------------------------------------------------
# Cypher statements
# ---------------------------------------------------------------------------

# Read — find which contract_ids have matching Contract nodes.
_MATCH_CONTRACT = (
    "UNWIND $ids AS cid "
    "MATCH (c:Contract {contract_id: cid}) "
    "RETURN cid"
)

# Write — MERGE Chunk nodes; SET overwrites embedding on repeat runs.
_MERGE_CHUNKS = (
    "UNWIND $rows AS row "
    "MERGE (c:Chunk {chunk_id: row.chunk_id}) "
    "SET c.contract_id = row.contract_id, "
    "c.text = row.text, "
    "c.embedding = row.embedding"
)

# Write — MERGE FROM_CONTRACT (Chunk → Contract).
_MERGE_FROM_CONTRACT = (
    "UNWIND $rows AS row "
    "MATCH (ch:Chunk {chunk_id: row.chunk_id}) "
    "MATCH (ct:Contract {contract_id: row.contract_id}) "
    "MERGE (ch)-[:FROM_CONTRACT]->(ct)"
)

# Write — MERGE ABOUT_COMPANY (Chunk → Company); OPTIONAL MATCH silently drops
# any company_id that has no matching Company node.
_MERGE_ABOUT_COMPANY = (
    "UNWIND $rows AS row "
    "MATCH (ch:Chunk {chunk_id: row.chunk_id}) "
    "OPTIONAL MATCH (co:Company {id: row.company_id}) "
    "WITH ch, co "
    "WHERE co IS NOT NULL "
    "MERGE (ch)-[:ABOUT_COMPANY]->(co)"
)

# Write — MERGE RELATED_TO (Chunk → any node matched by id); rel_label is set
# to the first Neo4j label of the target node.  OPTIONAL MATCH silently drops
# absent entity ids.
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
class ChunkLoadResult:
    chunks_loaded: int
    from_contract_edges: int
    about_company_edges: int
    related_to_edges: int


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def load_chunks(path: str) -> list[dict]:
    """Parse *path* as newline-delimited JSON and return validated records.

    Each record must contain: chunk_id, contract_id, text, company_ids,
    related_entity_ids.  Blank lines are skipped.  The first malformed record
    raises ValueError with the line number and missing keys so the caller gets
    actionable failure information.

    Args:
        path: Filesystem path to a JSONL file.

    Returns:
        List of dicts, one per non-blank line.

    Raises:
        ValueError: On invalid JSON or missing required fields.
        OSError: If the file cannot be opened.
    """
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {lineno}: invalid JSON — {exc}") from exc
            missing = _REQUIRED_FIELDS - record.keys()
            if missing:
                raise ValueError(
                    f"line {lineno}: missing required fields {sorted(missing)!r}"
                )
            records.append(record)
    return records


def load_chunks_to_neo4j(
    chunks: list[dict],
    driver: Driver,
    embedding_provider: EmbeddingProvider,
) -> ChunkLoadResult:
    """MERGE Chunk nodes and FROM_CONTRACT / ABOUT_COMPANY / RELATED_TO edges.

    Each chunk dict must have the five fields validated by load_chunks.
    The embedding provider is injected so callers control the model and unit
    tests can pass a deterministic stub.  embed() is called exactly once per
    invocation with all chunk texts batched together.

    Chunks whose contract_id has no matching Contract node in Neo4j are logged
    at WARNING level and skipped entirely (Chunk node not created, no edges).
    ABOUT_COMPANY and RELATED_TO edges are silently skipped when the target
    node is absent; no exception is raised.

    Running this function twice on the same chunks produces the same node count
    because MERGE on chunk_id is idempotent and SET overwrites the embedding.

    Args:
        chunks:             List of validated chunk dicts.
        driver:             Live Neo4j driver.
        embedding_provider: Provider whose embed() is called once per run.

    Returns:
        ChunkLoadResult with counts of loaded chunks and each edge type.
    """
    if not chunks:
        return ChunkLoadResult(0, 0, 0, 0)

    # Embed all texts in a single batch call — one embed() per run.
    embeddings = embedding_provider.embed([c["text"] for c in chunks])
    embedding_map = {c["chunk_id"]: emb for c, emb in zip(chunks, embeddings, strict=True)}

    with driver.session() as session:
        # Identify which contract_ids have matching Contract nodes.
        contract_ids = list({c["contract_id"] for c in chunks})
        found_contracts = {
            row["cid"]
            for row in session.run(_MATCH_CONTRACT, {"ids": contract_ids})
        }

        # Separate valid chunks; warn once per chunk with a missing contract.
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
            return ChunkLoadResult(0, 0, 0, 0)

        # MERGE Chunk nodes with embeddings.
        chunk_rows = [
            {
                "chunk_id": c["chunk_id"],
                "contract_id": c["contract_id"],
                "text": c["text"],
                "embedding": embedding_map[c["chunk_id"]],
            }
            for c in valid_chunks
        ]
        session.run(_MERGE_CHUNKS, {"rows": chunk_rows})

        # MERGE FROM_CONTRACT edges.
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

        # MERGE ABOUT_COMPANY edges (one row per company_id per chunk).
        ac_rows = [
            {"chunk_id": c["chunk_id"], "company_id": cid}
            for c in valid_chunks
            for cid in (c.get("company_ids") or [])
        ]
        about_company_count = 0
        if ac_rows:
            ac_result = session.run(_MERGE_ABOUT_COMPANY, {"rows": ac_rows})
            about_company_count = ac_result.consume().counters.relationships_created

        # MERGE RELATED_TO edges (one row per entity_id per chunk).
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
        "Chunks: loaded=%d from_contract=%d about_company=%d related_to=%d",
        len(valid_chunks),
        from_contract_count,
        about_company_count,
        related_to_count,
    )
    return ChunkLoadResult(
        chunks_loaded=len(valid_chunks),
        from_contract_edges=from_contract_count,
        about_company_edges=about_company_count,
        related_to_edges=related_to_count,
    )
