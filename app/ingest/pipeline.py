"""Ingest pipeline orchestrator: raw contract text → Neo4j graph (T029.b).

Public API
----------
ingest_contract(contract_id, raw_text, neo4j_session, embedding_provider) -> dict

Design
------
* Splits raw_text into fixed-size token windows (512 tokens, 64-token overlap).
* Assigns deterministic SHA-256 chunk IDs via make_chunk_id(contract_id, chunk_index).
* Embeds all chunk texts in a single batch call per invocation.
* Writes to Neo4j using MERGE keyed on chunk_id (Chunk) and contract_id (Contract)
  so repeated calls are fully idempotent.
* nodes_merged and edges_merged reflect actual nodes_created / relationships_created
  from Neo4j transaction summaries — both are 0 on repeat calls.
"""
from __future__ import annotations

import logging
from typing import Any

from app.ingest.contract_ids import make_chunk_id
from graphrag_assistant.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)

_CHUNK_SIZE: int = 512
_OVERLAP: int = 64

_MERGE_CONTRACT: str = "MERGE (c:Contract {contract_id: $contract_id}) RETURN c"

_MERGE_CHUNKS: str = (
    "UNWIND $rows AS row "
    "MERGE (ch:Chunk {chunk_id: row.chunk_id}) "
    "SET ch.contract_id = row.contract_id, "
    "    ch.text = row.text, "
    "    ch.embedding = row.embedding, "
    "    ch.chunk_index = row.chunk_index"
)

_MERGE_FROM_CONTRACT: str = (
    "UNWIND $rows AS row "
    "MATCH (ch:Chunk {chunk_id: row.chunk_id}) "
    "MATCH (ct:Contract {contract_id: row.contract_id}) "
    "MERGE (ch)-[:FROM_CONTRACT]->(ct)"
)


def _split_text(raw_text: str, contract_id: str) -> list[dict]:
    """Split *raw_text* into overlapping token-window chunks with deterministic IDs.

    Uses whitespace tokenisation and stride = max(1, chunk_size - overlap).
    Each chunk receives a SHA-256 chunk_id via make_chunk_id(contract_id, index).
    """
    tokens = raw_text.split()
    stride = max(1, _CHUNK_SIZE - _OVERLAP)
    chunks: list[dict] = []
    i = 0
    while i < len(tokens):
        window = tokens[i : i + _CHUNK_SIZE]
        chunk_index = len(chunks)
        chunks.append(
            {
                "chunk_id": make_chunk_id(contract_id, chunk_index),
                "contract_id": contract_id,
                "text": " ".join(window),
                "chunk_index": chunk_index,
            }
        )
        i += stride
    return chunks


def ingest_contract(
    contract_id: str,
    raw_text: str,
    neo4j_session: Any,
    embedding_provider: EmbeddingProvider,
) -> dict:
    """Chunk, embed, and write a contract to Neo4j using MERGE semantics.

    Every Neo4j write uses MERGE keyed on chunk_id (Chunk nodes) or contract_id
    (Contract node), making repeated calls fully idempotent.  chunks_merged,
    nodes_merged, and edges_merged reflect actual nodes_created /
    relationships_created reported by Neo4j — all are 0 on repeat calls when
    the data is already present.

    Args:
        contract_id:        Normalised contract identifier.
        raw_text:           Full contract text (.md or .txt content).
        neo4j_session:      Open Neo4j session.
        embedding_provider: Provider whose embed() is called once per invocation.

    Returns:
        dict with keys:
            contract_id   (str) — echoed back.
            chunks_merged (int) — new Chunk nodes created (0 on repeat).
            nodes_merged  (int) — total new nodes created (0 on repeat).
            edges_merged  (int) — new FROM_CONTRACT edges created (0 on repeat).

    Raises:
        ValueError: raw_text is empty or whitespace-only.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("raw_text must not be empty")

    chunks = _split_text(raw_text, contract_id)

    # Single batched embed call — one embed() per invocation.
    embeddings = embedding_provider.embed([c["text"] for c in chunks])
    embedded = [
        {**c, "embedding": emb}
        for c, emb in zip(chunks, embeddings, strict=True)
    ]

    # MERGE the Contract node; track whether it was newly created.
    contract_result = neo4j_session.run(
        _MERGE_CONTRACT, {"contract_id": contract_id}
    )
    contract_nodes_created = contract_result.consume().counters.nodes_created

    # MERGE Chunk nodes with embeddings.
    chunk_rows = [
        {
            "chunk_id": c["chunk_id"],
            "contract_id": c["contract_id"],
            "text": c["text"],
            "embedding": c["embedding"],
            "chunk_index": c["chunk_index"],
        }
        for c in embedded
    ]
    chunk_result = neo4j_session.run(_MERGE_CHUNKS, {"rows": chunk_rows})
    chunks_created = chunk_result.consume().counters.nodes_created

    # MERGE FROM_CONTRACT edges.
    edge_rows = [
        {"chunk_id": c["chunk_id"], "contract_id": contract_id}
        for c in embedded
    ]
    edge_result = neo4j_session.run(_MERGE_FROM_CONTRACT, {"rows": edge_rows})
    edges_created = edge_result.consume().counters.relationships_created

    logger.info(
        "ingest_contract: contract_id=%r chunks_merged=%d nodes_merged=%d edges_merged=%d",
        contract_id,
        chunks_created,
        contract_nodes_created + chunks_created,
        edges_created,
    )

    return {
        "contract_id": contract_id,
        "chunks_merged": chunks_created,
        "nodes_merged": contract_nodes_created + chunks_created,
        "edges_merged": edges_created,
    }
