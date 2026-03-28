"""Chunk loader orchestrator: reads a JSONL file, embeds texts, and writes all
graph artefacts to Neo4j (T018.e).

Public API
----------
load_chunks_to_graph(chunks_path, embed_fn, session)
    Read → embed → write_chunk_nodes → write_chunk_contract_edges →
    write_chunk_company_edges → write_related_to_edges.

Design
------
* load_chunks_to_graph is the single entry point for ingesting processed chunks
  into Neo4j.  It delegates to four lower-level writers so that each concern
  stays independently testable.
* embed_fn is injected so callers control the model and unit tests can pass a
  deterministic stub.  It is called exactly once per invocation with all chunk
  texts batched together.
* Chunks with a falsy contract_id trigger a WARNING log in
  write_chunk_contract_edges and are skipped for that edge type; their Chunk
  nodes are still written (the acceptance criterion is that Chunk count equals
  record count, not that every chunk has a valid contract).
* All four writers use MERGE, so running the loader twice is idempotent: the
  node count and edge counts are unchanged on repeat runs.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from graphrag_assistant.loaders.chunk_edge_writer import (
    write_chunk_company_edges,
    write_chunk_contract_edges,
)
from graphrag_assistant.loaders.chunk_node_writer import write_chunk_nodes
from graphrag_assistant.loaders.chunk_reader import load_chunks
from graphrag_assistant.loaders.chunk_related_writer import write_related_to_edges

logger = logging.getLogger(__name__)


def load_chunks_to_graph(
    chunks_path: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    session: Any,
) -> None:
    """Read chunks from *chunks_path*, embed, and write all graph artefacts.

    Steps performed in order:
    1. load_chunks — parse and validate the JSONL file.
    2. embed_fn — batch-embed all chunk texts in a single call.
    3. Attach the returned embeddings to fresh chunk dicts (originals unchanged).
    4. write_chunk_nodes — MERGE Chunk nodes with embeddings.
    5. write_chunk_contract_edges — MERGE FROM_CONTRACT edges.
    6. write_chunk_company_edges — MERGE ABOUT_COMPANY edges.
    7. write_related_to_edges — MERGE RELATED_TO edges.

    embed_fn is called exactly once per invocation regardless of chunk count.
    Chunks with a falsy contract_id are logged at WARNING by
    write_chunk_contract_edges and skipped for that edge only; their Chunk
    nodes are still written.  All four writers use MERGE so repeated calls
    produce the same node and edge counts without duplicates.

    Args:
        chunks_path: Filesystem path to a validated JSONL chunks file.
        embed_fn:    Callable accepting list[str], returning list[list[float]].
        session:     Open Neo4j session.
    """
    chunks = load_chunks(chunks_path)
    if not chunks:
        logger.info("load_chunks_to_graph: no chunks found in %r", chunks_path)
        return

    # Single batched embed call — one embed_fn call per invocation.
    embeddings = embed_fn([c["text"] for c in chunks])

    # Attach embeddings to fresh dicts; leave original dicts unchanged.
    embedded = [
        {**c, "embedding": emb}
        for c, emb in zip(chunks, embeddings, strict=True)
    ]

    write_chunk_nodes(session, embedded)
    write_chunk_contract_edges(session, embedded)
    write_chunk_company_edges(session, embedded)
    write_related_to_edges(session, embedded)

    logger.info(
        "load_chunks_to_graph: loaded %d chunks from %r",
        len(chunks),
        chunks_path,
    )
