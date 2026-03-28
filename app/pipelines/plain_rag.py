"""Plain RAG pipeline: vector-only retrieval and generation (T024).

Embeds the query, runs an unconstrained vector similarity search on Chunk nodes
using Neo4j's vector index, passes retrieved chunks to the generation provider,
and returns a schema-compliant AnswerSchema with mode='plain_rag'.

This pipeline shares no code paths with the graph-RAG pipeline. That separation
is intentional: shared internals would blur the retrieval modes and make
benchmark comparisons unreliable.

Cypher safety: all queries use $param placeholders — no f-strings, no
%-format.  The single query issued uses CALL db.index.vector.queryNodes (not
MATCH) so callers can assert that no graph-traversal Cypher was executed.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from graphrag_assistant.providers.base import EmbeddingProvider, GenerationProvider
from graphrag_assistant.schemas import AnswerSchema, RetrievalDebug

logger = logging.getLogger(__name__)

# Vector index name — must match neo4j_client._DDL_CHUNK_VECTOR_INDEX.
_INDEX_NAME = "chunk_embedding_idx"

# ---------------------------------------------------------------------------
# Cypher — no MATCH, no f-strings, no %-format, all values via $params.
# CALL db.index.vector.queryNodes limits results to $top_k directly; no
# multiplier needed because there is no relationship-filter post-step.
# ---------------------------------------------------------------------------

_QUERY_VECTOR = (
    "CALL db.index.vector.queryNodes('chunk_embedding_idx', $top_k, $query_vector) "
    "YIELD node AS chunk, score "
    "RETURN chunk.chunk_id AS chunk_id, "
    "chunk.contract_id AS contract_id, "
    "chunk.text AS text, "
    "score "
    "ORDER BY score DESC"
)


class PlainRagPipeline:
    """Vector-only retrieval and generation pipeline.

    No graph traversal, no entity extraction, no Cypher beyond the vector
    index query.  Returns a fully schema-compliant AnswerSchema so the
    benchmark scorer needs no special-casing for plain-RAG responses.

    Args:
        embedding_provider:  Converts the raw query string to a dense vector.
        generation_provider: Synthesises an answer from retrieved chunks.
        driver:              Open Neo4j driver (session managed internally).
        top_k:               Maximum number of chunks to retrieve. Default 5.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        generation_provider: GenerationProvider,
        driver: Any,
        *,
        top_k: int = 5,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._generation_provider = generation_provider
        self._driver = driver
        self._top_k = top_k

    def execute(self, query: str) -> AnswerSchema:
        """Run the plain-RAG pipeline and return a schema-compliant answer.

        Phases (each timed individually):
          1. embed   — encode the query string into a dense vector.
          2. retrieve — run the unconstrained vector index query in Neo4j.
          3. generate — pass retrieved chunks to the generation provider.

        retrieval_debug is always populated:
          - graph_query=None   (no graph work performed)
          - entity_matches=[]  (no entity resolution)
          - retrieved_node_ids=[] (no graph nodes resolved)
          - chunk_ids          list of chunk IDs returned by the vector index
          - timings            embed_ms, retrieve_ms, generate_ms (all >= 0)

        Args:
            query: Raw user question string.

        Returns:
            AnswerSchema with mode='plain_rag' and graph_evidence=[].
        """
        # ------------------------------------------------------------------ #
        # Phase 1: embed                                                       #
        # ------------------------------------------------------------------ #
        t0 = time.monotonic()
        vector = self._embedding_provider.embed([query])[0]
        embed_ms = (time.monotonic() - t0) * 1000.0

        # ------------------------------------------------------------------ #
        # Phase 2: vector retrieval — no MATCH, no graph traversal            #
        # ------------------------------------------------------------------ #
        t1 = time.monotonic()
        with self._driver.session() as session:
            rows = session.run(
                _QUERY_VECTOR,
                {"top_k": self._top_k, "query_vector": vector},
            )
            raw_chunks = [
                {
                    "doc_id": row.get("contract_id") or "",
                    "chunk_id": row["chunk_id"],
                    "text": row["text"],
                }
                for row in rows
            ]
        retrieve_ms = (time.monotonic() - t1) * 1000.0

        # ------------------------------------------------------------------ #
        # Phase 3: generate                                                    #
        # ------------------------------------------------------------------ #
        t2 = time.monotonic()
        generated = self._generation_provider.generate(
            prompt="[mode:plain_rag]" + query,
            graph_facts=[],
            chunks=raw_chunks,
        )
        generate_ms = (time.monotonic() - t2) * 1000.0

        chunk_ids = [c["chunk_id"] for c in raw_chunks]

        logger.info(
            "plain_rag: query=%r chunks=%d embed_ms=%.1f retrieve_ms=%.1f generate_ms=%.1f",
            query,
            len(chunk_ids),
            embed_ms,
            retrieve_ms,
            generate_ms,
        )

        return AnswerSchema(
            answer=generated.answer,
            graph_evidence=[],
            text_citations=generated.text_citations,
            retrieval_debug=RetrievalDebug(
                graph_query=None,
                entity_matches=[],
                retrieved_node_ids=[],
                chunk_ids=chunk_ids,
                timings={
                    "embed_ms": embed_ms,
                    "retrieve_ms": retrieve_ms,
                    "generate_ms": generate_ms,
                },
            ),
            mode="plain_rag",
        )
