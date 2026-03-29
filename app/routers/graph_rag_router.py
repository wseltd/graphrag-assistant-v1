"""Graph-RAG query router: POST /query/graph-rag (T030.c).

Accepts QueryRequest, runs the four graph-RAG pipeline stages, and returns
AnswerSchema with mode='graph_rag'.  All five retrieval_debug fields are
always populated; a missing field raises HTTP 500.

Authentication
--------------
X-Api-Key request header via FastAPI's APIKeyHeader Security scheme.
Header-based API key authentication carries no session cookie so CSRF
mitigations are not applicable.  The key is validated against the API_KEY
environment variable (comma-separated for multiple keys).

Using Security(APIKeyHeader) directly in the route signature ensures OpenAPI
documents the security requirement and governance scanners recognise the
endpoint as authenticated.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.pipelines.constrained_retrieval import retrieve_constrained
from app.pipelines.entity_resolver import resolve_entities
from app.pipelines.graph_traversal import traverse_from_anchors
from app.schemas.query_schemas import QueryRequest
from graphrag_assistant.schemas import AnswerSchema, GraphFact, RetrievalDebug

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_KEYS: frozenset[str] = frozenset(
    k.strip()
    for k in os.environ.get("API_KEY", "dev-key-change-in-prod").split(",")
    if k.strip()
)

# auto_error=False: return None on missing header so the handler can
# distinguish missing (→ 422) from invalid (→ 401).
_api_key_scheme = APIKeyHeader(name="X-Api-Key", auto_error=False)

_DEBUG_FIELDS: tuple[str, ...] = (
    "graph_query",
    "entity_matches",
    "retrieved_node_ids",
    "chunk_ids",
    "timings",
)

# Cypher for vector search — matches the plain-RAG index query.
_VECTOR_QUERY = (
    "CALL db.index.vector.queryNodes('chunk_embedding_idx', $top_k, $query_vector) "
    "YIELD node AS chunk, score "
    "RETURN chunk.chunk_id AS chunk_id, "
    "chunk.contract_id AS contract_id, "
    "chunk.text AS text, "
    "score "
    "ORDER BY score DESC"
)

# Tag stored in retrieval_debug.graph_query for all graph-RAG responses.
_GRAPH_QUERY_TAG = "GRAPH_TRAVERSAL_2HOP"


class _Neo4jVectorStore:
    """Adapter: embeds query and searches the Neo4j chunk_embedding_idx.

    Provides the vector_store.search(query, top_k) interface required by
    retrieve_constrained without duplicating embedding logic outside the
    pipeline boundary.
    """

    def __init__(self, embedding_provider: Any, driver: Any) -> None:
        self._embedding_provider = embedding_provider
        self._driver = driver

    def search(self, query: str, top_k: int) -> list[dict]:
        """Embed *query* and return up to *top_k* chunks by cosine similarity."""
        vector = self._embedding_provider.embed([query])[0]
        with self._driver.session() as session:
            rows = session.run(
                _VECTOR_QUERY,
                {"top_k": top_k, "query_vector": vector},
            )
            return [
                {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["contract_id"] or "",
                    "text": row["text"],
                    "score": float(row["score"]),
                }
                for row in rows
            ]


@router.post(
    "/query/graph-rag",
    response_model=AnswerSchema,
    status_code=status.HTTP_200_OK,
)
async def query_graph_rag(
    body: QueryRequest,
    request: Request,
    _api_key: str | None = Security(_api_key_scheme),
) -> AnswerSchema:
    """Run the graph-RAG pipeline and return a schema-compliant AnswerSchema.

    Authentication: X-Api-Key header validated against API_KEY env var.
    Missing header → 422.  Invalid key → 401.

    Pipeline stages (each timed):
      1. resolve_entities      — query → list[EntityMatch]
      2. traverse_from_anchors — node_ids → GraphTraversalResult
      3. retrieve_constrained  — chunk_ids → list[RankedChunk]
      4. generate              — triples + chunks → AnswerSchema

    All five retrieval_debug fields (graph_query, entity_matches,
    retrieved_node_ids, chunk_ids, timings) are always populated; a missing
    field raises HTTP 500.

    Args:
        body:     QueryRequest with question and optional top_k.
        request:  FastAPI request for accessing app.state providers.
        _api_key: API key from X-Api-Key header (None if absent).

    Returns:
        AnswerSchema with mode='graph_rag'.

    Raises:
        HTTP 422: question is empty, body invalid, or X-Api-Key absent.
        HTTP 401: key present but not in the allowed set.
        HTTP 500: pipeline result missing a required retrieval_debug field.
    """
    if not _api_key:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="X-Api-Key header required",
        )
    if _api_key not in _ALLOWED_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    driver = request.app.state.neo4j_driver
    vector_store = _Neo4jVectorStore(
        request.app.state.embedding_provider, driver
    )
    generation_provider = request.app.state.generation_provider

    # Stages 1 + 2: entity resolution and graph traversal (shared session).
    t0 = time.monotonic()
    with driver.session() as session:
        entity_matches = resolve_entities(
            body.question, session, top_k=body.top_k
        )
        node_ids = [m.node_id for m in entity_matches]
        traversal = traverse_from_anchors(node_ids, session, max_hops=2)
    graph_ms = (time.monotonic() - t0) * 1000.0

    # Stage 3: constrained vector retrieval.
    t1 = time.monotonic()
    chunks = retrieve_constrained(
        body.question, traversal.chunk_ids, vector_store, top_k=body.top_k
    )
    retrieve_ms = (time.monotonic() - t1) * 1000.0

    # Stage 4: answer generation.
    graph_facts = [
        {"source_id": t.src, "target_id": t.dst, "label": t.rel}
        for t in traversal.triples
    ]
    chunks_dicts = [
        {"doc_id": c.doc_id, "chunk_id": c.chunk_id, "text": c.text}
        for c in chunks
    ]
    t2 = time.monotonic()
    generated = generation_provider.generate(
        prompt="[mode:graph_rag]" + body.question,
        graph_facts=graph_facts,
        chunks=chunks_dicts,
    )
    generate_ms = (time.monotonic() - t2) * 1000.0

    graph_evidence = [
        GraphFact(source_id=t.src, target_id=t.dst, label=t.rel)
        for t in traversal.triples
    ]

    debug = RetrievalDebug(
        graph_query=_GRAPH_QUERY_TAG,
        entity_matches=[m.node_id for m in entity_matches],
        retrieved_node_ids=node_ids,
        chunk_ids=[c.chunk_id for c in chunks],
        timings={
            "graph_ms": graph_ms,
            "retrieve_ms": retrieve_ms,
            "generate_ms": generate_ms,
        },
    )

    for field in _DEBUG_FIELDS:
        if not hasattr(debug, field):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Pipeline result missing retrieval_debug.{field}",
            )

    logger.info(
        "query_graph_rag: question=%r triples=%d chunks=%d",
        body.question,
        len(traversal.triples),
        len(chunks),
    )

    return AnswerSchema(
        answer=generated.answer,
        graph_evidence=graph_evidence,
        text_citations=generated.text_citations,
        retrieval_debug=debug,
        mode="graph_rag",
    )
