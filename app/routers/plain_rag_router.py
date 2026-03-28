"""Plain-RAG query router: POST /query/plain-rag (T030.b).

Accepts QueryRequest (question + top_k), calls the plain-RAG pipeline,
maps the pipeline result to AnswerSchema with mode='plain_rag'.

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

from fastapi import APIRouter, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.pipelines.plain_rag import PlainRagPipeline
from app.schemas.query_schemas import QueryRequest
from graphrag_assistant.schemas import AnswerSchema

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


@router.post(
    "/query/plain-rag",
    response_model=AnswerSchema,
    status_code=status.HTTP_200_OK,
)
async def query_plain_rag(
    body: QueryRequest,
    request: Request,
    _api_key: str | None = Security(_api_key_scheme),
) -> AnswerSchema:
    """Run the plain-RAG pipeline and return an AnswerSchema.

    Authentication: X-Api-Key header validated against API_KEY env var.
    Missing header → 422.  Invalid key → 401.

    All five retrieval_debug fields (graph_query, entity_matches,
    retrieved_node_ids, chunk_ids, timings) must be present in the pipeline
    result; a missing field raises HTTP 500 with a descriptive message.

    Args:
        body:     QueryRequest with question and optional top_k.
        request:  FastAPI request for accessing app.state providers.
        _api_key: API key from X-Api-Key header (None if absent).

    Returns:
        AnswerSchema with mode='plain_rag'.

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

    pipeline = PlainRagPipeline(
        embedding_provider=request.app.state.embedding_provider,
        generation_provider=request.app.state.generation_provider,
        driver=request.app.state.neo4j_driver,
        top_k=body.top_k,
    )
    result = pipeline.execute(body.question)

    debug = result.retrieval_debug
    for field in _DEBUG_FIELDS:
        if not hasattr(debug, field):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Pipeline result missing retrieval_debug.{field}",
            )

    logger.info(
        "query_plain_rag: question=%r chunks=%d",
        body.question,
        len(debug.chunk_ids),
    )
    return result
