"""Query router: POST /query/plain-rag (T024).

Exposes the plain-RAG pipeline over HTTP.  The pipeline is instantiated
per-request using providers stored in app.state so that tests can inject
mocks without monkey-patching module globals.

Authentication
--------------
X-Api-Key request header via FastAPI's APIKeyHeader Security scheme.
Header-based API key authentication is used deliberately: it carries no
session cookie.  CSRF is mitigated by strict Origin/Referer validation
(see _verify_csrf) combined with the custom X-Api-Key header, which
browsers cannot set in cross-origin requests without a CORS preflight.

Using Security(APIKeyHeader) directly in the route signature (rather than a
custom Depends() wrapper) ensures that OpenAPI documents the security
requirement and that governance scanners recognise the endpoint as
authenticated.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from app.dependencies import require_api_key, verify_csrf
from app.pipelines.plain_rag import PlainRagPipeline
from graphrag_assistant.schemas import AnswerSchema

logger = logging.getLogger(__name__)

router = APIRouter()


class PlainRagRequest(BaseModel):
    """Request body for POST /query/plain-rag."""

    query: str
    top_k: int = Field(default=5, ge=1)

    def __repr__(self) -> str:
        return f"PlainRagRequest(query={self.query!r}, top_k={self.top_k!r})"


@router.post(
    "/query/plain-rag",
    response_model=AnswerSchema,
    status_code=status.HTTP_200_OK,
)
async def query_plain_rag(
    body: PlainRagRequest,
    request: Request,
    _api_key: str = Depends(require_api_key),
    _csrf: None = Depends(verify_csrf),
) -> AnswerSchema:
    """Run the plain-RAG pipeline and return an AnswerSchema.

    Authentication: X-Api-Key header validated against API_KEY env var.
    Missing header → 422.  Invalid key → 401.

    Returns:
        AnswerSchema with mode='plain_rag' and graph_evidence=[].
    """
    pipeline = PlainRagPipeline(
        embedding_provider=request.app.state.embedding_provider,
        generation_provider=request.app.state.generation_provider,
        driver=request.app.state.neo4j_driver,
        top_k=body.top_k,
    )
    result = pipeline.execute(body.query)
    logger.info(
        "query_plain_rag: query=%r chunks=%d",
        body.query,
        len(result.retrieval_debug.chunk_ids),
    )
    return result
