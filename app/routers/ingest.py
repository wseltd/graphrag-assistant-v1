"""Ingest router: POST /ingest/contracts (T029.c).

Accepts multipart-encoded contract files (.md or .txt), runs the
ingest pipeline for each file, and returns per-file MERGE counts.
All writes are idempotent.

Authentication
--------------
Header-based API key authentication is used: the X-Api-Key header is
validated against the API_KEY environment variable (comma-separated list).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from app.dependencies import require_api_key, verify_csrf
from app.ingest.contract_ids import normalise_contract_id
from app.ingest.pipeline import ingest_contract

logger = logging.getLogger(__name__)

router = APIRouter()


class IngestFileResult(BaseModel):
    """Per-file result returned by POST /ingest/contracts."""

    contract_id: str
    chunks_merged: int
    nodes_merged: int
    edges_merged: int

    def __repr__(self) -> str:
        return (
            f"IngestFileResult(contract_id={self.contract_id!r}, "
            f"chunks_merged={self.chunks_merged!r}, "
            f"nodes_merged={self.nodes_merged!r}, "
            f"edges_merged={self.edges_merged!r})"
        )


class MultiIngestResponse(BaseModel):
    """Response body for POST /ingest/contracts."""

    results: list[IngestFileResult]

    def __repr__(self) -> str:
        return f"MultiIngestResponse(results={self.results!r})"


@router.post(
    "/ingest/contracts",
    response_model=MultiIngestResponse,
    status_code=status.HTTP_200_OK,
)
async def ingest_contracts(
    request: Request,
    files: list[UploadFile],
    _csrf: None = Depends(verify_csrf),
    _api_key: str = Depends(require_api_key),
) -> MultiIngestResponse:
    """Ingest one or more contract files into Neo4j.

    Each file is decoded as UTF-8, split into overlapping token-window
    chunks, embedded, and written to Neo4j via MERGE semantics keyed on
    contract_id (Contract node) and chunk_id (Chunk nodes).  Posting the
    same file twice returns the same contract_id and zero counts on the
    second call.

    Args:
        request: FastAPI request; neo4j_driver and embedding_provider
                 are read from app.state.
        files:   One or more multipart-encoded contract files.

    Returns:
        MultiIngestResponse containing one IngestFileResult per file.

    Raises:
        HTTP 422: Any uploaded file contains zero bytes, or the decoded
                  text is whitespace-only.
        HTTP 500: An unexpected error occurs during pipeline execution.
    """
    driver = request.app.state.neo4j_driver
    embedding_provider = request.app.state.embedding_provider

    results: list[IngestFileResult] = []
    for upload in files:
        raw_bytes = await upload.read()
        if not raw_bytes:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"File '{upload.filename}' is empty",
            )
        raw_text = raw_bytes.decode("utf-8")
        contract_id = normalise_contract_id(upload.filename or "unknown")

        try:
            with driver.session() as session:
                counts = ingest_contract(
                    contract_id, raw_text, session, embedding_provider
                )
        except ValueError as exc:
            logger.warning(
                "ingest_contracts: invalid content contract_id=%r: %s",
                contract_id,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.warning(
                "ingest_contracts: pipeline error contract_id=%r: %r",
                contract_id,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"ingest failed for '{contract_id}': {exc}",
            ) from exc

        results.append(IngestFileResult(**counts))

    return MultiIngestResponse(results=results)
