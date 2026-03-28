"""Seed router: POST /seed (T028).

Thin HTTP adapter. All MERGE/DELETE logic lives in the orchestrator.
No authentication required for MVP.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from graphrag_assistant.seed.orchestrator import seed

logger = logging.getLogger(__name__)

router = APIRouter()


class SeedResult(BaseModel):
    """Response body for POST /seed."""

    nodes_created: int
    relationships_created: int
    reset_performed: bool
    duration_ms: float

    def __repr__(self) -> str:
        return (
            f"SeedResult(nodes_created={self.nodes_created!r}, "
            f"relationships_created={self.relationships_created!r}, "
            f"reset_performed={self.reset_performed!r}, "
            f"duration_ms={self.duration_ms!r})"
        )


@router.post("/seed", response_model=SeedResult, status_code=status.HTTP_200_OK)
async def seed_endpoint(
    request: Request,
    reset: bool = False,
) -> SeedResult:
    """Run the full graph seeding pipeline.

    Args:
        request: FastAPI request; neo4j_driver read from app.state.
        reset:   Clear the graph before loading when True.

    Returns:
        SeedResult with node/relationship counts, reset flag, and wall-clock ms.

    Raises:
        HTTP 500: Orchestrator raised any exception.
    """
    driver = request.app.state.neo4j_driver
    t0 = time.monotonic()
    try:
        counts = seed(driver, reset=reset)
    except Exception as exc:
        logger.warning("seed_endpoint: orchestrator raised %r", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"seed failed: {exc}",
        ) from exc
    duration_ms = (time.monotonic() - t0) * 1000.0

    logger.info(
        "seed_endpoint: reset=%s nodes=%d edges=%d duration_ms=%.1f",
        reset,
        counts["nodes_written"],
        counts["edges_written"],
        duration_ms,
    )

    return SeedResult(
        nodes_created=counts["nodes_written"],
        relationships_created=counts["edges_written"],
        reset_performed=reset,
        duration_ms=duration_ms,
    )
