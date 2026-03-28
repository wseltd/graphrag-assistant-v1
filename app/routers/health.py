"""Health check endpoint (T027).

GET /health — runs a RETURN 1 probe against Neo4j.
Returns HTTP 200 on success; HTTP 503 when the driver is absent or the probe fails.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

_PROBE_QUERY = "RETURN 1 AS ok"


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    neo4j_status: Literal["ok", "unreachable"]
    timestamp: str
    error: str | None = None

    def __repr__(self) -> str:
        return (
            f"HealthResponse(status={self.status!r}, "
            f"neo4j_status={self.neo4j_status!r}, "
            f"timestamp={self.timestamp!r})"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request, response: Response) -> HealthResponse:
    """Run a Neo4j liveness probe and return service status."""
    timestamp = datetime.now(UTC).isoformat()

    try:
        driver = request.app.state.neo4j_driver
    except AttributeError:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="degraded",
            neo4j_status="unreachable",
            timestamp=timestamp,
            error="Neo4j driver not initialised",
        )

    try:
        with driver.session() as session:
            session.run(_PROBE_QUERY).consume()
        return HealthResponse(status="ok", neo4j_status="ok", timestamp=timestamp)
    except Exception as exc:  # noqa: BLE001 — any driver error → degraded
        logger.warning("Neo4j health probe failed: %s", exc)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="degraded",
            neo4j_status="unreachable",
            timestamp=timestamp,
            error=str(exc),
        )
