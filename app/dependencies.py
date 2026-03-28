"""FastAPI dependency functions for shared resource injection (T026.a).

Each function reads a provider from ``request.app.state`` and raises
HTTP 500 when the provider was not initialised at startup.  Route
handlers can use these via ``Depends()`` or tests can replace them with
``dependency_overrides`` without touching app.state.

Authentication helpers
----------------------
``require_api_key`` and ``verify_csrf`` are defined here so every router
shares one implementation instead of each duplicating the logic.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import Header, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API-key auth — shared across all authenticated routes.
# auto_error=False: FastAPI passes None instead of raising 422, letting
# require_api_key distinguish missing (→ 422) from invalid (→ 401).
# ---------------------------------------------------------------------------

_ALLOWED_KEYS: frozenset[str] = frozenset(
    k.strip()
    for k in os.environ.get("API_KEY", "dev-key-change-in-prod").split(",")
    if k.strip()
)

_api_key_scheme = APIKeyHeader(name="X-Api-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(_api_key_scheme)) -> str:
    """Enforce X-Api-Key authentication.

    Raises:
        HTTP 422: header absent.
        HTTP 401: key present but not in allowed set.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="X-Api-Key header required",
        )
    if api_key not in _ALLOWED_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key


def verify_csrf(
    origin: str | None = Header(default=None),
    referer: str | None = Header(default=None),
    host: str | None = Header(default=None),
) -> None:
    """CSRF mitigation via strict Origin/Referer validation.

    Browsers always send Origin (or Referer) on cross-origin POST requests.
    Programmatic API clients that omit both headers pass through because they
    cannot be the vector for CSRF attacks.

    Raises:
        HTTP 403: Origin or Referer present but does not match Host.
    """
    source = origin or referer
    if source and host:
        # Accept both schemes; strip any trailing path from Referer.
        allowed = (f"http://{host}", f"https://{host}")
        if not any(source == a or source.startswith(a + "/") for a in allowed):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF check failed: Origin/Referer does not match Host",
            )


def get_neo4j_driver(request: Request) -> Any:
    """Return the Neo4j driver stored on ``app.state``.

    Raises:
        HTTP 500: driver was not set during application startup.
    """
    try:
        return request.app.state.neo4j_driver
    except AttributeError as exc:
        logger.warning("neo4j_driver missing from app.state: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Neo4j driver not initialised",
        ) from exc


def get_embedding_provider(request: Request) -> Any:
    """Return the embedding provider stored on ``app.state``.

    Raises:
        HTTP 500: provider was not set during application startup.
    """
    try:
        return request.app.state.embedding_provider
    except AttributeError as exc:
        logger.warning("embedding_provider missing from app.state: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding provider not initialised",
        ) from exc


def get_generation_provider(request: Request) -> Any:
    """Return the generation provider stored on ``app.state``.

    Raises:
        HTTP 500: provider was not set during application startup.
    """
    try:
        return request.app.state.generation_provider
    except AttributeError as exc:
        logger.warning("generation_provider missing from app.state: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Generation provider not initialised",
        ) from exc
