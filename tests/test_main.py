"""Tests for app/main.py create_app() factory (T026.c).

Coverage:
- router_registration  (1): all expected path prefixes present after app init
- lifespan             (2): successful startup stores providers; unreachable Neo4j raises
  RuntimeError
- dependency_injection (3): driver injected from state; override replaces provider;
  missing key → 500
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.dependencies import get_neo4j_driver
from app.main import create_app

_STUB_PATHS = {
    "/health",
    "/seed",
    "/entities/{entity_type}/{entity_id}",
    "/benchmark/run",
    "/benchmark/results/{run_id}",
}


def _lifespan_patches(fail: bool = False):
    """Return (mock_client, mock_driver) ready for patching app.lifespan."""
    mock_driver = MagicMock()
    mock_client = MagicMock()
    mock_client._driver = mock_driver
    if fail:
        mock_client.verify_connectivity.side_effect = Exception("connection refused")
    else:
        mock_client.verify_connectivity.return_value = None
    return mock_client, mock_driver


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------


def test_all_expected_path_prefixes_registered():
    app = create_app()
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    for expected in _STUB_PATHS:
        assert expected in paths, f"Path {expected!r} not found; registered: {paths}"
    # ingest and query routers land under /api/v1
    assert any(p.startswith("/api/v1") for p in paths)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


def test_successful_startup_stores_providers_on_app_state():
    mock_client, mock_driver = _lifespan_patches()
    mock_embed = MagicMock()
    mock_gen = MagicMock()
    with patch("app.lifespan.Neo4jClient", return_value=mock_client):
        with patch("app.lifespan.SentenceTransformerProvider", return_value=mock_embed):
            with patch("app.lifespan.TemplateGenerationProvider", return_value=mock_gen):
                app = create_app()
                with TestClient(app):
                    assert app.state.neo4j_driver is mock_driver
                    assert app.state.embedding_provider is mock_embed
                    assert app.state.generation_provider is mock_gen


def test_unreachable_neo4j_raises_runtime_error():
    mock_client, _ = _lifespan_patches(fail=True)
    with patch("app.lifespan.Neo4jClient", return_value=mock_client):
        with patch("app.lifespan.SentenceTransformerProvider"):
            app = create_app()
            with pytest.raises(RuntimeError, match="Neo4j unreachable"):
                with TestClient(app):
                    pass


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


def test_neo4j_provider_injected_from_app_state():
    """get_neo4j_driver returns the object stored on app.state.neo4j_driver."""
    app = FastAPI()
    mock_driver = MagicMock()
    app.state.neo4j_driver = mock_driver

    @app.get("/probe")
    async def probe(driver=Depends(get_neo4j_driver)):  # noqa: B008
        return {"id": id(driver)}

    with TestClient(app) as client:
        resp = client.get("/probe")
    assert resp.status_code == 200
    assert resp.json()["id"] == id(mock_driver)


def test_dependency_override_replaces_real_provider():
    """dependency_overrides replaces the injected driver without touching app.state."""
    mock_client, _ = _lifespan_patches()
    mock_override = MagicMock()
    with patch("app.lifespan.Neo4jClient", return_value=mock_client):
        with patch("app.lifespan.SentenceTransformerProvider"):
            with patch("app.lifespan.TemplateGenerationProvider"):
                app = create_app()

                @app.get("/probe")
                async def probe(driver=Depends(get_neo4j_driver)):  # noqa: B008
                    return {"id": id(driver)}

                app.dependency_overrides[get_neo4j_driver] = lambda: mock_override
                with TestClient(app) as client:
                    resp = client.get("/probe")
                assert resp.status_code == 200
                assert resp.json()["id"] == id(mock_override)


def test_missing_provider_key_on_app_state_raises_http_500():
    """get_neo4j_driver raises HTTP 500 when neo4j_driver is absent from app.state."""
    app = FastAPI()

    @app.get("/probe")
    async def probe(driver=Depends(get_neo4j_driver)):  # noqa: B008
        return {"id": id(driver)}

    with TestClient(app) as client:
        resp = client.get("/probe")
    assert resp.status_code == 500
