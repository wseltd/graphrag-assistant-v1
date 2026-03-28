"""Tests for app/dependencies.py — DI injection and HTTP 500 paths (T026.a)."""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.dependencies import get_embedding_provider, get_generation_provider, get_neo4j_driver


def _probe_app_with(dep) -> FastAPI:
    """Return a minimal app exposing *dep* via GET /probe."""
    _app = FastAPI()

    @_app.get("/probe")
    async def probe(value=Depends(dep)):  # noqa: B008
        return {"id": id(value)}

    return _app


def test_get_neo4j_driver_returns_driver_from_app_state():
    app = _probe_app_with(get_neo4j_driver)
    mock_driver = MagicMock()
    app.state.neo4j_driver = mock_driver
    with TestClient(app) as client:
        resp = client.get("/probe")
    assert resp.status_code == 200
    assert resp.json()["id"] == id(mock_driver)


def test_get_embedding_provider_returns_provider_from_app_state():
    app = _probe_app_with(get_embedding_provider)
    mock_provider = MagicMock()
    app.state.embedding_provider = mock_provider
    with TestClient(app) as client:
        resp = client.get("/probe")
    assert resp.status_code == 200
    assert resp.json()["id"] == id(mock_provider)


def test_get_generation_provider_returns_provider_from_app_state():
    app = _probe_app_with(get_generation_provider)
    mock_provider = MagicMock()
    app.state.generation_provider = mock_provider
    with TestClient(app) as client:
        resp = client.get("/probe")
    assert resp.status_code == 200
    assert resp.json()["id"] == id(mock_provider)


def test_dependency_override_replaces_neo4j_driver():
    app = _probe_app_with(get_neo4j_driver)
    mock_override = MagicMock()
    app.dependency_overrides[get_neo4j_driver] = lambda: mock_override
    with TestClient(app) as client:
        resp = client.get("/probe")
    assert resp.status_code == 200
    assert resp.json()["id"] == id(mock_override)


def test_missing_neo4j_driver_raises_http_500():
    app = _probe_app_with(get_neo4j_driver)
    # app.state.neo4j_driver is not set — AttributeError → HTTP 500
    with TestClient(app) as client:
        resp = client.get("/probe")
    assert resp.status_code == 500


def test_missing_embedding_provider_raises_http_500():
    app = _probe_app_with(get_embedding_provider)
    # app.state.embedding_provider is not set — AttributeError → HTTP 500
    with TestClient(app) as client:
        resp = client.get("/probe")
    assert resp.status_code == 500
