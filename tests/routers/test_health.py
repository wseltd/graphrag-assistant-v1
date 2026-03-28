"""Tests for GET /health endpoint (T027).

health_ok       — probe succeeds → HTTP 200, neo4j_status='ok'
health_neo4j_down — probe raises ServiceUnavailable → HTTP 503, neo4j_status='unreachable'
health_no_driver  — driver absent from app.state → HTTP 503
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from neo4j.exceptions import ServiceUnavailable

from app.routers.health import router


def _make_app(*, driver=None, missing_driver: bool = False) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    if not missing_driver:
        app.state.neo4j_driver = driver
    return app


def test_health_ok():
    """RETURN 1 probe succeeds → HTTP 200, neo4j_status='ok'."""
    mock_driver = MagicMock()
    app = _make_app(driver=mock_driver)
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["neo4j_status"] == "ok"
    assert data["timestamp"]


def test_health_neo4j_down():
    """ServiceUnavailable on probe → HTTP 503, neo4j_status='unreachable', error present."""
    mock_driver = MagicMock()
    mock_driver.session.side_effect = ServiceUnavailable("connection refused")
    app = _make_app(driver=mock_driver)
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 503
    data = resp.json()
    assert data["neo4j_status"] == "unreachable"
    assert data["error"] is not None


def test_health_no_driver():
    """No driver on app.state → HTTP 503, neo4j_status='unreachable'."""
    app = _make_app(missing_driver=True)
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 503
    data = resp.json()
    assert data["neo4j_status"] == "unreachable"
