"""Unit tests for app.routers.seed — POST /seed (T028).

All tests run without a live Neo4j instance.  The FastAPI app is built inline
so each test controls app.state and the orchestrator mock independently.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.seed import router


def _build_app(neo4j_driver: object | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.neo4j_driver = neo4j_driver or MagicMock()
    return app


_MOCK_COUNTS = {"nodes_written": 42, "edges_written": 17}


def test_seed_success() -> None:
    """Mock orchestrator returns counts → 200 with correct SeedResult fields."""
    app = _build_app()
    client = TestClient(app)

    with patch("app.routers.seed.seed", return_value=_MOCK_COUNTS) as mock_seed:
        resp = client.post("/seed")

    assert resp.status_code == 200
    body = resp.json()
    assert body["nodes_created"] == 42
    assert body["relationships_created"] == 17
    assert body["reset_performed"] is False
    assert isinstance(body["duration_ms"], float)
    assert body["duration_ms"] >= 0.0
    mock_seed.assert_called_once()


def test_seed_with_reset() -> None:
    """?reset=true → orchestrator called with reset=True, reset_performed=true."""
    app = _build_app()
    client = TestClient(app)

    with patch("app.routers.seed.seed", return_value=_MOCK_COUNTS) as mock_seed:
        resp = client.post("/seed?reset=true")

    assert resp.status_code == 200
    body = resp.json()
    assert body["reset_performed"] is True
    _args, _kwargs = mock_seed.call_args
    assert _kwargs.get("reset") is True or _args[1] is True


def test_seed_default_no_reset() -> None:
    """No reset param → orchestrator called with reset=False, reset_performed=false."""
    app = _build_app()
    client = TestClient(app)

    with patch("app.routers.seed.seed", return_value=_MOCK_COUNTS) as mock_seed:
        resp = client.post("/seed")

    assert resp.status_code == 200
    body = resp.json()
    assert body["reset_performed"] is False
    _args, _kwargs = mock_seed.call_args
    reset_val = _kwargs.get("reset", _args[1] if len(_args) > 1 else False)
    assert reset_val is False


def test_seed_orchestrator_error_message() -> None:
    """Orchestrator raises ValueError → 500 with detail containing exception message."""
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)

    with patch("app.routers.seed.seed", side_effect=ValueError("neo4j unreachable")):
        resp = client.post("/seed")

    assert resp.status_code == 500
    body = resp.json()
    assert "seed failed" in body["detail"]
    assert "neo4j unreachable" in body["detail"]


def test_seed_orchestrator_error_no_traceback() -> None:
    """500 response body must not contain raw Python traceback text."""
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)

    with patch("app.routers.seed.seed", side_effect=RuntimeError("boom")):
        resp = client.post("/seed")

    assert resp.status_code == 500
    body_text = resp.text
    assert "Traceback" not in body_text
    assert "File " not in body_text
