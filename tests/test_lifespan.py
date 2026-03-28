"""Tests for app/lifespan.py — startup, shutdown, and error paths (T026.b)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.lifespan import app_lifespan


def _make_app() -> FastAPI:
    """Create a minimal FastAPI app with app_lifespan attached."""
    _app = FastAPI(lifespan=app_lifespan)

    @_app.get("/probe")
    async def probe():
        return {"ok": True}

    return _app


def _mock_client(fail_connectivity: bool = False) -> tuple[MagicMock, MagicMock]:
    """Return (mock_client, mock_driver) for Neo4jClient patching."""
    mock_driver = MagicMock()
    mock_client = MagicMock()
    mock_client._driver = mock_driver
    if fail_connectivity:
        mock_client.verify_connectivity.side_effect = Exception("connection refused")
    else:
        mock_client.verify_connectivity.return_value = None
    return mock_client, mock_driver


class TestLifespanStartupSuccess:
    """Successful startup stores all three providers on app.state."""

    def test_stores_neo4j_driver_on_app_state(self):
        mock_client, mock_driver = _mock_client()
        with patch("app.lifespan.Neo4jClient", return_value=mock_client):
            with patch("app.lifespan.SentenceTransformerProvider"):
                with patch("app.lifespan.TemplateGenerationProvider"):
                    app = _make_app()
                    with TestClient(app):
                        assert app.state.neo4j_driver is mock_driver

    def test_stores_embedding_provider_on_app_state(self):
        mock_client, _ = _mock_client()
        mock_embed = MagicMock()
        with patch("app.lifespan.Neo4jClient", return_value=mock_client):
            with patch(
                "app.lifespan.SentenceTransformerProvider",
                return_value=mock_embed,
            ):
                with patch("app.lifespan.TemplateGenerationProvider"):
                    app = _make_app()
                    with TestClient(app):
                        assert app.state.embedding_provider is mock_embed

    def test_stores_generation_provider_on_app_state(self):
        mock_client, _ = _mock_client()
        mock_gen = MagicMock()
        with patch("app.lifespan.Neo4jClient", return_value=mock_client):
            with patch("app.lifespan.SentenceTransformerProvider"):
                with patch("app.lifespan.TemplateGenerationProvider", return_value=mock_gen):
                    app = _make_app()
                    with TestClient(app):
                        assert app.state.generation_provider is mock_gen


class TestLifespanStartupFailure:
    """Neo4j unreachable at startup raises a descriptive RuntimeError."""

    def test_unreachable_neo4j_raises_runtime_error(self):
        mock_client, _ = _mock_client(fail_connectivity=True)
        with patch("app.lifespan.Neo4jClient", return_value=mock_client):
            with patch("app.lifespan.SentenceTransformerProvider"):
                app = _make_app()
                with pytest.raises(RuntimeError, match="Neo4j unreachable at startup"):
                    with TestClient(app):
                        pass

    def test_runtime_error_message_contains_original_exception_text(self):
        mock_client = MagicMock()
        mock_client._driver = MagicMock()
        original_msg = "service unavailable"
        mock_client.verify_connectivity.side_effect = Exception(original_msg)
        with patch("app.lifespan.Neo4jClient", return_value=mock_client):
            with patch("app.lifespan.SentenceTransformerProvider"):
                app = _make_app()
                with pytest.raises(RuntimeError) as exc_info:
                    with TestClient(app):
                        pass
                assert original_msg in str(exc_info.value)


class TestLifespanShutdown:
    """driver.close() is called when the TestClient context exits."""

    def test_driver_close_called_on_shutdown(self):
        mock_client, mock_driver = _mock_client()
        with patch("app.lifespan.Neo4jClient", return_value=mock_client):
            with patch("app.lifespan.SentenceTransformerProvider"):
                with patch("app.lifespan.TemplateGenerationProvider"):
                    app = _make_app()
                    with TestClient(app):
                        pass
        mock_driver.close.assert_called_once()
        assert mock_driver.close.call_count == 1
