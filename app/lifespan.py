"""FastAPI lifespan hook: startup and teardown of shared application resources."""
from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from graphrag_assistant.graph.neo4j_client import Neo4jClient
from graphrag_assistant.providers.embedding import SentenceTransformerProvider
from graphrag_assistant.providers.generation_stub import TemplateGenerationProvider

logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise shared resources at startup; release them at shutdown.

    Stores ``neo4j_driver``, ``embedding_provider``, and
    ``generation_provider`` on ``app.state`` for the duration of the
    application lifetime.

    Raises:
        RuntimeError: when Neo4j is unreachable at startup, preventing the
            server from accepting traffic in a broken state.
    """
    client = Neo4jClient(
        uri=os.environ.get("GRAPHRAG_NEO4J_URI", "bolt://localhost:7687"),
        user=os.environ.get("GRAPHRAG_NEO4J_USER", "neo4j"),
        password=os.environ.get("GRAPHRAG_NEO4J_PASSWORD", "test"),
    )
    driver = client._driver
    try:
        try:
            client.verify_connectivity()
        except Exception as exc:
            raise RuntimeError(
                f"Neo4j unreachable at startup ({type(exc).__name__}): {exc}"
            ) from exc
        app.state.neo4j_driver = driver
        app.state.embedding_provider = SentenceTransformerProvider()
        app.state.generation_provider = TemplateGenerationProvider()
        logger.info("Application startup complete")
        yield
    finally:
        driver.close()
        logger.info("Neo4j driver closed")
