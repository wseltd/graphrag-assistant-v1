"""FastAPI application factory (T026.c)."""
from __future__ import annotations

from fastapi import FastAPI

from app.benchmark.router import router as benchmark_router
from app.lifespan import app_lifespan
from app.routers.entities import router as entities_router
from app.routers.health import router as health_router
from app.routers.ingest import router as ingest_router
from app.routers.query import router as query_router
from app.routers.seed import router as seed_router


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI application.

    Registers all routers and attaches the application lifespan hook.
    """
    _app = FastAPI(lifespan=app_lifespan)

    _app.include_router(health_router)
    _app.include_router(seed_router)
    _app.include_router(entities_router)
    _app.include_router(benchmark_router)
    _app.include_router(ingest_router, prefix="/api/v1")
    _app.include_router(query_router, prefix="/api/v1")

    return _app


app = create_app()
