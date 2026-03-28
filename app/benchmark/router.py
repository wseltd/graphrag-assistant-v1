"""Benchmark FastAPI router (T032.e).

Exposes two endpoints:
  POST /benchmark/run           — run all benchmark queries, return summary.
  GET  /benchmark/results/{id}  — retrieve a stored run result.

Authentication
--------------
X-Api-Key request header via FastAPI's APIKeyHeader Security scheme.
The key is validated against the API_KEY environment variable
(comma-separated for multiple keys).

CSRF mitigation
---------------
State-changing endpoints (POST) apply strict Origin validation: if an
Origin header is present the origin host must match the request Host header,
rejecting cross-site browser-initiated requests.  Direct API/CLI callers
(no Origin header) are unaffected.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.benchmark.loader import load_benchmark_data
from app.benchmark.runner import run_benchmark
from app.benchmark.store import get_result, save_result, store_result
from app.dependencies import require_api_key, verify_csrf
from app.pipelines.graph_rag import run_graph_rag
from app.pipelines.plain_rag import PlainRagPipeline

logger = logging.getLogger(__name__)

router = APIRouter()

# Maps run_id → the API key that created it, used for ownership checks.
_run_owners: dict[str, str] = {}

# Cypher for vector search used by the graph-RAG callable — matches the
# chunk_embedding_idx created by neo4j_client.bootstrap_schema.
_VECTOR_QUERY = (
    "CALL db.index.vector.queryNodes('chunk_embedding_idx', $top_k, $query_vector) "
    "YIELD node AS chunk, score "
    "RETURN chunk.chunk_id AS chunk_id, "
    "chunk.contract_id AS contract_id, "
    "chunk.text AS text, "
    "score "
    "ORDER BY score DESC"
)


class _Neo4jVectorStore:
    """Adapter: embeds a query string and searches the Neo4j vector index.

    Provides the vector_store.search(query, top_k) interface expected by
    run_graph_rag without duplicating embedding logic outside the pipeline.
    """

    def __init__(self, embedding_provider: Any, driver: Any) -> None:
        self._embedding_provider = embedding_provider
        self._driver = driver

    def search(self, query: str, top_k: int) -> list[dict]:
        """Return up to top_k chunks by cosine similarity to query."""
        vector = self._embedding_provider.embed([query])[0]
        with self._driver.session() as session:
            rows = session.run(
                _VECTOR_QUERY,
                {"top_k": top_k, "query_vector": vector},
            )
            return [
                {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["contract_id"] or "",
                    "text": row["text"],
                    "score": float(row["score"]),
                }
                for row in rows
            ]


@router.post("/benchmark/run", status_code=status.HTTP_202_ACCEPTED)
async def run_benchmark_endpoint(
    request: Request,
    api_key: str = Depends(require_api_key),
    _csrf: None = Depends(verify_csrf),
) -> dict:
    """Run all benchmark queries against both pipeline modes.

    Loads benchmark data, executes plain_rag and graph_rag pipelines for
    every query, stores results in memory and writes a JSON file to disk.

    Authentication: X-Api-Key header validated against API_KEY env var.
    Missing header → 422.  Invalid key → 401.

    Returns:
        dict with run_id, status, result_count, output_file.

    Raises:
        HTTP 422: X-Api-Key header absent.
        HTTP 401: key present but not in allowed set.
    """
    queries, answers = load_benchmark_data()

    embedding_provider = request.app.state.embedding_provider
    generation_provider = request.app.state.generation_provider
    driver = request.app.state.neo4j_driver
    vector_store = _Neo4jVectorStore(embedding_provider, driver)

    # Create once; execute() is stateless so the instance is safe to reuse.
    pipeline = PlainRagPipeline(
        embedding_provider=embedding_provider,
        generation_provider=generation_provider,
        driver=driver,
    )

    def _plain_rag_fn(query: str) -> dict:
        return pipeline.execute(query).model_dump()

    def _graph_rag_fn(query: str) -> dict:
        with driver.session() as session:
            result = run_graph_rag(query, session, vector_store)
        return {
            "answer": result.answer,
            "text_citations": [
                {"chunk_id": c.chunk_id, "excerpt": c.excerpt}
                for c in result.text_citations
            ],
        }

    result = run_benchmark(queries, answers, _plain_rag_fn, _graph_rag_fn)
    run_id = result["run_id"]

    store_result(run_id, result)
    _run_owners[run_id] = api_key
    output_file = save_result(run_id, result)

    logger.info(
        "benchmark run %s complete: %d queries",
        run_id,
        len(result["query_results"]),
    )

    return {
        "run_id": run_id,
        "status": "completed",
        "result_count": len(result["query_results"]),
        "output_file": output_file,
    }


@router.get("/benchmark/results/{run_id}", status_code=status.HTTP_200_OK)
async def get_benchmark_results(
    run_id: str,
    api_key: str = Depends(require_api_key),
) -> dict:
    """Return the stored benchmark result for run_id.

    Authentication: X-Api-Key header validated against API_KEY env var.
    Ownership: the requesting key must be the one that created this run.

    Returns:
        Full benchmark result dict from the in-memory store.

    Raises:
        HTTP 422: X-Api-Key header absent.
        HTTP 401: key present but not in allowed set.
        HTTP 403: run exists but was created by a different API key.
        HTTP 404: no result found for run_id.
    """
    result = get_result(run_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No benchmark result found for run_id={run_id!r}",
        )
    if _run_owners.get(run_id) != api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: run_id does not belong to this API key",
        )
    return result
