"""Benchmark runner module (T032.c).

Executes all benchmark queries sequentially against both pipeline functions,
records per-query scores and latencies, writes a result file, and returns a
structured summary dict.

Design notes
------------
- Sequential execution only — no threads or async. Benchmark integrity requires
  predictable, comparable wall-clock latencies; parallelism would invalidate
  those measurements.
- plain_rag_fn and graph_rag_fn may return a plain dict *or* a Pydantic model
  with a .model_dump() method. _to_dict() normalises both before scoring.
- The result file is written to data/benchmark_results/benchmark-{run_id}.json
  and can be retrieved by run_id.
"""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from app.benchmark.scoring import score_query


def _to_dict(result: object) -> dict:
    """Normalise a pipeline result to a plain dict for scoring."""
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return dict(result)  # type: ignore[call-overload]


def run_benchmark(
    queries: list[dict],
    answers: list[dict],
    plain_rag_fn: Callable,
    graph_rag_fn: Callable,
) -> dict:
    """Run all queries against both pipelines and return scored results.

    Iterates queries sequentially (no parallelism). For each query:
      1. Calls plain_rag_fn(query_text) with wall-clock timing.
      2. Calls graph_rag_fn(query_text) with wall-clock timing.
      3. Scores each result via score_query.

    After all queries, computes summary means and writes the result to
    data/benchmark_results/benchmark-{run_id}.json.

    Args:
        queries:       List of query dicts with at least ``query_id`` and
                       ``query`` keys.
        answers:       Parallel list of expected-answer dicts with at least
                       ``query_id``, ``answer``, and ``chunk_ids`` keys.
        plain_rag_fn:  Callable accepting a query string and returning a dict
                       (or Pydantic model) with ``answer`` and
                       ``text_citations`` fields.
        graph_rag_fn:  Same signature as plain_rag_fn.

    Returns:
        dict with keys:
          run_id         — 32-character lowercase hex UUID.
          timestamp      — ISO 8601 UTC string.
          query_results  — list of per-query score dicts.
          summary        — mean accuracy, citation coverage, and latency for
                           each pipeline mode.
    """
    query_results: list[dict] = []

    for query_entry, answer_entry in zip(queries, answers, strict=False):
        query_text = query_entry["query"]
        query_id = query_entry["query_id"]

        t0 = time.perf_counter()
        plain_result = _to_dict(plain_rag_fn(query_text))
        plain_latency = time.perf_counter() - t0

        t0 = time.perf_counter()
        graph_result = _to_dict(graph_rag_fn(query_text))
        graph_latency = time.perf_counter() - t0

        plain_scores = score_query(answer_entry, plain_result, plain_latency)
        graph_scores = score_query(answer_entry, graph_result, graph_latency)

        query_results.append(
            {
                "query_id": query_id,
                "query": query_text,
                "plain_rag": plain_scores,
                "graph_rag": graph_scores,
            }
        )

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    run_id = uuid.uuid4().hex
    timestamp = datetime.now(UTC).isoformat()

    output = {
        "run_id": run_id,
        "timestamp": timestamp,
        "query_results": query_results,
        "summary": {
            "plain_rag": {
                "mean_accuracy": _mean(
                    [r["plain_rag"]["accuracy"] for r in query_results]
                ),
                "mean_citation_coverage": _mean(
                    [r["plain_rag"]["citation_coverage"] for r in query_results]
                ),
                "mean_latency": _mean(
                    [r["plain_rag"]["latency_seconds"] for r in query_results]
                ),
            },
            "graph_rag": {
                "mean_accuracy": _mean(
                    [r["graph_rag"]["accuracy"] for r in query_results]
                ),
                "mean_citation_coverage": _mean(
                    [r["graph_rag"]["citation_coverage"] for r in query_results]
                ),
                "mean_latency": _mean(
                    [r["graph_rag"]["latency_seconds"] for r in query_results]
                ),
            },
        },
    }

    out_path = Path("data/benchmark_results") / f"benchmark-{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    return output
