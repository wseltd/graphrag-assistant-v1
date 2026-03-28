"""Benchmark persistence module.

T032.d — file-based save_result/load_result + in-memory store_result/get_result.
T033   — atomic BenchmarkStore with Pydantic schema and typed NotFoundError.

Design notes
------------
- BenchmarkStore path pattern: {runs_dir}/{run_id}.json
- Atomic write: write to a .tmp sibling, then os.rename so a crashed mid-write
  never corrupts an existing file for that run_id.
- BenchmarkStore is the single public class for the T033 API; callers never
  touch the filesystem directly.
- The T032.d module-level functions (save_result / load_result / store_result /
  get_result) are retained for backward compatibility with T032.d tests.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# T032.d — in-memory + file helpers
# ---------------------------------------------------------------------------

_in_memory_store: dict[str, dict] = {}


def save_result(run_id: str, result: dict, output_dir: str = "data/benchmark_results") -> str:
    """Write result to {output_dir}/benchmark-{run_id}.json.

    Returns the path of the written file as a string.
    """
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"benchmark-{run_id}.json"
    file_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return str(file_path)


def load_result(run_id: str, output_dir: str = "data/benchmark_results") -> dict:
    """Return parsed JSON for run_id from output_dir.

    Raises:
        FileNotFoundError: If no file exists for run_id.
    """
    file_path = Path(output_dir) / f"benchmark-{run_id}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"No benchmark result found for run_id={run_id!r}")
    return json.loads(file_path.read_text(encoding="utf-8"))


def store_result(run_id: str, result: dict) -> None:
    """Store result in the module-level in-memory dict keyed by run_id."""
    _in_memory_store[run_id] = result


def get_result(run_id: str) -> dict | None:
    """Return the stored result for run_id, or None if not present."""
    return _in_memory_store.get(run_id)


# ---------------------------------------------------------------------------
# T033 — Pydantic schema for a benchmark run result
# ---------------------------------------------------------------------------


class QueryScores(BaseModel):
    """Scores for one query under one pipeline mode."""

    accuracy: float
    citation_coverage: float
    latency_seconds: float

    def __repr__(self) -> str:
        return (
            f"QueryScores(accuracy={self.accuracy!r}, "
            f"citation_coverage={self.citation_coverage!r}, "
            f"latency_seconds={self.latency_seconds!r})"
        )


class QueryResult(BaseModel):
    """Per-query benchmark entry with scores for both pipeline modes."""

    query_id: str
    query: str
    plain_rag: QueryScores
    graph_rag: QueryScores

    def __repr__(self) -> str:
        return f"QueryResult(query_id={self.query_id!r})"


class PipelineSummary(BaseModel):
    """Aggregate metrics for one pipeline mode across all queries."""

    mean_accuracy: float
    mean_citation_coverage: float
    mean_latency: float

    def __repr__(self) -> str:
        return (
            f"PipelineSummary(mean_accuracy={self.mean_accuracy!r}, "
            f"mean_citation_coverage={self.mean_citation_coverage!r})"
        )


class RunSummary(BaseModel):
    """Summary section of a benchmark run result."""

    plain_rag: PipelineSummary
    graph_rag: PipelineSummary

    def __repr__(self) -> str:
        return f"RunSummary(plain_rag={self.plain_rag!r}, graph_rag={self.graph_rag!r})"


class BenchmarkRunResult(BaseModel):
    """Full persisted record for one benchmark run."""

    run_id: str
    timestamp: str
    query_results: list[QueryResult]
    summary: RunSummary

    def __repr__(self) -> str:
        return (
            f"BenchmarkRunResult(run_id={self.run_id!r}, "
            f"n_results={len(self.query_results)})"
        )


# ---------------------------------------------------------------------------
# T033 — typed exception
# ---------------------------------------------------------------------------


class NotFoundError(Exception):
    """Raised by BenchmarkStore.load when run_id does not exist."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"No benchmark result found for run_id={run_id!r}")
        self.run_id = run_id

    def __repr__(self) -> str:
        return f"NotFoundError(run_id={self.run_id!r})"


# ---------------------------------------------------------------------------
# T033 — BenchmarkStore: atomic read / write
# ---------------------------------------------------------------------------

_DEFAULT_RUNS_DIR = Path("data/benchmark_results") / "benchmark_runs"


class BenchmarkStore:
    """Atomic, file-backed store for BenchmarkRunResult objects.

    Each result is stored as JSON at {runs_dir}/{run_id}.json.
    Writes are atomic: the payload is written to a .tmp sibling then renamed
    into place, so an interrupted write never corrupts an existing file.
    """

    def __init__(self, runs_dir: Path | str = _DEFAULT_RUNS_DIR) -> None:
        self._runs_dir = Path(runs_dir)

    def __repr__(self) -> str:
        return f"BenchmarkStore(runs_dir={self._runs_dir!r})"

    def save(self, run_id: str, result: BenchmarkRunResult) -> None:
        """Persist result under run_id, overwriting any previous entry atomically.

        The runs directory is created on first call if it does not exist.
        """
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        target = self._runs_dir / f"{run_id}.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        tmp.rename(target)
        logger.debug("BenchmarkStore: saved run_id=%r to %s", run_id, target)

    def load(self, run_id: str) -> BenchmarkRunResult:
        """Return the BenchmarkRunResult for run_id.

        Raises:
            NotFoundError: If no file exists for run_id.
        """
        path = self._runs_dir / f"{run_id}.json"
        if not path.exists():
            raise NotFoundError(run_id)
        return BenchmarkRunResult.model_validate_json(path.read_text(encoding="utf-8"))
