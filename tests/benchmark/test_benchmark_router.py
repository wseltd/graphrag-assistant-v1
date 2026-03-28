"""Tests for app.benchmark.router — POST /benchmark/run and GET /benchmark/results/{run_id}.

Test plan (14 tests):
  Unit (scoring edge cases): 8
    - 2 keyword overlap: all stop-words, empty answer
    - 2 citation coverage with duplicates
    - 2 citation coverage zero matches
    - 2 partial keyword match threshold
  Integration (full run via mocked router): 4
    - full run produces 20 result entries
    - every entry has required fields
    - GET /benchmark/results/{run_id} retrieves stored result
    - result file is written after run
  Happy-path HTTP (endpoint smoke): 2
    - POST /benchmark/run → 202 with run_id and status
    - GET /benchmark/results/{run_id} → 200 with run result
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.benchmark.router import _run_owners, router
from app.benchmark.scoring import score_citation_coverage, score_keyword_overlap
from app.benchmark.store import _in_memory_store

_VALID_KEY = "dev-key-change-in-prod"
_HEADERS = {"X-Api-Key": _VALID_KEY}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__ = lambda s: mock_session
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
    app.state.neo4j_driver = mock_driver
    app.state.embedding_provider = MagicMock()
    app.state.generation_provider = MagicMock()
    return app


def _make_mock_data(n: int = 20) -> tuple[list[dict], list[dict]]:
    queries = [{"query_id": f"q{i:02d}", "query": f"test question {i}"} for i in range(n)]
    answers = [
        {"query_id": f"q{i:02d}", "answer": f"expected answer {i}", "chunk_ids": [f"c{i}"]}
        for i in range(n)
    ]
    return queries, answers


def _make_canned_result(n: int = 20) -> dict:
    return {
        "run_id": "cafebabe01234567",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "query_results": [
            {
                "query_id": f"q{i:02d}",
                "query": f"test question {i}",
                "plain_rag": {
                    "accuracy": 0.5,
                    "citation_coverage": 0.5,
                    "latency_seconds": 0.1,
                },
                "graph_rag": {
                    "accuracy": 0.6,
                    "citation_coverage": 0.6,
                    "latency_seconds": 0.2,
                },
            }
            for i in range(n)
        ],
        "summary": {
            "plain_rag": {"mean_accuracy": 0.5, "mean_citation_coverage": 0.5, "mean_latency": 0.1},
            "graph_rag": {"mean_accuracy": 0.6, "mean_citation_coverage": 0.6, "mean_latency": 0.2},
        },
    }


# ---------------------------------------------------------------------------
# Unit tests — keyword overlap edge cases (2)
# ---------------------------------------------------------------------------


def test_keyword_overlap_all_stop_words_returns_zero() -> None:
    # When all expected tokens are stop-words the token set is empty;
    # denominator is clamped to 1, so the result must be 0.0.
    result = score_keyword_overlap("the a an is in of to and or for", "contract procurement")
    assert result == 0.0


def test_keyword_overlap_empty_answer_returns_zero() -> None:
    result = score_keyword_overlap("procurement contract supplier", "")
    assert result == 0.0


# ---------------------------------------------------------------------------
# Unit tests — citation coverage with duplicates (2)
# ---------------------------------------------------------------------------


def test_citation_coverage_duplicates_counted_once() -> None:
    # "c1" appears twice in text_citations; it should count only once.
    citations = [{"chunk_id": "c1"}, {"chunk_id": "c1"}, {"chunk_id": "c2"}]
    result = score_citation_coverage(["c1", "c2"], citations)
    assert result == 1.0


def test_citation_coverage_all_same_id_counts_as_one_match() -> None:
    # Three citations with the same chunk_id still count as one hit against
    # two expected chunk_ids → coverage = 0.5.
    citations = [{"chunk_id": "c1"}, {"chunk_id": "c1"}, {"chunk_id": "c1"}]
    result = score_citation_coverage(["c1", "c2"], citations)
    assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Unit tests — citation coverage zero matches (2)
# ---------------------------------------------------------------------------


def test_citation_coverage_zero_matches_no_overlap() -> None:
    citations = [{"chunk_id": "x1"}, {"chunk_id": "x2"}]
    result = score_citation_coverage(["c1", "c2"], citations)
    assert result == 0.0


def test_citation_coverage_zero_matches_empty_citations() -> None:
    result = score_citation_coverage(["c1", "c2"], [])
    assert result == 0.0


# ---------------------------------------------------------------------------
# Unit tests — partial keyword match threshold (2)
# ---------------------------------------------------------------------------


def test_partial_keyword_match_one_third() -> None:
    # "contract supplier procurement" → 3 tokens; actual has only "contract"
    # → overlap = 1/3.
    result = score_keyword_overlap("contract supplier procurement", "contract")
    assert result == pytest.approx(1 / 3, abs=1e-6)


def test_partial_keyword_match_half() -> None:
    # "contract supplier" → 2 tokens; actual has "contract" only → 0.5.
    result = score_keyword_overlap("contract supplier", "contract")
    assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Integration tests (4) — router calls patched; no live Neo4j
# ---------------------------------------------------------------------------


def test_full_run_produces_20_result_entries() -> None:
    app = _build_test_app()
    mock_queries, mock_answers = _make_mock_data(20)
    canned = _make_canned_result(20)

    with (
        patch(
            "app.benchmark.router.load_benchmark_data",
            return_value=(mock_queries, mock_answers),
        ),
        patch("app.benchmark.router.run_benchmark", return_value=canned),
        patch(
            "app.benchmark.router.save_result",
            return_value="data/benchmark_results/benchmark-cafebabe01234567.json",
        ),
    ):
        client = TestClient(app)
        resp = client.post("/benchmark/run", headers=_HEADERS)

    assert resp.status_code == 202
    assert resp.json()["result_count"] == 20


def test_full_run_all_entries_have_required_fields() -> None:
    app = _build_test_app()
    mock_queries, mock_answers = _make_mock_data(20)
    canned = _make_canned_result(20)

    with (
        patch(
            "app.benchmark.router.load_benchmark_data",
            return_value=(mock_queries, mock_answers),
        ),
        patch("app.benchmark.router.run_benchmark", return_value=canned),
        patch(
            "app.benchmark.router.save_result",
            return_value="data/benchmark_results/benchmark-cafebabe01234567.json",
        ),
    ):
        client = TestClient(app)
        resp = client.post("/benchmark/run", headers=_HEADERS)

    run_id = resp.json()["run_id"]
    # Retrieve the stored result and verify every entry shape.
    stored = _in_memory_store.get(run_id)
    assert stored is not None
    assert len(stored["query_results"]) == 20
    for entry in stored["query_results"]:
        assert "query_id" in entry
        assert "query" in entry
        assert "plain_rag" in entry
        assert "graph_rag" in entry


def test_run_id_retrieval() -> None:
    app = _build_test_app()
    mock_queries, mock_answers = _make_mock_data(20)
    canned = _make_canned_result(20)

    with (
        patch(
            "app.benchmark.router.load_benchmark_data",
            return_value=(mock_queries, mock_answers),
        ),
        patch("app.benchmark.router.run_benchmark", return_value=canned),
        patch(
            "app.benchmark.router.save_result",
            return_value="data/benchmark_results/benchmark-cafebabe01234567.json",
        ),
    ):
        client = TestClient(app)
        post_resp = client.post("/benchmark/run", headers=_HEADERS)

    run_id = post_resp.json()["run_id"]
    get_resp = client.get(f"/benchmark/results/{run_id}", headers=_HEADERS)

    assert get_resp.status_code == 200
    assert get_resp.json()["run_id"] == run_id


def test_result_file_written(tmp_path: pytest.MonkeyPatch) -> None:
    app = _build_test_app()
    mock_queries, mock_answers = _make_mock_data(20)
    canned = _make_canned_result(20)
    run_id = canned["run_id"]

    written: dict[str, str] = {}

    def _fake_save(rid: str, result: dict, output_dir: str = "data/benchmark_results") -> str:
        path = tmp_path / f"benchmark-{rid}.json"  # type: ignore[operator]
        path.write_text(json.dumps(result), encoding="utf-8")
        written["path"] = str(path)
        return str(path)

    with (
        patch(
            "app.benchmark.router.load_benchmark_data",
            return_value=(mock_queries, mock_answers),
        ),
        patch("app.benchmark.router.run_benchmark", return_value=canned),
        patch("app.benchmark.router.save_result", side_effect=_fake_save),
    ):
        client = TestClient(app)
        resp = client.post("/benchmark/run", headers=_HEADERS)

    assert resp.status_code == 202
    assert "path" in written
    import pathlib

    assert pathlib.Path(written["path"]).exists()
    stored = json.loads(pathlib.Path(written["path"]).read_text(encoding="utf-8"))
    assert stored["run_id"] == run_id


# ---------------------------------------------------------------------------
# Happy-path tests (2) — endpoint smoke tests
# ---------------------------------------------------------------------------


def test_post_benchmark_run_returns_202() -> None:
    app = _build_test_app()
    mock_queries, mock_answers = _make_mock_data(20)
    canned = _make_canned_result(20)

    with (
        patch(
            "app.benchmark.router.load_benchmark_data",
            return_value=(mock_queries, mock_answers),
        ),
        patch("app.benchmark.router.run_benchmark", return_value=canned),
        patch(
            "app.benchmark.router.save_result",
            return_value="data/benchmark_results/benchmark-cafebabe01234567.json",
        ),
    ):
        client = TestClient(app)
        resp = client.post("/benchmark/run", headers=_HEADERS)

    assert resp.status_code == 202
    data = resp.json()
    assert "run_id" in data
    assert "status" in data
    assert data["status"] == "completed"


def test_get_benchmark_results_returns_200() -> None:
    app = _build_test_app()
    canned = _make_canned_result(20)
    run_id = canned["run_id"]
    # Pre-load the result and ownership record so GET can find and authorize it.
    _in_memory_store[run_id] = canned
    _run_owners[run_id] = _VALID_KEY

    client = TestClient(app)
    resp = client.get(f"/benchmark/results/{run_id}", headers=_HEADERS)

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == run_id
    assert len(data["query_results"]) == 20
