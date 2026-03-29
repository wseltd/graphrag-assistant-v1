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


# ---------------------------------------------------------------------------
# Tests for _graph_rag_fn output structure (3 required by T009)
# ---------------------------------------------------------------------------

from app.pipelines.citation_generator import Citation, GenerationResult  # noqa: E402
from graphrag_assistant.schemas import AnswerSchema, RetrievalDebug, TextCitation  # noqa: E402


def _make_plain_answer() -> AnswerSchema:
    return AnswerSchema(
        answer="plain answer",
        graph_evidence=[],
        text_citations=[TextCitation(doc_id="doc1", chunk_id="ch1", quote="text excerpt")],
        retrieval_debug=RetrievalDebug(
            graph_query=None,
            entity_matches=[],
            retrieved_node_ids=[],
            chunk_ids=["ch1"],
            timings={},
        ),
        mode="plain_rag",
    )


def _make_graph_gen_result() -> GenerationResult:
    return GenerationResult(
        answer="graph answer",
        text_citations=[Citation(chunk_id="ch1", quote="some text", doc_id="doc1")],
    )


def _capture_fn_results(app: FastAPI) -> tuple[dict, dict]:
    """Post to /benchmark/run; spy on run_benchmark to capture fn outputs.

    Both _plain_rag_fn and _graph_rag_fn are called inside the spy while all
    patches are still active, ensuring the patched run_graph_rag and
    PlainRagPipeline are in effect.
    """
    captured: dict = {}

    mock_pipeline_inst = MagicMock()
    mock_pipeline_inst.execute.return_value = _make_plain_answer()

    mock_gen_result = _make_graph_gen_result()

    def _spy(queries, answers, plain_rag_fn, graph_rag_fn):
        captured["plain_result"] = plain_rag_fn("what is X?")
        captured["graph_result"] = graph_rag_fn("what is X?")
        return {
            "run_id": "abcd1234abcd1234",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "query_results": [],
            "summary": {},
        }

    with (
        patch("app.benchmark.router.load_benchmark_data", return_value=([], [])),
        patch("app.benchmark.router.run_benchmark", side_effect=_spy),
        patch("app.benchmark.router.save_result", return_value="out.json"),
        patch("app.benchmark.router.run_graph_rag", return_value=mock_gen_result),
        patch("app.benchmark.router.PlainRagPipeline", return_value=mock_pipeline_inst),
    ):
        client = TestClient(app)
        resp = client.post("/benchmark/run", headers=_HEADERS)

    assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"
    return captured["plain_result"], captured["graph_result"]


def test_graph_rag_fn_result_top_level_keys_match_plain_rag_fn() -> None:
    # Both callables must return dicts with exactly the same top-level keys so
    # run_benchmark and scoring functions can consume them identically.
    app = FastAPI()
    app.include_router(router)
    app.state.neo4j_driver = MagicMock()
    app.state.embedding_provider = MagicMock()
    app.state.generation_provider = MagicMock()

    plain_result, graph_result = _capture_fn_results(app)

    assert set(plain_result.keys()) == set(graph_result.keys())


def test_graph_rag_fn_text_citation_has_doc_id_field() -> None:
    # doc_id must be present in every text_citations item so the benchmark
    # output shape is structurally identical to plain_rag.
    app = FastAPI()
    app.include_router(router)
    app.state.neo4j_driver = MagicMock()
    app.state.embedding_provider = MagicMock()
    app.state.generation_provider = MagicMock()

    _, graph_result = _capture_fn_results(app)

    assert len(graph_result["text_citations"]) > 0
    for citation in graph_result["text_citations"]:
        assert "doc_id" in citation, f"citation missing doc_id: {citation}"


def test_graph_rag_fn_text_citation_has_quote_not_excerpt() -> None:
    # Regression: _graph_rag_fn previously used the old field name `excerpt`.
    # After Bug 5 renamed Citation.excerpt → Citation.quote, the output key
    # must be `quote` to match TextCitation and plain_rag text_citations shape.
    app = FastAPI()
    app.include_router(router)
    app.state.neo4j_driver = MagicMock()
    app.state.embedding_provider = MagicMock()
    app.state.generation_provider = MagicMock()

    _, graph_result = _capture_fn_results(app)

    assert len(graph_result["text_citations"]) > 0
    for citation in graph_result["text_citations"]:
        assert "quote" in citation, f"citation missing quote: {citation}"
        assert "excerpt" not in citation, f"citation should not have excerpt: {citation}"


def test_graph_rag_fn_retrieval_debug_populated_from_pipeline_stages() -> None:
    # Regression: _graph_rag_fn previously hardcoded empty lists for
    # entity_matches, retrieved_node_ids, and chunk_ids.  After the fix those
    # fields must be populated from the real pipeline stage outputs.
    from app.pipelines.entity_resolver import EntityMatch
    from app.pipelines.graph_traversal import GraphTraversalResult

    app = FastAPI()
    app.include_router(router)
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__ = lambda s: mock_session
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
    app.state.neo4j_driver = mock_driver
    app.state.embedding_provider = MagicMock()
    app.state.generation_provider = MagicMock()

    captured: dict = {}
    mock_entities = [
        EntityMatch(node_id="node-1", label="Company", name="Acme Corp", score=1.0),
        EntityMatch(node_id="node-2", label="Contract", name="Contract A", score=0.8),
    ]
    mock_traversal = GraphTraversalResult(chunk_ids=["ck-1", "ck-2"], triples=[])
    mock_gen_result = _make_graph_gen_result()

    def _spy(queries, answers, plain_rag_fn, graph_rag_fn):
        captured["graph_result"] = graph_rag_fn("Acme Corp contract terms")
        return {
            "run_id": "deadbeef00000000",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "query_results": [],
            "summary": {},
        }

    with (
        patch("app.benchmark.router.load_benchmark_data", return_value=([], [])),
        patch("app.benchmark.router.run_benchmark", side_effect=_spy),
        patch("app.benchmark.router.save_result", return_value="out.json"),
        patch("app.benchmark.router.run_graph_rag", return_value=mock_gen_result),
        patch("app.benchmark.router.resolve_entities", return_value=mock_entities),
        patch("app.benchmark.router.traverse_from_anchors", return_value=mock_traversal),
        patch("app.benchmark.router.retrieve_constrained", return_value=[]),
    ):
        client = TestClient(app)
        resp = client.post("/benchmark/run", headers=_HEADERS)

    assert resp.status_code == 202
    debug = captured["graph_result"]["retrieval_debug"]
    # entity_matches and retrieved_node_ids come from resolve_entities node IDs
    assert debug["entity_matches"] == ["node-1", "node-2"]
    assert debug["retrieved_node_ids"] == ["node-1", "node-2"]
    # chunk_ids come from traversal.chunk_ids
    assert debug["chunk_ids"] == ["ck-1", "ck-2"]
