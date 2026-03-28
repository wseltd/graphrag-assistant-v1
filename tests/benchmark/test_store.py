"""Tests for app/benchmark/store.py (T032.d).

Test breakdown (14 total):
  - 2 keyword overlap edge cases      (all stop-words, empty answer)
  - 2 citation coverage with dupes    (same chunk_id repeated N times)
  - 2 citation coverage zero matches  (no overlap, empty citations)
  - 2 partial keyword match threshold (0.5 and 2/3 ratios)
  - 2 integration: full run count     (20 entries, required keys present)
  - 2 integration: run_id / file      (save_result path, load_result round-trip)
  - 2 happy-path                      (store_result+get_result, FileNotFoundError on miss)
"""
from __future__ import annotations

import time

import pytest

from app.benchmark.runner import run_benchmark
from app.benchmark.scoring import score_citation_coverage, score_keyword_overlap
from app.benchmark.store import (
    _in_memory_store,
    get_result,
    load_result,
    save_result,
    store_result,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_fn(answer: str = "result", chunk_ids: list[str] | None = None):
    citations = [{"chunk_id": cid} for cid in (chunk_ids or [])]

    def fn(query_text: str) -> dict:  # noqa: ARG001
        time.sleep(0.001)
        return {"answer": answer, "text_citations": citations}

    return fn


def _make_queries(n: int) -> list[dict]:
    return [{"query_id": f"q{i}", "query": f"question {i}"} for i in range(n)]


def _make_answers(n: int) -> list[dict]:
    return [
        {
            "query_id": f"q{i}",
            "answer": f"procurement contract {i}",
            "chunk_ids": [f"c{i}"],
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Unit: keyword overlap edge cases
# ---------------------------------------------------------------------------


def test_keyword_overlap_all_stop_words_returns_zero() -> None:
    # Every expected token is a stop-word; effective token set is empty;
    # denominator is clamped to 1, so result is 0/1 = 0.0.
    result = score_keyword_overlap("the a an is in of to and", "the a an is in of")
    assert result == 0.0


def test_keyword_overlap_empty_actual_returns_zero() -> None:
    # No tokens in actual → intersection is empty → 0.0 regardless of expected.
    result = score_keyword_overlap("contract procurement supplier", "")
    assert result == 0.0


# ---------------------------------------------------------------------------
# Unit: citation coverage with duplicates
# ---------------------------------------------------------------------------


def test_citation_coverage_duplicate_chunk_ids_counted_once() -> None:
    # "c1" appears three times; coverage must use distinct chunk_ids only.
    citations = [{"chunk_id": "c1"}, {"chunk_id": "c1"}, {"chunk_id": "c1"}]
    result = score_citation_coverage(["c1", "c2"], citations)
    # Only c1 covered out of {c1, c2} → 0.5, not 1.0.
    assert result == 0.5


def test_citation_coverage_all_dupes_all_covered() -> None:
    # c1 × 5 and c2 × 5; expected = {c1, c2} → both covered → 1.0.
    citations = [{"chunk_id": "c1"}] * 5 + [{"chunk_id": "c2"}] * 5
    result = score_citation_coverage(["c1", "c2"], citations)
    assert result == 1.0


# ---------------------------------------------------------------------------
# Unit: citation coverage zero matches
# ---------------------------------------------------------------------------


def test_citation_coverage_zero_matches_disjoint_ids() -> None:
    citations = [{"chunk_id": "c3"}, {"chunk_id": "c4"}]
    result = score_citation_coverage(["c1", "c2"], citations)
    assert result == 0.0


def test_citation_coverage_zero_matches_empty_citations() -> None:
    result = score_citation_coverage(["c1", "c2"], [])
    assert result == 0.0


# ---------------------------------------------------------------------------
# Unit: partial keyword match threshold
# ---------------------------------------------------------------------------


def test_partial_keyword_match_exactly_half() -> None:
    # "contract" matches; "supplier" does not → 1/2.
    result = score_keyword_overlap("contract supplier", "contract weather")
    assert pytest.approx(result, abs=1e-9) == 0.5


def test_partial_keyword_match_two_thirds() -> None:
    # "contract" and "supplier" match; "procurement" does not → 2/3.
    result = score_keyword_overlap(
        "contract supplier procurement", "contract supplier weather"
    )
    assert pytest.approx(result, abs=1e-9) == 2 / 3


# ---------------------------------------------------------------------------
# Integration: full run produces 20 results
# ---------------------------------------------------------------------------


def test_run_benchmark_produces_20_results(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = run_benchmark(
        _make_queries(20), _make_answers(20), _stub_fn(), _stub_fn()
    )
    assert len(result["query_results"]) == 20


def test_run_benchmark_entries_have_required_keys(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = run_benchmark(
        _make_queries(20), _make_answers(20), _stub_fn(), _stub_fn()
    )
    score_keys = {"accuracy", "citation_coverage", "latency_seconds"}
    for entry in result["query_results"]:
        assert {"query_id", "query", "plain_rag", "graph_rag"} == set(entry.keys())
        assert score_keys == set(entry["plain_rag"].keys())
        assert score_keys == set(entry["graph_rag"].keys())
        assert entry["plain_rag"]["latency_seconds"] > 0
        assert entry["graph_rag"]["latency_seconds"] > 0


# ---------------------------------------------------------------------------
# Integration: save_result / load_result file existence and round-trip
# ---------------------------------------------------------------------------


def test_save_result_writes_file_at_expected_path(tmp_path) -> None:
    run_id = "abc123"
    data = {"run_id": run_id, "query_results": []}
    returned_path = save_result(run_id, data, output_dir=str(tmp_path))
    expected = tmp_path / f"benchmark-{run_id}.json"
    assert expected.exists()
    assert returned_path == str(expected)


def test_load_result_round_trips_saved_data(tmp_path) -> None:
    run_id = "def456"
    data = {"run_id": run_id, "query_results": [{"query_id": "q0"}]}
    save_result(run_id, data, output_dir=str(tmp_path))
    loaded = load_result(run_id, output_dir=str(tmp_path))
    assert loaded == data


# ---------------------------------------------------------------------------
# Happy-path: in-memory store round-trip and unknown key returns None
# ---------------------------------------------------------------------------


def test_store_result_and_get_result_round_trip() -> None:
    run_id = "happy-post-1"
    data = {"run_id": run_id, "summary": {"plain_rag": {}, "graph_rag": {}}}
    _in_memory_store.pop(run_id, None)
    store_result(run_id, data)
    assert get_result(run_id) == data


def test_load_result_raises_file_not_found_for_unknown(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_result("nonexistent-run-id", output_dir=str(tmp_path))
