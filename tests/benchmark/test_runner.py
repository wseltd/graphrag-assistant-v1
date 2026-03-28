"""Tests for app/benchmark/runner.py (T032.c).

Test breakdown (14 total):
  - 2 keyword overlap edge cases      (all stop-words, empty answer)
  - 2 citation coverage with dupes    (same chunk_id repeated N times)
  - 2 citation coverage zero matches  (no overlap, empty list)
  - 2 partial keyword match threshold (0.5 and 2/3 ratios)
  - 2 integration: full-run count     (entry count, required keys)
  - 2 integration: run_id / file      (hex format, file exists and matches)
  - 2 happy-path                      (POST analogue structure, GET analogue file load)
"""
from __future__ import annotations

import json
import re
import time

import pytest

from app.benchmark.runner import run_benchmark
from app.benchmark.scoring import score_citation_coverage, score_keyword_overlap

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _stub_fn(answer: str = "result", chunk_ids: list[str] | None = None):
    """Return a callable that sleeps 1 ms to guarantee measurable latency."""
    citations = [{"chunk_id": cid} for cid in (chunk_ids or [])]

    def fn(query_text: str) -> dict:
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
    # When every expected token is a stop-word the effective token set is empty;
    # the denominator is clamped to 1, so the result is 0/1 = 0.0.
    result = score_keyword_overlap("the a an is in of to and", "the a an is in of")
    assert result == 0.0


def test_keyword_overlap_empty_actual_returns_zero() -> None:
    # No tokens in the actual answer → intersection is empty → 0.0 regardless
    # of how many meaningful tokens the expected answer contains.
    result = score_keyword_overlap("contract procurement supplier", "")
    assert result == 0.0


# ---------------------------------------------------------------------------
# Unit: citation coverage with duplicates
# ---------------------------------------------------------------------------


def test_citation_coverage_duplicate_chunk_ids_counted_once() -> None:
    # "c1" appears three times; coverage must use distinct chunk_ids only.
    citations = [{"chunk_id": "c1"}, {"chunk_id": "c1"}, {"chunk_id": "c1"}]
    result = score_citation_coverage(["c1", "c2"], citations)
    # Only c1 is covered out of {c1, c2} → 0.5, not 1.0 or 1.5.
    assert result == 0.5


def test_citation_coverage_all_dupes_multiple_unique_still_correct() -> None:
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
# Integration: full run entry count and key completeness
# ---------------------------------------------------------------------------


def test_run_benchmark_produces_correct_number_of_results(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = run_benchmark(
        _make_queries(20), _make_answers(20), _stub_fn(), _stub_fn()
    )
    assert len(result["query_results"]) == 20


def test_run_benchmark_all_entries_have_required_keys(
    tmp_path, monkeypatch
) -> None:
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
# Integration: run_id format and result file existence
# ---------------------------------------------------------------------------


def test_run_benchmark_run_id_is_32_char_hex(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = run_benchmark(
        _make_queries(1), _make_answers(1), _stub_fn(), _stub_fn()
    )
    run_id = result["run_id"]
    assert re.fullmatch(r"[0-9a-f]{32}", run_id), f"unexpected run_id: {run_id!r}"


def test_run_benchmark_result_file_exists_and_matches_return(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = run_benchmark(
        _make_queries(2), _make_answers(2), _stub_fn(), _stub_fn()
    )
    run_id = result["run_id"]
    file_path = tmp_path / "data" / "benchmark_results" / f"benchmark-{run_id}.json"
    assert file_path.exists(), f"expected file not found: {file_path}"
    loaded = json.loads(file_path.read_text(encoding="utf-8"))
    assert loaded["run_id"] == run_id
    assert len(loaded["query_results"]) == len(result["query_results"])


# ---------------------------------------------------------------------------
# Happy-path: POST analogue (structure) and GET analogue (file load)
# ---------------------------------------------------------------------------


def test_run_benchmark_returns_full_top_level_structure(
    tmp_path, monkeypatch
) -> None:
    """Analogous to POST /benchmark/run returning run_id, status, and summary."""
    monkeypatch.chdir(tmp_path)
    result = run_benchmark(
        _make_queries(3), _make_answers(3), _stub_fn(), _stub_fn()
    )
    assert set(result.keys()) == {"run_id", "timestamp", "query_results", "summary"}
    summary = result["summary"]
    for mode in ("plain_rag", "graph_rag"):
        assert mode in summary
        assert set(summary[mode].keys()) == {
            "mean_accuracy",
            "mean_citation_coverage",
            "mean_latency",
        }


def test_run_benchmark_file_loadable_and_contains_all_query_results(
    tmp_path, monkeypatch
) -> None:
    """Analogous to GET /benchmark/results/{run_id} returning all entries."""
    monkeypatch.chdir(tmp_path)
    queries = _make_queries(5)
    answers = _make_answers(5)
    result = run_benchmark(queries, answers, _stub_fn(), _stub_fn())
    run_id = result["run_id"]
    result_path = tmp_path / "data" / "benchmark_results" / f"benchmark-{run_id}.json"
    loaded = json.loads(result_path.read_text(encoding="utf-8"))
    assert loaded["run_id"] == run_id
    assert len(loaded["query_results"]) == 5
    assert all("query_id" in e for e in loaded["query_results"])
