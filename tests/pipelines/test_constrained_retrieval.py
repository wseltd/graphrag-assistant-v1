"""Tests for app/pipelines/constrained_retrieval (T025.c).

Three test cases covering the constrained vector retrieval function:
  1. Chunks filtered to graph-adjacent chunk IDs only.
  2. Fallback: empty allowed_chunk_ids returns [] without querying.
  3. Ranking order is preserved (descending score) after the filter step.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.pipelines.constrained_retrieval import RankedChunk, retrieve_constrained

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_store(*results: dict) -> MagicMock:
    """Return a mock vector store whose search() returns *results*."""
    store = MagicMock()
    store.search.return_value = list(results)
    return store


def _chunk(chunk_id: str, text: str, score: float) -> dict:
    return {"chunk_id": chunk_id, "text": text, "score": score}


# ---------------------------------------------------------------------------
# Case 1: chunks filtered to graph-adjacent only
# ---------------------------------------------------------------------------


class TestConstrainedFiltering:
    """retrieve_constrained returns only chunks in allowed_chunk_ids."""

    def test_filters_to_allowed_chunk_ids(self) -> None:
        store = _make_store(
            _chunk("c1", "text one", 0.9),
            _chunk("c2", "text two", 0.8),
            _chunk("c3", "text three", 0.7),
            _chunk("c4", "text four", 0.6),
        )
        result = retrieve_constrained(
            query="test query",
            allowed_chunk_ids=["c1", "c3"],
            vector_store=store,
            top_k=5,
        )
        ids = [r.chunk_id for r in result]
        assert ids == ["c1", "c3"]

    def test_non_matching_chunks_excluded(self) -> None:
        store = _make_store(
            _chunk("c1", "text one", 0.9),
            _chunk("c2", "text two", 0.8),
        )
        result = retrieve_constrained(
            query="test query",
            allowed_chunk_ids=["c99"],
            vector_store=store,
        )
        assert result == []

    def test_result_items_are_ranked_chunk_instances(self) -> None:
        store = _make_store(_chunk("c1", "hello world", 0.95))
        result = retrieve_constrained(
            query="q",
            allowed_chunk_ids=["c1"],
            vector_store=store,
        )
        assert len(result) == 1
        chunk = result[0]
        assert isinstance(chunk, RankedChunk)
        assert chunk.chunk_id == "c1"
        assert chunk.text == "hello world"
        assert chunk.score == pytest.approx(0.95)

    def test_search_called_with_candidate_multiplier(self) -> None:
        store = _make_store()
        retrieve_constrained(
            query="q",
            allowed_chunk_ids=["c1"],
            vector_store=store,
            top_k=5,
        )
        # _CANDIDATE_MULTIPLIER = 3, so expected candidates_k = 5 * 3 = 15
        assert store.search.call_count == 1
        assert store.search.call_args == (("q", 15),)


# ---------------------------------------------------------------------------
# Case 2: fallback returns empty list when allowed_chunk_ids is empty
# ---------------------------------------------------------------------------


class TestEmptyAllowedFallback:
    """When allowed_chunk_ids is empty, returns [] without querying."""

    def test_empty_allowed_returns_empty_list(self) -> None:
        store = _make_store(_chunk("c1", "text one", 0.9))
        result = retrieve_constrained(
            query="some query",
            allowed_chunk_ids=[],
            vector_store=store,
        )
        assert result == []

    def test_empty_allowed_does_not_call_search(self) -> None:
        store = MagicMock()
        retrieve_constrained(
            query="some query",
            allowed_chunk_ids=[],
            vector_store=store,
        )
        assert store.search.call_count == 0

    def test_empty_allowed_returns_list_type(self) -> None:
        store = MagicMock()
        result = retrieve_constrained(
            query="q",
            allowed_chunk_ids=[],
            vector_store=store,
        )
        assert isinstance(result, list)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Case 3: ranking order is preserved after filter
# ---------------------------------------------------------------------------


class TestRankingOrder:
    """After filtering, results are sorted by descending score."""

    def test_ranking_order_preserved_after_filter(self) -> None:
        # Store returns chunks in ascending score order; result must be descending.
        store = _make_store(
            _chunk("c1", "low", 0.3),
            _chunk("c2", "mid", 0.6),
            _chunk("c3", "high", 0.9),
        )
        result = retrieve_constrained(
            query="q",
            allowed_chunk_ids=["c1", "c2", "c3"],
            vector_store=store,
        )
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self) -> None:
        store = _make_store(
            _chunk("c1", "a", 0.9),
            _chunk("c2", "b", 0.8),
            _chunk("c3", "c", 0.7),
            _chunk("c4", "d", 0.6),
        )
        result = retrieve_constrained(
            query="q",
            allowed_chunk_ids=["c1", "c2", "c3", "c4"],
            vector_store=store,
            top_k=2,
        )
        assert len(result) == 2
        assert result[0].score > result[1].score

    def test_highest_scoring_allowed_chunks_returned(self) -> None:
        # c2 scores higher than c1 but only c1 is allowed — c1 must be returned.
        store = _make_store(
            _chunk("c2", "not allowed", 0.99),
            _chunk("c1", "allowed", 0.5),
        )
        result = retrieve_constrained(
            query="q",
            allowed_chunk_ids=["c1"],
            vector_store=store,
            top_k=5,
        )
        assert len(result) == 1
        assert result[0].chunk_id == "c1"
        assert result[0].score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# doc_id field tests (T005)
# ---------------------------------------------------------------------------


class TestDocId:
    """doc_id is preserved from vector store results and defaults to ""."""

    def test_doc_id_preserved_from_search_result(self) -> None:
        store = _make_store(
            {"chunk_id": "c1", "text": "some text", "score": 0.9, "doc_id": "contract-42"},
        )
        result = retrieve_constrained(
            query="q",
            allowed_chunk_ids=["c1"],
            vector_store=store,
        )
        assert len(result) == 1
        assert result[0].doc_id == "contract-42"

    def test_doc_id_defaults_to_empty_string_when_absent_from_dict(self) -> None:
        # _chunk() helper omits doc_id; .get fallback must supply "" not raise KeyError.
        store = _make_store(_chunk("c1", "text", 0.8))
        result = retrieve_constrained(
            query="q",
            allowed_chunk_ids=["c1"],
            vector_store=store,
        )
        assert len(result) == 1
        assert result[0].doc_id == ""

    def test_ranked_chunk_doc_id_field_defaults_to_empty_string(self) -> None:
        # Positional constructor must still work without doc_id — existing tests depend on this.
        chunk = RankedChunk(chunk_id="x", text="y", score=0.5)
        assert chunk.doc_id == ""
