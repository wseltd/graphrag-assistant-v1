from __future__ import annotations

import pytest

from app.benchmark.scoring import (
    score_citation_coverage,
    score_keyword_overlap,
    score_query,
)


class TestScoreKeywordOverlap:
    def test_all_stop_words_returns_zero(self) -> None:
        # Expected is all stop-words → expected_tokens is empty → denominator is 1
        result = score_keyword_overlap("the a an is in", "the a an is in")
        assert result == 0.0

    def test_empty_actual_returns_zero(self) -> None:
        result = score_keyword_overlap("contract procurement supplier", "")
        assert result == 0.0

    def test_case_insensitive_full_match(self) -> None:
        result = score_keyword_overlap("Contract Supplier", "contract supplier")
        assert result == 1.0

    def test_punctuation_stripped(self) -> None:
        result = score_keyword_overlap("contract, supplier.", "contract supplier")
        assert result == 1.0

    def test_partial_match(self) -> None:
        result = score_keyword_overlap(
            "contract supplier procurement", "contract supplier"
        )
        # 2 matched out of 3 expected
        assert pytest.approx(result, abs=1e-6) == 2 / 3

    def test_no_match_returns_zero(self) -> None:
        result = score_keyword_overlap("contract supplier", "weather forecast")
        assert result == 0.0

    def test_empty_expected_returns_zero(self) -> None:
        result = score_keyword_overlap("", "contract supplier")
        assert result == 0.0


class TestScoreCitationCoverage:
    def test_duplicate_chunk_ids_counted_once(self) -> None:
        citations = [
            {"chunk_id": "c1"},
            {"chunk_id": "c1"},
            {"chunk_id": "c2"},
        ]
        result = score_citation_coverage(["c1", "c2"], citations)
        assert result == 1.0

    def test_zero_matches(self) -> None:
        citations = [{"chunk_id": "c3"}, {"chunk_id": "c4"}]
        result = score_citation_coverage(["c1", "c2"], citations)
        assert result == 0.0

    def test_partial_coverage(self) -> None:
        citations = [{"chunk_id": "c1"}]
        result = score_citation_coverage(["c1", "c2"], citations)
        assert result == 0.5

    def test_empty_expected_returns_zero(self) -> None:
        citations = [{"chunk_id": "c1"}]
        result = score_citation_coverage([], citations)
        assert result == 0.0

    def test_full_coverage(self) -> None:
        citations = [{"chunk_id": "c1"}, {"chunk_id": "c2"}, {"chunk_id": "c3"}]
        result = score_citation_coverage(["c1", "c2", "c3"], citations)
        assert result == 1.0

    def test_missing_chunk_id_key_ignored(self) -> None:
        citations = [{"chunk_id": "c1"}, {"doc_id": "d1"}]
        result = score_citation_coverage(["c1"], citations)
        assert result == 1.0


class TestScoreQuery:
    def test_returns_all_keys(self) -> None:
        expected = {"answer": "contract", "chunk_ids": ["c1"]}
        result_data = {"answer": "contract", "text_citations": [{"chunk_id": "c1"}]}
        out = score_query(expected, result_data, 0.5)
        assert set(out.keys()) == {"accuracy", "citation_coverage", "latency_seconds"}

    def test_latency_preserved(self) -> None:
        out = score_query({"answer": "x", "chunk_ids": []}, {"answer": "x"}, 1.23)
        assert out["latency_seconds"] == 1.23

    def test_accuracy_and_coverage_computed(self) -> None:
        expected = {"answer": "procurement contract", "chunk_ids": ["c1"]}
        result_data = {
            "answer": "procurement contract",
            "text_citations": [{"chunk_id": "c1"}],
        }
        out = score_query(expected, result_data, 0.1)
        assert out["accuracy"] == 1.0
        assert out["citation_coverage"] == 1.0
