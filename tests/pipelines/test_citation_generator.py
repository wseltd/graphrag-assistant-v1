"""Tests for app/pipelines/citation_generator.py (T025.d)."""
from __future__ import annotations

import dataclasses

from app.pipelines.citation_generator import (
    Citation,
    GenerationResult,
    _first_sentence,
    generate_answer,
)
from app.pipelines.constrained_retrieval import RankedChunk
from app.pipelines.graph_traversal import Triple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, text: str, score: float = 0.9) -> RankedChunk:
    return RankedChunk(chunk_id=chunk_id, text=text, score=score)


def _triple(src: str, rel: str, dst: str) -> Triple:
    return Triple(src=src, rel=rel, dst=dst)


# ---------------------------------------------------------------------------
# Query line
# ---------------------------------------------------------------------------


class TestQueryLine:
    def test_query_line_present(self):
        result = generate_answer("What is X?", [], [])
        assert result.answer.startswith("Query: What is X?")

    def test_query_line_with_special_characters(self):
        result = generate_answer("Find 'Acme Corp' & partners?", [], [])
        assert "Query: Find 'Acme Corp' & partners?" in result.answer


# ---------------------------------------------------------------------------
# Graph context section
# ---------------------------------------------------------------------------


class TestGraphContextSection:
    def test_single_triple_formatted(self):
        result = generate_answer("q", [], [_triple("1", "PARTY_TO", "2")])
        assert "1 --[PARTY_TO]--> 2" in result.answer

    def test_multiple_triples_all_present(self):
        triples = [
            _triple("10", "DIRECTOR_OF", "20"),
            _triple("20", "REGISTERED_AT", "30"),
        ]
        result = generate_answer("q", [], triples)
        assert "10 --[DIRECTOR_OF]--> 20" in result.answer
        assert "20 --[REGISTERED_AT]--> 30" in result.answer

    def test_graph_context_header_present(self):
        result = generate_answer("q", [], [_triple("a", "R", "b")])
        assert "Graph context:" in result.answer

    def test_no_graph_context_header_when_empty(self):
        result = generate_answer("q", [], [])
        assert "Graph context:" not in result.answer


# ---------------------------------------------------------------------------
# Supporting text section
# ---------------------------------------------------------------------------


class TestSupportingTextSection:
    def test_citation_marker_in_answer(self):
        result = generate_answer("q", [_chunk("chunk-001", "Sentence one. More text.")], [])
        assert "[chunk-001]" in result.answer

    def test_first_sentence_used_in_answer(self):
        result = generate_answer("q", [_chunk("c1", "First sentence. Second sentence.")], [])
        assert "First sentence." in result.answer
        assert "Second sentence." not in result.answer

    def test_supporting_text_header_present(self):
        result = generate_answer("q", [_chunk("c1", "text")], [])
        assert "Supporting text:" in result.answer

    def test_no_supporting_text_header_when_no_chunks(self):
        result = generate_answer("q", [], [_triple("a", "R", "b")])
        assert "Supporting text:" not in result.answer

    def test_multiple_chunks_all_cited(self):
        chunks = [_chunk("c1", "Text one."), _chunk("c2", "Text two."), _chunk("c3", "Text three.")]
        result = generate_answer("q", chunks, [])
        assert "[c1]" in result.answer
        assert "[c2]" in result.answer
        assert "[c3]" in result.answer


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


class TestFallback:
    def test_fallback_when_both_empty(self):
        result = generate_answer("q", [], [])
        assert "No evidence retrieved" in result.answer

    def test_no_fallback_when_triples_present(self):
        result = generate_answer("q", [], [_triple("a", "R", "b")])
        assert "No evidence retrieved" not in result.answer

    def test_no_fallback_when_chunks_present(self):
        result = generate_answer("q", [_chunk("c", "text")], [])
        assert "No evidence retrieved" not in result.answer

    def test_triples_and_chunks_no_fallback(self):
        result = generate_answer("q", [_chunk("c", "text")], [_triple("a", "R", "b")])
        assert "No evidence retrieved" not in result.answer


# ---------------------------------------------------------------------------
# text_citations
# ---------------------------------------------------------------------------


class TestTextCitations:
    def test_count_matches_chunks(self):
        chunks = [_chunk(f"c{i}", f"text {i}") for i in range(4)]
        result = generate_answer("q", chunks, [])
        assert len(result.text_citations) == 4

    def test_chunk_id_preserved(self):
        result = generate_answer("q", [_chunk("abc-123", "some text")], [])
        assert result.text_citations[0].chunk_id == "abc-123"

    def test_quote_truncated_to_max(self):
        long_text = "x" * 300
        result = generate_answer("q", [_chunk("c1", long_text)], [])
        assert len(result.text_citations[0].quote) == 200

    def test_quote_short_text_unchanged(self):
        result = generate_answer("q", [_chunk("c1", "short text")], [])
        assert result.text_citations[0].quote == "short text"

    def test_citation_field_is_named_quote_not_excerpt(self):
        # Regression: Citation used to have `excerpt`; aligning with TextCitation.quote
        # removes the field-name mismatch that caused benchmark/router.py to output
        # the wrong key name.
        citation = Citation(chunk_id="c1", quote="some text")
        assert hasattr(citation, "quote")
        assert not hasattr(citation, "excerpt")

    def test_citation_field_is_named_quote(self):
        # Structural guard: dataclasses.fields() is authoritative — hasattr passes even
        # if a property or __getattr__ shim fakes the name.
        field_names = {f.name for f in dataclasses.fields(Citation)}
        assert "quote" in field_names

    def test_citation_has_no_excerpt_field(self):
        # Structural guard: ensures the rename from `excerpt` to `quote` is permanent
        # at the dataclass level, not just masked by an attribute alias.
        field_names = {f.name for f in dataclasses.fields(Citation)}
        assert "excerpt" not in field_names

    def test_empty_chunks_empty_citations(self):
        result = generate_answer("q", [], [_triple("a", "R", "b")])
        assert result.text_citations == []

    def test_citation_order_matches_chunk_order(self):
        chunks = [_chunk("first", "a."), _chunk("second", "b."), _chunk("third", "c.")]
        result = generate_answer("q", chunks, [])
        ids = [c.chunk_id for c in result.text_citations]
        assert ids == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


class TestReturnTypes:
    def test_returns_generation_result(self):
        assert isinstance(generate_answer("q", [], []), GenerationResult)

    def test_citations_are_citation_instances(self):
        result = generate_answer("q", [_chunk("c1", "text")], [])
        assert all(isinstance(c, Citation) for c in result.text_citations)

    def test_answer_is_string(self):
        result = generate_answer("q", [], [])
        assert isinstance(result.answer, str)


# ---------------------------------------------------------------------------
# _first_sentence
# ---------------------------------------------------------------------------


class TestFirstSentence:
    def test_period_terminates(self):
        assert _first_sentence("Hello. World.") == "Hello."

    def test_no_period_returns_full(self):
        assert _first_sentence("No period here") == "No period here"

    def test_empty_string(self):
        assert _first_sentence("") == ""

    def test_period_at_start(self):
        assert _first_sentence(". rest") == "."
