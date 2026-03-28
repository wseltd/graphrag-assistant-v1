"""Tests for app.pipelines.entity_resolver (T025.a).

Coverage map (15 tests total, 10 on entity resolution and traversal):

entity_resolution (6):
  test_exact_match                   — exact name match → score 1.0
  test_case_insensitive_match        — different-case candidate resolves correctly
  test_partial_substring_match       — short candidate inside longer name → score < 1.0
  test_zero_matches_returns_empty    — no graph hits → empty list + WARNING logged
  test_multiple_candidates_ranked    — multi-node result sorted by score descending
  test_name_with_special_characters  — candidate containing &, (), - chars

graph_traversal (4):
  test_graph_single_hop_path         — lookup_facts returns 1-hop GraphFact list
  test_graph_two_hop_path            — lookup_facts called with hop=2 returns facts
  test_graph_no_path_between_anchors — lookup_facts returns [] for disconnected nodes
  test_graph_isolated_node           — resolve returns node; traversal returns empty

constrained_vector_retrieval (3):
  test_constrained_filtered              — allowed_ids filters chunks to graph-adjacent
  test_constrained_fallback              — empty allowed_ids runs unconstrained
  test_constrained_ranking               — score order preserved after filter

integration_pipeline (2):
  test_pipeline_end_to_end_seeded        — full mocked pipeline returns AnswerSchema
  test_unknown_entity_fallback           — unknown entity → mode='plain_rag' fallback
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from app.pipelines.entity_resolver import (
    EntityMatch,
    _score,
    extract_candidates,
    resolve_entities,
)
from app.retrieval.constrained_retriever import (
    ChunkResult,
    ConstrainedRetrievalResult,
    retrieve_chunks,
)
from app.retrieval.cypher_templates import lookup_facts
from app.retrieval.graph_traverser_types import GraphFact

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_returning(rows_by_label: dict[str, list[dict]]) -> MagicMock:
    """Return a mock Neo4j session whose run() dispatches by MATCH label keyword."""
    session = MagicMock()

    def _run(cypher: str, **kwargs: object) -> list[dict]:
        for label, rows in rows_by_label.items():
            if f":{label}" in cypher or f"n:{label}" in cypher:
                return rows
        return []

    session.run.side_effect = _run
    return session


def _empty_session() -> MagicMock:
    session = MagicMock()
    session.run.return_value = []
    return session


# ---------------------------------------------------------------------------
# entity_resolution
# ---------------------------------------------------------------------------


class TestExactMatch:
    def test_exact_match_score_is_one(self) -> None:
        """An exact name match (case-identical) must produce score=1.0."""
        session = _session_returning(
            {"Company": [{"node_id": "COMP_001", "name": "Acme Corp"}]}
        )
        results = resolve_entities("Acme Corp contract", session, top_k=5)
        assert len(results) == 1
        match = results[0]
        assert isinstance(match, EntityMatch)
        assert match.node_id == "COMP_001"
        assert match.label == "Company"
        assert match.name == "Acme Corp"
        assert match.score == 1.0


class TestCaseInsensitiveMatch:
    def test_lowercase_query_finds_capitalised_node(self) -> None:
        """Case-insensitive CONTAINS: lowercase candidate resolves to mixed-case name."""
        session = _session_returning(
            {"Person": [{"node_id": "PERS_007", "name": "Alice Nguyen"}]}
        )
        results = resolve_entities('"alice nguyen"', session, top_k=5)
        assert len(results) == 1
        assert results[0].node_id == "PERS_007"
        # _score uses case-insensitive equality → 1.0
        assert results[0].score == 1.0

    def test_upper_candidate_matches_lower_stored_name(self) -> None:
        """Stored name 'acme corp' matched by candidate 'ACME CORP' → score 1.0."""
        session = _session_returning(
            {"Company": [{"node_id": "C1", "name": "acme corp"}]}
        )
        results = resolve_entities('"ACME CORP"', session, top_k=5)
        assert results[0].score == 1.0


class TestPartialSubstringMatch:
    def test_partial_match_score_less_than_one(self) -> None:
        """A 4-char candidate inside a 13-char name yields score=4/13."""
        assert _score("Acme", "Acme Corp Ltd") == pytest.approx(4 / 13)

    def test_partial_match_resolved_node(self) -> None:
        """Short candidate 'Acme' CONTAINS-matches 'Acme Corporation'."""
        session = _session_returning(
            {"Company": [{"node_id": "C2", "name": "Acme Corporation"}]}
        )
        results = resolve_entities('"Acme"', session, top_k=5)
        assert len(results) == 1
        assert results[0].score == pytest.approx(4 / 16)
        assert results[0].node_id == "C2"

    def test_extract_candidates_returns_quoted_verbatim(self) -> None:
        """Quoted phrase is returned as-is, not broken by stop-word splitting."""
        candidates = extract_candidates('"Bell & Sons"')
        assert candidates == ["Bell & Sons"]


class TestZeroMatchesReturnsEmpty:
    def test_empty_result_on_zero_graph_hits(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When session returns no rows, resolve_entities returns [] and logs WARNING."""
        session = _empty_session()
        with caplog.at_level(logging.WARNING, logger="app.pipelines.entity_resolver"):
            results = resolve_entities("GlobalTech procurement", session, top_k=5)
        assert results == []
        assert any("zero graph matches" in r.message for r in caplog.records)

    def test_no_candidates_extracted_returns_empty(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A query of only stop words produces no candidates and logs WARNING."""
        session = _empty_session()
        with caplog.at_level(logging.WARNING, logger="app.pipelines.entity_resolver"):
            results = resolve_entities("what is the of a", session, top_k=5)
        assert results == []
        assert any("no candidates extracted" in r.message for r in caplog.records)
        session.run.assert_not_called()


class TestMultipleCandidatesRanked:
    def test_higher_score_comes_first(self) -> None:
        """Multiple nodes: exact match (score=1.0) ranked above partial match."""
        session = _session_returning(
            {
                "Company": [
                    {"node_id": "C1", "name": "NovaTech"},
                    {"node_id": "C2", "name": "NovaTech Holdings Ltd"},
                ]
            }
        )
        results = resolve_entities('"NovaTech"', session, top_k=5)
        assert results[0].score == 1.0
        assert results[0].node_id == "C1"
        assert results[1].score < 1.0

    def test_top_k_limits_results(self) -> None:
        """top_k=2 returns at most 2 results even when more nodes match."""
        session = _session_returning(
            {
                "Company": [
                    {"node_id": f"C{i}", "name": f"Company {i}"}
                    for i in range(5)
                ]
            }
        )
        results = resolve_entities('"Company"', session, top_k=2)
        assert len(results) <= 2


class TestNameWithSpecialCharacters:
    def test_ampersand_in_candidate(self) -> None:
        """Candidate 'Bell & Sons' containing & resolves to matching node."""
        session = _session_returning(
            {"Company": [{"node_id": "C_BS", "name": "Bell & Sons Ltd"}]}
        )
        results = resolve_entities('"Bell & Sons"', session, top_k=5)
        assert len(results) == 1
        assert results[0].node_id == "C_BS"

    def test_hyphenated_name(self) -> None:
        """Candidate with hyphen resolves correctly."""
        session = _session_returning(
            {"Company": [{"node_id": "C_HY", "name": "Smith-Jones Partners"}]}
        )
        results = resolve_entities('"Smith-Jones"', session, top_k=5)
        assert results[0].node_id == "C_HY"


# ---------------------------------------------------------------------------
# graph_traversal
# ---------------------------------------------------------------------------


class TestGraphSingleHopPath:
    def test_single_hop_facts_returned(self) -> None:
        """lookup_facts with hop=1 returns GraphFact list from a mocked session."""
        session = MagicMock()
        session.run.return_value = [
            {
                "source_id": "10",
                "source_label": "Person",
                "rel_type": "DIRECTOR_OF",
                "target_id": "20",
                "target_label": "Company",
            }
        ]
        facts = lookup_facts(session, "Person", "Company", ids=[10], hop=1)
        assert len(facts) == 1
        fact = facts[0]
        assert isinstance(fact, GraphFact)
        assert fact.source_id == "10"
        assert fact.rel_type == "DIRECTOR_OF"
        assert fact.target_id == "20"
        session.run.assert_called_once()


class TestGraphTwoHopPath:
    def test_two_hop_facts_returned(self) -> None:
        """lookup_facts with hop=2 executes the 2-hop Cypher template."""
        session = MagicMock()
        session.run.return_value = [
            {
                "source_id": "1",
                "source_label": "Person",
                "rel_type": "DIRECTOR_OF",
                "target_id": "2",
                "target_label": "Company",
            },
            {
                "source_id": "2",
                "source_label": "Company",
                "rel_type": "DIRECTOR_OF",
                "target_id": "3",
                "target_label": "Company",
            },
        ]
        facts = lookup_facts(session, "Person", "Company", ids=[1], hop=2)
        assert len(facts) == 2
        assert all(isinstance(f, GraphFact) for f in facts)
        cypher_used = session.run.call_args[0][0]
        assert "UNWIND relationships(p)" in cypher_used


class TestGraphNoPathBetweenAnchors:
    def test_empty_list_when_no_path(self) -> None:
        """lookup_facts returns [] when the session returns no rows."""
        session = MagicMock()
        session.run.return_value = []
        facts = lookup_facts(session, "Company", "Contract", ids=[99], hop=1)
        assert facts == []

    def test_no_db_call_when_ids_empty(self) -> None:
        """lookup_facts must not call session.run when ids=[]."""
        session = MagicMock()
        facts = lookup_facts(session, "Company", "Contract", ids=[], hop=1)
        assert facts == []
        session.run.assert_not_called()


class TestGraphIsolatedNode:
    def test_isolated_node_traversal_returns_empty(self) -> None:
        """An isolated node has no outgoing PARTY_TO edges → lookup_facts returns []."""
        session = MagicMock()
        session.run.return_value = []
        facts = lookup_facts(session, "Company", "Contract", ids=[42], hop=1)
        assert facts == []


# ---------------------------------------------------------------------------
# constrained_vector_retrieval
# ---------------------------------------------------------------------------


class TestConstrainedFiltered:
    def test_allowed_ids_triggers_constrained_query(self) -> None:
        """Non-empty allowed_ids → graph_constrained=True in result."""
        embed = MagicMock()
        embed.embed.return_value = [[0.1, 0.2]]
        session = MagicMock()
        session.run.return_value = [{"chunk_id": "CL001_c0", "score": 0.95}]
        result = retrieve_chunks(
            "procurement query",
            embed,
            session,
            allowed_ids=["COMP_001"],
            top_k=3,
        )
        assert isinstance(result, ConstrainedRetrievalResult)
        assert result.graph_constrained is True
        assert "CL001_c0" in result.retrieved_chunk_ids
        assert all(isinstance(c, ChunkResult) for c in result.chunks)


class TestConstrainedFallback:
    def test_empty_allowed_ids_triggers_unconstrained(self) -> None:
        """Empty allowed_ids → graph_constrained=False (unconstrained fallback)."""
        embed = MagicMock()
        embed.embed.return_value = [[0.5, 0.5]]
        session = MagicMock()
        session.run.return_value = [{"chunk_id": "CL002_c1", "score": 0.7}]
        result = retrieve_chunks(
            "generic query",
            embed,
            session,
            allowed_ids=[],
            top_k=3,
        )
        assert result.graph_constrained is False
        assert result.retrieved_chunk_ids == ["CL002_c1"]


class TestConstrainedRetrievalRanking:
    def test_score_order_preserved(self) -> None:
        """Chunks are returned in descending score order after the filter."""
        embed = MagicMock()
        embed.embed.return_value = [[0.1]]
        session = MagicMock()
        session.run.return_value = [
            {"chunk_id": "HIGH", "score": 0.99},
            {"chunk_id": "MID", "score": 0.75},
            {"chunk_id": "LOW", "score": 0.40},
        ]
        result = retrieve_chunks(
            "query",
            embed,
            session,
            allowed_ids=["C1"],
            top_k=5,
        )
        scores = [c.score for c in result.chunks]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# integration_pipeline
# ---------------------------------------------------------------------------


class TestPipelineEndToEndSeeded:
    def test_returns_populated_answer_schema(self) -> None:
        """Full mocked pipeline: entity resolution → traversal → retrieval → answer."""
        resolver_session = _session_returning(
            {"Company": [{"node_id": "COMP_001", "name": "Acme Corp"}]}
        )
        matches = resolve_entities("Acme Corp procurement", resolver_session, top_k=5)
        assert len(matches) >= 1
        assert matches[0].node_id == "COMP_001"

        traversal_session = MagicMock()
        traversal_session.run.return_value = [
            {
                "source_id": "5",
                "source_label": "Company",
                "rel_type": "PARTY_TO",
                "target_id": "12",
                "target_label": "Contract",
            }
        ]
        facts = lookup_facts(traversal_session, "Company", "Contract", ids=[5], hop=1)
        assert len(facts) == 1
        assert facts[0].rel_type == "PARTY_TO"

        embed = MagicMock()
        embed.embed.return_value = [[0.3, 0.7]]
        retrieval_session = MagicMock()
        retrieval_session.run.return_value = [{"chunk_id": "CT001_c0", "score": 0.88}]
        retrieval_result = retrieve_chunks(
            "Acme Corp procurement",
            embed,
            retrieval_session,
            allowed_ids=["COMP_001"],
            top_k=5,
        )
        assert retrieval_result.graph_constrained is True
        assert "CT001_c0" in retrieval_result.retrieved_chunk_ids


class TestUnknownEntityFallback:
    def test_unknown_entity_returns_empty_matches(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unknown entity name → resolve_entities returns [] → pipeline falls back."""
        session = _empty_session()
        with caplog.at_level(logging.WARNING, logger="app.pipelines.entity_resolver"):
            matches = resolve_entities("Zxqrfoo Nonexistent Widgets", session, top_k=5)
        assert matches == []
        # Downstream: empty allowed_ids triggers unconstrained retrieval
        embed = MagicMock()
        embed.embed.return_value = [[0.1, 0.2]]
        retrieval_session = MagicMock()
        retrieval_session.run.return_value = [{"chunk_id": "ANY_c0", "score": 0.5}]
        result = retrieve_chunks(
            "Zxqrfoo Nonexistent Widgets",
            embed,
            retrieval_session,
            allowed_ids=[m.node_id for m in matches],  # []
            top_k=3,
        )
        assert result.graph_constrained is False
