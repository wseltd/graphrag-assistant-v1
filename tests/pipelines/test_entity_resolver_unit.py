"""Unit tests for app.pipelines.entity_resolver — T036.

Four isolated test functions covering the four key resolver behaviours using a
fake Neo4j driver; no live database or network required.

Cases
-----
  test_entity_resolver_happy_path    — one exact match, score >= 0.9
  test_entity_resolver_partial_match — one substring match, score < 0.9
  test_entity_resolver_no_match      — empty result + warning includes queried name
  test_entity_resolver_ambiguous_name — two matches returned; count surfaced in result
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from app.pipelines.entity_resolver import resolve_entities

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_returning(rows: list[dict]) -> MagicMock:
    """Return a mock session whose run() always returns *rows* for every label query."""
    session = MagicMock()
    session.run.return_value = rows
    return session


def _empty_session() -> MagicMock:
    session = MagicMock()
    session.run.return_value = []
    return session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_entity_resolver_happy_path() -> None:
    """One exact-match node returned: resolved entity id matches and score >= 0.9."""
    session = _session_returning([{"node_id": "COMP_001", "name": "Acme Corp"}])
    results = resolve_entities('"Acme Corp"', session, top_k=5)
    assert results[0].node_id == "COMP_001"
    assert results[0].score >= 0.9


def test_entity_resolver_partial_match() -> None:
    """Candidate is a substring of the stored name: resolved id matches and score < 0.9."""
    session = _session_returning(
        [{"node_id": "COMP_002", "name": "Acme Corporation Holdings"}]
    )
    results = resolve_entities('"Acme"', session, top_k=5)
    assert results[0].node_id == "COMP_002"
    assert results[0].score < 0.9


def test_entity_resolver_no_match(caplog: pytest.LogCaptureFixture) -> None:
    """No graph matches: resolver returns [] and the warning log includes the queried name."""
    queried_name = "Zxqfoo Industries"
    session = _empty_session()
    with caplog.at_level(logging.WARNING, logger="app.pipelines.entity_resolver"):
        results = resolve_entities(f'"{queried_name}"', session, top_k=5)
    assert results == []
    assert any(queried_name in r.message for r in caplog.records)


def test_entity_resolver_ambiguous_name() -> None:
    """Two nodes match the candidate: resolver returns both, surfacing the candidate count."""
    session = _session_returning(
        [
            {"node_id": "COMP_010", "name": "Global Corp"},
            {"node_id": "COMP_011", "name": "Global Corporation Ltd"},
        ]
    )
    results = resolve_entities('"Global"', session, top_k=5)
    assert len(results) >= 2
    assert len(results) == 2


def _session_returning_for_label(label_keyword: str, rows: list[dict]) -> MagicMock:
    """Return a mock session that yields *rows* only when the Cypher mentions *label_keyword*.

    All other label queries return an empty list, ensuring the EntityMatch label
    field reflects the correct _LABEL_QUERIES key rather than whichever label
    happens to run first.
    """
    session = MagicMock()
    session.run.side_effect = lambda cypher, **_kwargs: (
        rows if label_keyword in cypher else []
    )
    return session


def test_clause_id_candidate_resolves_to_clause_entity_match() -> None:
    """Exact clause_id string resolves to a Clause EntityMatch with label='Clause'."""
    # Only the Clause query returns a row; all other label queries return empty.
    # This ensures the EntityMatch.label is set from the 'Clause' key, not a
    # coincidentally-first label that also received the mocked row.
    session = _session_returning_for_label(
        ":Clause",
        [{"node_id": "CL-042", "name": "CL-042"}],
    )
    results = resolve_entities('"CL-042"', session, top_k=5)
    assert len(results) >= 1
    clause_match = next((r for r in results if r.node_id == "CL-042"), None)
    assert clause_match is not None, "Expected a match for clause_id CL-042"
    assert clause_match.label == "Clause"
    assert clause_match.score == 1.0  # candidate == name → exact match


def test_clause_type_keyword_candidate_resolves_to_clause_entity_match() -> None:
    """Clause type keyword resolves via forward CONTAINS to a Clause EntityMatch.

    Simulates: candidate 'termination' matches a Clause node whose clause_type
    is 'Termination Clause' via the forward CONTAINS branch.
    """
    session = _session_returning_for_label(
        ":Clause",
        [{"node_id": "CL-007", "name": "CL-007"}],
    )
    results = resolve_entities('"termination"', session, top_k=5)
    assert len(results) >= 1
    clause_match = next((r for r in results if r.node_id == "CL-007"), None)
    assert clause_match is not None, "Expected a match for termination clause keyword"
    assert clause_match.label == "Clause"
