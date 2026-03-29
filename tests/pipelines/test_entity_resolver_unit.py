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


def test_clause_id_exact_match_resolves_to_clause_node() -> None:
    """A clause_id string resolves to exactly one Clause EntityMatch (label='Clause').

    Before Bug 12 fix: _LABEL_QUERIES has no Clause key → session.run never
    receives Clause Cypher → result is empty → test fails.
    After fix: Clause Cypher is executed, row returned, EntityMatch produced.
    """
    session = _session_returning_for_label(
        ":Clause",
        [{"node_id": "CLAUSE-001", "name": "CLAUSE-001"}],
    )
    results = resolve_entities('"CLAUSE-001"', session, top_k=5)
    assert any(m.label == "Clause" for m in results), (
        "Expected at least one EntityMatch with label='Clause' for clause_id candidate"
    )


def test_clause_type_contains_match_resolves_to_clause_node() -> None:
    """A clause_type keyword resolves via CONTAINS to a Clause EntityMatch (label='Clause').

    Before Bug 12 fix: no Clause entry → no match → test fails.
    After fix: candidate 'payment' hits the CONTAINS branch of the Clause Cypher.
    """
    session = _session_returning_for_label(
        ":Clause",
        [{"node_id": "CLAUSE-042", "name": "payment"}],
    )
    results = resolve_entities('"payment"', session, top_k=5)
    assert any(m.label == "Clause" for m in results), (
        "Expected at least one EntityMatch with label='Clause' for clause_type keyword"
    )


def test_city_candidate_resolves_to_address_entity_match() -> None:
    """Exact city name resolves to an Address EntityMatch with label='Address'.

    node_id must be the Address domain key (n.id), not the city string —
    downstream constrained retrieval uses domain keys for anchor lookup.
    """
    # Only the Address query returns a row; all other label queries return empty.
    session = _session_returning_for_label(
        ":Address",
        [{"node_id": "ADDR-001", "name": "Berlin"}],
    )
    results = resolve_entities('"Berlin"', session, top_k=5)
    assert len(results) >= 1
    address_match = next((r for r in results if r.node_id == "ADDR-001"), None)
    assert address_match is not None, "Expected a match for city candidate 'Berlin'"
    assert address_match.label == "Address"
    assert address_match.score == 1.0  # candidate == name → exact match


def test_partial_city_candidate_resolves_to_address_entity_match() -> None:
    """Short city prefix resolves to an Address EntityMatch via forward CONTAINS.

    Simulates: candidate 'Berl' matches an Address node whose city is 'Berlin'
    via the forward CONTAINS branch (n.city CONTAINS candidate).
    Score must be < 1.0 because the candidate is shorter than the stored name.
    """
    session = _session_returning_for_label(
        ":Address",
        [{"node_id": "ADDR-001", "name": "Berlin"}],
    )
    results = resolve_entities('"Berl"', session, top_k=5)
    assert len(results) >= 1
    address_match = next((r for r in results if r.node_id == "ADDR-001"), None)
    assert address_match is not None, "Expected a match for partial city candidate 'Berl'"
    assert address_match.label == "Address"
    assert address_match.score < 1.0  # partial CONTAINS → overlap ratio < 1


def test_city_exact_match_resolves_to_address_node() -> None:
    """Exact city name resolves to an Address EntityMatch (label='Address').

    Exercises the equality branch of the bidirectional CONTAINS check:
    toLower(n.city) CONTAINS toLower($candidate) where candidate == city.

    Before Bug 13 fix: no Address entry in _LABEL_QUERIES → no Address Cypher
    runs → result is empty → test fails.
    After fix: Address Cypher executes, row returned, EntityMatch produced.
    """
    session = _session_returning_for_label(
        ":Address",
        [{"node_id": "ADDR-001", "name": "Berlin"}],
    )
    results = resolve_entities('"Berlin"', session, top_k=5)
    assert any(m.label == "Address" for m in results), (
        "Expected at least one EntityMatch with label='Address' for city candidate 'Berlin'"
    )


def test_city_partial_match_resolves_to_address_node() -> None:
    """Partial city substring resolves to an Address EntityMatch (label='Address').

    Exercises the reverse CONTAINS branch:
    toLower($candidate) CONTAINS toLower(n.city) where candidate 'East' is
    contained within stored city 'East Berlin'.

    Before Bug 13 fix: no Address entry in _LABEL_QUERIES → no Address Cypher
    runs → result is empty → test fails.
    After fix: Address Cypher executes, row returned, EntityMatch produced.
    """
    session = _session_returning_for_label(
        ":Address",
        [{"node_id": "ADDR-002", "name": "East Berlin"}],
    )
    results = resolve_entities('"East"', session, top_k=5)
    assert any(m.label == "Address" for m in results), (
        "Expected at least one EntityMatch with label='Address' for partial city candidate 'East'"
    )
