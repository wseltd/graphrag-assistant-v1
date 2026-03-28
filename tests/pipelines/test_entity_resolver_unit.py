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
