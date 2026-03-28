"""Tests for app.pipelines.graph_traversal (T025.b).

Coverage map (4 tests — concentrated on graph traversal as the highest-risk
path):

  TestSingleHopTraversal    — 1-hop path: triples and Chunk IDs collected
  TestTwoHopTraversal       — 2-hop path: deeper triples collected, dedup applied
  TestNoPathTraversal       — no outgoing edges: empty result, no DB call for []
  TestIsolatedNodeTraversal — isolated anchor: empty triples and chunk_ids
"""
from __future__ import annotations

from unittest.mock import MagicMock

from app.pipelines.graph_traversal import GraphTraversalResult, Triple, traverse_from_anchors

# ---------------------------------------------------------------------------
# TestSingleHopTraversal
# ---------------------------------------------------------------------------


class TestSingleHopTraversal:
    def test_triple_and_chunk_on_direct_path(self) -> None:
        """1-hop path with one Chunk neighbour: triple stored, chunk_id collected."""
        session = MagicMock()
        session.run.return_value = [
            {"src": "1", "rel": "PARTY_TO", "dst": "2", "chunk_id": None},
            {"src": "1", "rel": "ABOUT_COMPANY", "dst": "3", "chunk_id": "CL001_c0"},
        ]
        result = traverse_from_anchors(["COMP_001"], session, max_hops=1)
        assert isinstance(result, GraphTraversalResult)
        assert Triple(src="1", rel="PARTY_TO", dst="2") in result.triples
        assert "CL001_c0" in result.chunk_ids

    def test_max_hops_one_issues_single_db_call(self) -> None:
        """max_hops=1 issues exactly one session.run call."""
        session = MagicMock()
        session.run.return_value = []
        traverse_from_anchors(["C1"], session, max_hops=1)
        assert session.run.call_count == 1


# ---------------------------------------------------------------------------
# TestTwoHopTraversal
# ---------------------------------------------------------------------------


class TestTwoHopTraversal:
    def test_two_hop_triples_present(self) -> None:
        """2-hop traversal: triples from both hops appear in result."""
        session = MagicMock()
        session.run.return_value = [
            {"src": "1", "rel": "PARTY_TO", "dst": "2", "chunk_id": None},
            {"src": "2", "rel": "HAS_CLAUSE", "dst": "3", "chunk_id": "CT001_c0"},
        ]
        result = traverse_from_anchors(["COMP_001"], session, max_hops=2)
        assert len(result.triples) == 2
        assert any(t.src == "2" and t.rel == "HAS_CLAUSE" for t in result.triples)
        assert "CT001_c0" in result.chunk_ids

    def test_duplicate_rows_are_deduplicated(self) -> None:
        """Rows sharing identical (src, rel, dst) are collapsed to one triple."""
        session = MagicMock()
        session.run.return_value = [
            {"src": "1", "rel": "PARTY_TO", "dst": "2", "chunk_id": None},
            {"src": "1", "rel": "PARTY_TO", "dst": "2", "chunk_id": None},
        ]
        result = traverse_from_anchors(["COMP_001"], session, max_hops=2)
        assert len(result.triples) == 1


# ---------------------------------------------------------------------------
# TestNoPathTraversal
# ---------------------------------------------------------------------------


class TestNoPathTraversal:
    def test_empty_when_session_returns_no_rows(self) -> None:
        """When the session returns no rows, both lists are empty."""
        session = MagicMock()
        session.run.return_value = []
        result = traverse_from_anchors(["NODE_X"], session)
        assert result.chunk_ids == []
        assert result.triples == []

    def test_empty_node_ids_skips_db_call(self) -> None:
        """Empty node_ids must not issue any DB call."""
        session = MagicMock()
        result = traverse_from_anchors([], session)
        assert result.chunk_ids == []
        assert result.triples == []
        session.run.assert_not_called()


# ---------------------------------------------------------------------------
# TestIsolatedNodeTraversal
# ---------------------------------------------------------------------------


class TestIsolatedNodeTraversal:
    def test_isolated_anchor_gives_empty_lists(self) -> None:
        """Anchor with no outgoing edges → empty triples and chunk_ids."""
        session = MagicMock()
        session.run.return_value = []
        result = traverse_from_anchors(["ISOLATED_001"], session)
        assert result.triples == []
        assert result.chunk_ids == []

    def test_return_type_is_correct(self) -> None:
        """Return value is always a GraphTraversalResult regardless of graph state."""
        session = MagicMock()
        session.run.return_value = []
        result = traverse_from_anchors(["ANY_ID"], session)
        assert isinstance(result, GraphTraversalResult)
