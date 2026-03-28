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

from app.pipelines.graph_traversal import (
    GraphTraversalResult,
    Triple,
    _expand_coparty,
    _expand_director_of,
    traverse_from_anchors,
)

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

    def test_max_hops_one_issues_three_db_calls(self) -> None:
        """max_hops=1 issues exactly three session.run calls: main query + director-of + coparty."""
        session = MagicMock()
        session.run.return_value = []
        traverse_from_anchors(["C1"], session, max_hops=1)
        assert session.run.call_count == 3


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


# ---------------------------------------------------------------------------
# TestDirectorOfInboundExpansion
# ---------------------------------------------------------------------------


class TestDirectorOfInboundExpansion:
    def test_director_of_inbound_triples_included(self) -> None:
        """Company anchor: DIRECTOR_OF inbound triples appear in traverse result.

        The edge is (Person)-[:DIRECTOR_OF]->(Company), so the main outbound
        query never sees it.  _expand_director_of must surface it.
        """
        session = MagicMock()
        # First call: main hop query (no outbound edges from this anchor)
        # Second call: _expand_director_of (one inbound DIRECTOR_OF edge)
        # Third call: _expand_coparty (no co-parties for this anchor)
        session.run.side_effect = [
            [],
            [
                {
                    "src": "Jane Doe",
                    "rel": "DIRECTOR_OF",
                    "dst": "Acme Ltd",
                    "chunk_id": None,
                }
            ],
            [],
        ]
        result = traverse_from_anchors(["COMP_001"], session, max_hops=1)
        assert Triple(src="Jane Doe", rel="DIRECTOR_OF", dst="Acme Ltd") in result.triples

    def test_director_of_expansion_not_called_when_node_ids_empty(self) -> None:
        """Empty node_ids: traverse returns early — no DB calls at all."""
        session = MagicMock()
        result = traverse_from_anchors([], session)
        assert result.triples == []
        # Early-return path must not call the DB for either the main query or
        # the director-of expansion.
        session.run.assert_not_called()

    def test_director_of_duplicate_rows_deduplicated(self) -> None:
        """Duplicate DIRECTOR_OF rows from the expansion are collapsed to one triple."""
        session = MagicMock()
        session.run.side_effect = [
            [],  # main query: no outbound edges
            [
                {
                    "src": "Jane Doe",
                    "rel": "DIRECTOR_OF",
                    "dst": "Acme Ltd",
                    "chunk_id": None,
                },
                {
                    "src": "Jane Doe",
                    "rel": "DIRECTOR_OF",
                    "dst": "Acme Ltd",
                    "chunk_id": None,
                },
            ],
            [],  # _expand_coparty: no co-parties
        ]
        result = traverse_from_anchors(["COMP_001"], session)
        director_triples = [
            t for t in result.triples if t.rel == "DIRECTOR_OF"
        ]
        assert len(director_triples) == 1

    def test_director_of_does_not_duplicate_triple_from_main_query(self) -> None:
        """If both queries return the same triple, it appears once in the result."""
        triple_row = {
            "src": "Jane Doe",
            "rel": "DIRECTOR_OF",
            "dst": "Acme Ltd",
            "chunk_id": None,
        }
        session = MagicMock()
        session.run.side_effect = [
            [triple_row],  # main query also returned this triple
            [triple_row],  # expansion returns the same one
            [],  # _expand_coparty: no co-parties
        ]
        result = traverse_from_anchors(["COMP_001"], session, max_hops=1)
        assert result.triples.count(
            Triple(src="Jane Doe", rel="DIRECTOR_OF", dst="Acme Ltd")
        ) == 1

    def test_expand_director_of_returns_empty_for_no_rows(self) -> None:
        """_expand_director_of returns an empty list when session yields no rows."""
        session = MagicMock()
        session.run.return_value = []
        result = _expand_director_of(["COMP_001"], session)
        assert result == []

    def test_expand_director_of_returns_triple_list(self) -> None:
        """_expand_director_of returns a list of Triple instances."""
        session = MagicMock()
        session.run.return_value = [
            {
                "src": "Alice Smith",
                "rel": "DIRECTOR_OF",
                "dst": "Beta Corp",
                "chunk_id": None,
            }
        ]
        result = _expand_director_of(["COMP_002"], session)
        assert len(result) == 1
        assert isinstance(result[0], Triple)
        assert result[0].rel == "DIRECTOR_OF"


# ---------------------------------------------------------------------------
# TestCopartyExpansion
# ---------------------------------------------------------------------------


class TestCopartyExpansion:
    def test_coparty_expansion_returns_supplier_triples(self) -> None:
        """Co-party supplier sharing a contract appears as a PARTY_TO triple.

        (Supplier)-[:PARTY_TO]->(Contract)<-[:PARTY_TO]-(Anchor) means the
        supplier triple (supplier.name, PARTY_TO, contract_id) must appear in
        the result so downstream DIRECTOR_OF expansion can find the supplier's
        directors.
        """
        session = MagicMock()
        session.run.side_effect = [
            [],   # main hop query: no outbound edges from anchor
            [],   # _expand_director_of: anchor has no directors
            [     # _expand_coparty: one co-party found
                {
                    "src": "Supplier Co",
                    "rel": "PARTY_TO",
                    "dst": "CTR-001",
                    "chunk_id": None,
                }
            ],
        ]
        result = traverse_from_anchors(["COMP_ANCHOR"], session, max_hops=1)
        assert Triple(src="Supplier Co", rel="PARTY_TO", dst="CTR-001") in result.triples

    def test_coparty_duplicate_rows_deduplicated(self) -> None:
        """Identical co-party rows returned by the query collapse to one triple."""
        session = MagicMock()
        session.run.side_effect = [
            [],  # main hop query
            [],  # _expand_director_of
            [    # _expand_coparty: duplicate rows from multi-path Cypher expansion
                {"src": "Supplier Co", "rel": "PARTY_TO", "dst": "CTR-001", "chunk_id": None},
                {"src": "Supplier Co", "rel": "PARTY_TO", "dst": "CTR-001", "chunk_id": None},
            ],
        ]
        result = traverse_from_anchors(["COMP_ANCHOR"], session, max_hops=1)
        coparty_triples = [t for t in result.triples if t.src == "Supplier Co"]
        assert len(coparty_triples) == 1

    def test_coparty_does_not_duplicate_triple_already_in_main_query(self) -> None:
        """If the main query already returned a co-party triple, it appears only once."""
        shared_row = {"src": "Supplier Co", "rel": "PARTY_TO", "dst": "CTR-001", "chunk_id": None}
        session = MagicMock()
        session.run.side_effect = [
            [shared_row],  # main hop query also returned the co-party edge
            [],            # _expand_director_of
            [shared_row],  # _expand_coparty returns the same triple
        ]
        result = traverse_from_anchors(["COMP_ANCHOR"], session, max_hops=1)
        assert result.triples.count(Triple(src="Supplier Co", rel="PARTY_TO", dst="CTR-001")) == 1

    def test_expand_coparty_returns_empty_for_no_rows(self) -> None:
        """_expand_coparty returns an empty list when session yields no rows."""
        session = MagicMock()
        session.run.return_value = []
        result = _expand_coparty(["COMP_001"], session)
        assert result == []

    def test_expand_coparty_returns_triple_list(self) -> None:
        """_expand_coparty returns a list of Triple instances with rel PARTY_TO."""
        session = MagicMock()
        session.run.return_value = [
            {"src": "Gamma Ltd", "rel": "PARTY_TO", "dst": "CTR-999", "chunk_id": None}
        ]
        result = _expand_coparty(["COMP_003"], session)
        assert len(result) == 1
        assert isinstance(result[0], Triple)
        assert result[0].rel == "PARTY_TO"
        assert result[0].src == "Gamma Ltd"
        assert result[0].dst == "CTR-999"
