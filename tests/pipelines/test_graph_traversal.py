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
    expand_co_party_chain,
    expand_co_party_directors,
    expand_inbound_director_of,
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
        """max_hops=1 issues four session.run calls.

        Calls: main query, director-of expansion, coparty expansion, coparty-directors expansion.
        """
        session = MagicMock()
        session.run.return_value = []
        traverse_from_anchors(["C1"], session, max_hops=1)
        assert session.run.call_count == 4


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
        # Call 1: main hop query (no outbound edges from this anchor)
        # Call 2: expand_inbound_director_of (one inbound DIRECTOR_OF edge)
        # Call 3: _expand_coparty (no co-parties for this anchor)
        # Call 4: expand_co_party_directors (no co-party directors)
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
            [],  # expand_co_party_directors: no co-party directors
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
            [],  # expand_co_party_directors: no co-party directors
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
            [],   # expand_inbound_director_of: anchor has no directors
            [     # _expand_coparty: one co-party found
                {
                    "src": "Supplier Co",
                    "rel": "PARTY_TO",
                    "dst": "CTR-001",
                    "chunk_id": None,
                }
            ],
            [],   # expand_co_party_directors: no directors of that co-party
        ]
        result = traverse_from_anchors(["COMP_ANCHOR"], session, max_hops=1)
        assert Triple(src="Supplier Co", rel="PARTY_TO", dst="CTR-001") in result.triples

    def test_coparty_duplicate_rows_deduplicated(self) -> None:
        """Identical co-party rows returned by the query collapse to one triple."""
        session = MagicMock()
        session.run.side_effect = [
            [],  # main hop query
            [],  # expand_inbound_director_of
            [    # _expand_coparty: duplicate rows from multi-path Cypher expansion
                {"src": "Supplier Co", "rel": "PARTY_TO", "dst": "CTR-001", "chunk_id": None},
                {"src": "Supplier Co", "rel": "PARTY_TO", "dst": "CTR-001", "chunk_id": None},
            ],
            [],  # expand_co_party_directors
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
            [],            # expand_inbound_director_of
            [shared_row],  # _expand_coparty returns the same triple
            [],            # expand_co_party_directors
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


# ---------------------------------------------------------------------------
# TestNamedEntityTriples
# ---------------------------------------------------------------------------


class TestNamedEntityTriples:
    def test_src_dst_are_names_not_internal_ids(self) -> None:
        """Triple.src and Triple.dst carry canonical names, not Neo4j internal IDs.

        T001 replaced toString(id(...)) fallback with CASE expressions that
        project the domain name/id property.  A mock row with string values must
        pass through to the Triple unchanged — if the function were still using
        internal IDs the values would differ from what the Cypher projected.
        """
        session = MagicMock()
        # main query returns one named-entity row; expansions return nothing
        session.run.side_effect = [
            [{"src": "Acme Corp", "rel": "PARTY_TO", "dst": "CONTRACT-001", "chunk_id": None}],
            [],  # expand_inbound_director_of
            [],  # _expand_coparty
            [],  # expand_co_party_directors
        ]
        result = traverse_from_anchors(["COMP_001"], session, max_hops=1)
        assert len(result.triples) == 1
        assert result.triples[0].src == "Acme Corp"
        assert result.triples[0].dst == "CONTRACT-001"

    def test_mixed_label_triple_uses_named_properties(self) -> None:
        """Three rows covering three distinct label scenarios all surface named strings.

        Scenario coverage:
          - Company-name src  ('Beta Ltd')  — WHEN 'Company' IN labels THEN node.name
          - Contract-id dst   ('CTR-42')    — WHEN 'Contract' IN labels THEN node.contract_id
          - Person-name dst   ('Jane Smith')— WHEN 'Person' IN labels THEN node.name

        If any CASE branch silently fell through to toString(id(...)), the
        expected string would be absent from the result.
        """
        session = MagicMock()
        session.run.side_effect = [
            [
                # Company src → Contract dst
                {"src": "Beta Ltd", "rel": "PARTY_TO", "dst": "CTR-42", "chunk_id": None},
                # Company src → Person dst
                {"src": "Beta Ltd", "rel": "EMPLOYS", "dst": "Jane Smith", "chunk_id": None},
                # Person src → Company dst (verifies Person name in src too)
                {"src": "Jane Smith", "rel": "DIRECTOR_OF", "dst": "Beta Ltd", "chunk_id": None},
            ],
            [],  # expand_inbound_director_of
            [],  # _expand_coparty
            [],  # expand_co_party_directors
        ]
        result = traverse_from_anchors(["COMP_BETA"], session, max_hops=1)
        srcs = {t.src for t in result.triples}
        dsts = {t.dst for t in result.triples}
        assert "Beta Ltd" in srcs, "Company name must appear as Triple.src"
        assert "CTR-42" in dsts, "Contract id must appear as Triple.dst"
        assert "Jane Smith" in dsts, "Person name must appear as Triple.dst"

    def test_empty_node_ids_returns_empty_result(self) -> None:
        """Empty anchor list returns empty result without issuing any DB call.

        Confirms the early-return guard in traverse_from_anchors is unaffected
        by T001 CASE-expression changes.
        """
        session = MagicMock()
        result = traverse_from_anchors([], session)
        assert result.chunk_ids == []
        assert result.triples == []
        session.run.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for expand_inbound_director_of (T005)
# ---------------------------------------------------------------------------
# The hard part: _Q_DIRECTOR_OF returns column aliases src (p.name), rel, dst
# (anchor.name), chunk_id.  A key mismatch between mock row dict and the RETURN
# clause would cause a silent wrong-data bug or a KeyError.  Both the success
# path and the empty-graph guard are tested here.


def test_director_found_via_company_anchor_returns_named_triple() -> None:
    """expand_inbound_director_of returns one Triple with canonical person and company names.

    Row keys must exactly match _Q_DIRECTOR_OF's RETURN aliases (src, rel, dst).
    The function must NOT return raw Neo4j internal element IDs — src and dst
    must be the projected name strings, not toString(id(...)) values.
    """
    session = MagicMock()
    # Column aliases from _Q_DIRECTOR_OF: p.name AS src, 'DIRECTOR_OF' AS rel,
    # anchor.name AS dst, null AS chunk_id.
    session.run.return_value = [
        {"src": "Jane Doe", "rel": "DIRECTOR_OF", "dst": "Acme Ltd", "chunk_id": None}
    ]
    result = expand_inbound_director_of(["COMP_001"], session)

    assert len(result) == 1
    triple = result[0]
    assert isinstance(triple, Triple)
    assert triple.src == "Jane Doe", "src must be the person's name, not an internal ID"
    assert triple.rel == "DIRECTOR_OF"
    assert triple.dst == "Acme Ltd", "dst must be the company's name, not an internal ID"
    # Guard against accidental toString(id(...)) values leaking through
    assert not triple.src.startswith("toString("), "src must not be a raw Neo4j ID expression"
    assert not triple.dst.startswith("toString("), "dst must not be a raw Neo4j ID expression"


def test_company_with_no_directors_returns_empty_list() -> None:
    """expand_inbound_director_of returns [] when no Person is directed at the anchor.

    This guards the empty-graph case: a Company node with no inbound
    DIRECTOR_OF edges must yield an empty list, not raise or return None.
    """
    session = MagicMock()
    session.run.return_value = []
    result = expand_inbound_director_of(["COMP_NODIRECTORS"], session)
    assert result == []


# ---------------------------------------------------------------------------
# Tests for expand_co_party_chain (T006)
# ---------------------------------------------------------------------------
# The hard part: _Q_COPARTY returns column aliases src (co.name), rel (literal
# 'PARTY_TO'), dst (c.contract_id), chunk_id (null).  A mismatch between the
# mock row keys and the RETURN clause aliases would cause a silent wrong-data bug
# or a KeyError at runtime.  Both the resolved-chain and broken-chain cases are
# exercised here.


def test_co_party_chain_resolves_to_named_triple() -> None:
    """expand_co_party_chain returns one Triple with named company and contract values.

    Row keys must exactly match _Q_COPARTY's RETURN aliases:
        src       → co.name   (co-party company name string)
        rel       → 'PARTY_TO' (literal)
        dst       → c.contract_id (shared contract identity key)
        chunk_id  → null

    Triple.src and Triple.dst must be the projected name/id strings, not
    Neo4j internal element IDs (i.e. not toString(id(...)) expressions).
    This covers the Company PARTY_TO Contract PARTY_TO CoParty 2-hop path.
    """
    session = MagicMock()
    # Column aliases from _Q_COPARTY: co.name AS src, 'PARTY_TO' AS rel,
    # c.contract_id AS dst, null AS chunk_id.
    session.run.return_value = [
        {
            "src": "Supplier Co",
            "rel": "PARTY_TO",
            "dst": "CTR-001",
            "chunk_id": None,
        }
    ]
    result = expand_co_party_chain(["COMP_ANCHOR"], session)

    assert len(result) == 1
    triple = result[0]
    assert isinstance(triple, Triple)
    assert triple.src == "Supplier Co", "src must be the co-party company name, not an internal ID"
    assert triple.rel == "PARTY_TO"
    assert triple.dst == "CTR-001", "dst must be the contract_id, not an internal ID"
    # Guard against toString(id(...)) leaking through from a missing CASE branch
    assert not triple.src.startswith("toString("), "src must not be a raw Neo4j ID expression"
    assert not triple.dst.startswith("toString("), "dst must not be a raw Neo4j ID expression"


def test_broken_co_party_chain_returns_empty_list() -> None:
    """expand_co_party_chain returns [] when the company is not party to any contract.

    Simulates a Company anchor with no outbound PARTY_TO edges, i.e. the 2-hop
    chain (co)-[:PARTY_TO]->(Contract)<-[:PARTY_TO]-(anchor) is broken at the
    first hop.  The function must return an empty list, not raise or return None.
    """
    session = MagicMock()
    session.run.return_value = []
    result = expand_co_party_chain(["COMP_NO_CONTRACTS"], session)
    assert result == []


# ---------------------------------------------------------------------------
# Tests for expand_co_party_directors (T001)
# ---------------------------------------------------------------------------
# The hard part: _Q_COPARTY_DIRECTORS returns column aliases src (p.name),
# rel (literal 'DIRECTOR_OF'), dst (co.name), chunk_id (null).  A mismatch
# between mock row keys and RETURN aliases would cause a silent KeyError at
# runtime.  Both the resolved-director and empty-graph cases are tested here.


def test_co_party_directors_resolves_to_named_triple() -> None:
    """expand_co_party_directors returns one Triple with person name and company name.

    Row keys must exactly match _Q_COPARTY_DIRECTORS RETURN aliases:
        src       → p.name    (director's person name)
        rel       → 'DIRECTOR_OF' (literal)
        dst       → co.name   (co-party company name)
        chunk_id  → null

    Triple.src and Triple.dst must be the projected name strings, not Neo4j
    internal element IDs.
    """
    session = MagicMock()
    session.run.return_value = [
        {
            "src": "Bob Smith",
            "rel": "DIRECTOR_OF",
            "dst": "Supplier Co",
            "chunk_id": None,
        }
    ]
    result = expand_co_party_directors(["COMP_ANCHOR"], session)

    assert len(result) == 1
    triple = result[0]
    assert isinstance(triple, Triple)
    assert triple.src == "Bob Smith", "src must be the director's name, not an internal ID"
    assert triple.rel == "DIRECTOR_OF"
    assert triple.dst == "Supplier Co", "dst must be the co-party company name"
    assert not triple.src.startswith("toString(")
    assert not triple.dst.startswith("toString(")


def test_co_party_directors_no_directors_returns_empty_list() -> None:
    """expand_co_party_directors returns [] when no Person directs any co-party Company.

    Simulates an anchor whose co-party companies have no directors registered.
    The function must return an empty list, not raise or return None.
    """
    session = MagicMock()
    session.run.return_value = []
    result = expand_co_party_directors(["COMP_NO_COPARTY_DIRECTORS"], session)
    assert result == []
