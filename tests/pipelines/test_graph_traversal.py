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

    def test_max_hops_one_issues_four_db_calls(self) -> None:
        """max_hops=1 issues five session.run calls.

        Calls: main query, director-of expansion, coparty expansion,
        coparty-directors expansion, _collect_anchor_chunks.
        """
        session = MagicMock()
        session.run.return_value = []
        traverse_from_anchors(["C1"], session, max_hops=1)
        assert session.run.call_count == 5


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
        # Call 5: _collect_anchor_chunks (no chunks for this anchor)
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
            [],  # _collect_anchor_chunks: no chunks
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
            [],  # _collect_anchor_chunks: no chunks
        ]
        result = traverse_from_anchors(["COMP_001"], session, max_hops=1)
        assert result.triples.count(
            Triple(src="Jane Doe", rel="DIRECTOR_OF", dst="Acme Ltd")
        ) == 1

    def test_expand_director_of_returns_empty_for_no_rows(self) -> None:
        """expand_inbound_director_of returns an empty list when session yields no rows."""
        session = MagicMock()
        session.run.return_value = []
        result = expand_inbound_director_of(["COMP_001"], session)
        assert result == []

    def test_expand_director_of_returns_triple_list(self) -> None:
        """expand_inbound_director_of returns a list of Triple instances."""
        session = MagicMock()
        session.run.return_value = [
            {
                "src": "Alice Smith",
                "rel": "DIRECTOR_OF",
                "dst": "Beta Corp",
                "chunk_id": None,
            }
        ]
        result = expand_inbound_director_of(["COMP_002"], session)
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
            [],   # _collect_anchor_chunks: no chunks
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
            [],  # _collect_anchor_chunks: no chunks
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
            [],            # _collect_anchor_chunks: no chunks
        ]
        result = traverse_from_anchors(["COMP_ANCHOR"], session, max_hops=1)
        assert result.triples.count(Triple(src="Supplier Co", rel="PARTY_TO", dst="CTR-001")) == 1

    def test_expand_coparty_returns_empty_for_no_rows(self) -> None:
        """expand_co_party_chain returns an empty list when session yields no rows."""
        session = MagicMock()
        session.run.return_value = []
        result = expand_co_party_chain(["COMP_001"], session)
        assert result == []

    def test_expand_coparty_returns_triple_list(self) -> None:
        """expand_co_party_chain returns a list of Triple instances with rel PARTY_TO."""
        session = MagicMock()
        session.run.return_value = [
            {"src": "Gamma Ltd", "rel": "PARTY_TO", "dst": "CTR-999", "chunk_id": None}
        ]
        result = expand_co_party_chain(["COMP_003"], session)
        assert len(result) == 1
        assert isinstance(result[0], Triple)
        assert result[0].rel == "PARTY_TO"
        assert result[0].src == "Gamma Ltd"
        assert result[0].dst == "CTR-999"


# ---------------------------------------------------------------------------
# TestCoPartyDirectorsExpansion
# ---------------------------------------------------------------------------
# The risky paths: (1) column-alias fidelity — mock row keys must exactly
# match _Q_COPARTY_DIRECTORS RETURN aliases (src, rel, dst, chunk_id); a wrong
# key causes a silent KeyError only at runtime. (2) deduplication — the same
# person directing multiple co-parties sharing one anchor contract produces
# duplicate rows; the seen: set[Triple] guard inside expand_co_party_directors
# must collapse them before they reach the caller.


class TestCoPartyDirectorsExpansion:
    def test_co_party_directors_returns_named_triples(self) -> None:
        """expand_co_party_directors maps _Q_COPARTY_DIRECTORS row aliases to Triple fields.

        Row keys must exactly match the RETURN aliases (src, rel, dst, chunk_id).
        A wrong key causes a silent KeyError at runtime — this test is the
        safety net for the column-alias contract.
        """
        session = MagicMock()
        # Column aliases from _Q_COPARTY_DIRECTORS:
        #   p.name AS src, 'DIRECTOR_OF' AS rel, co.name AS dst, null AS chunk_id
        session.run.return_value = [
            {
                "src": "Bob Smith",
                "rel": "DIRECTOR_OF",
                "dst": "Supplier Co",
                "chunk_id": None,
            }
        ]
        result = expand_co_party_directors(["ANCHOR_001"], session)

        assert len(result) == 1
        triple = result[0]
        assert isinstance(triple, Triple)
        assert triple.src == "Bob Smith", "src must be the director's name, not an internal ID"
        assert triple.rel == "DIRECTOR_OF"
        assert triple.dst == "Supplier Co", "dst must be the co-party company name"
        assert not triple.src.startswith("toString(")
        assert not triple.dst.startswith("toString(")

    def test_co_party_directors_duplicate_rows_deduplicated(self) -> None:
        """Duplicate rows from the query collapse to one Triple via the internal seen set.

        _Q_COPARTY_DIRECTORS can return duplicate rows when a director sits on
        multiple contracts that all share the same anchor, each producing an
        identical (p.name, DIRECTOR_OF, co.name) projection.  The seen: set[Triple]
        guard inside expand_co_party_directors must collapse them before they
        reach the caller.  Two identical rows in → one Triple out.
        """
        session = MagicMock()
        dup_row = {
            "src": "Alice Jones",
            "rel": "DIRECTOR_OF",
            "dst": "Beta Corp",
            "chunk_id": None,
        }
        session.run.return_value = [dup_row, dup_row]
        result = expand_co_party_directors(["ANCHOR_001"], session)

        assert len(result) == 1
        assert result[0] == Triple(src="Alice Jones", rel="DIRECTOR_OF", dst="Beta Corp")

    def test_co_party_directors_triples_appear_in_traverse_result(self) -> None:
        """Co-party director triples from expand_co_party_directors merge into traverse output.

        Call order: main query (1) → expand_inbound_director_of (2) →
        _expand_coparty (3) → expand_co_party_directors (4).  The fourth call's
        triples must appear in the final GraphTraversalResult.
        """
        session = MagicMock()
        session.run.side_effect = [
            [],  # main hop query: no outbound edges from anchor
            [],  # expand_inbound_director_of: no direct directors of anchor
            [],  # _expand_coparty: co-party PARTY_TO triples (not the focus here)
            [    # expand_co_party_directors: one director of a co-party company
                {
                    "src": "Carol White",
                    "rel": "DIRECTOR_OF",
                    "dst": "Gamma Ltd",
                    "chunk_id": None,
                }
            ],
            [],  # _collect_anchor_chunks: no chunks
        ]
        result = traverse_from_anchors(["ANCHOR_001"], session, max_hops=1)
        assert Triple(src="Carol White", rel="DIRECTOR_OF", dst="Gamma Ltd") in result.triples


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
            [],  # _collect_anchor_chunks: no chunks
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
            [],  # _collect_anchor_chunks: no chunks
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
    """expand_co_party_directors returns [] when no Person directs any co-party Company."""
    session = MagicMock()
    session.run.return_value = []
    result = expand_co_party_directors(["COMP_NO_COPARTY_DIRECTORS"], session)
    assert result == []


# ---------------------------------------------------------------------------
# Tests for Clause node dst projection (T003)
# ---------------------------------------------------------------------------
# The risk: Clause nodes lack a CASE branch in _Q_1HOP and _Q_2HOP dst
# expressions, so the ELSE toString(id(...)) fallback fires and dst becomes a
# numeric string like '42' instead of a domain identifier like 'CL-007'.
# These tests verify the Cypher constants directly — mock sessions cannot
# catch a missing branch because they return whatever value you provide.


def test_clause_node_1hop_dst_returns_clause_id() -> None:
    """_Q_1HOP dst CASE must have a Clause branch returning b.clause_id.

    Without this branch the ELSE fallback fires, returning toString(id(b)) —
    a numeric string instead of a clause identifier like 'CL-007'.
    """
    from app.pipelines.graph_traversal import _Q_1HOP

    assert "WHEN 'Clause' IN labels(b) THEN b.clause_id" in _Q_1HOP


def test_clause_node_2hop_dst_returns_clause_id() -> None:
    """_Q_2HOP dst CASE must have a Clause branch returning endNode(r).clause_id.

    Without this branch, Clause nodes in a 2-hop path return a numeric
    toString(id(endNode(r))) value instead of the clause_id domain key.
    """
    from app.pipelines.graph_traversal import _Q_2HOP

    assert (
        "WHEN 'Clause' IN labels(endNode(r)) THEN endNode(r).clause_id"
        in _Q_2HOP
    )


def test_clause_node_1hop_dst_not_internal_id() -> None:
    """The Clause branch must appear before the ELSE fallback in _Q_1HOP dst CASE.

    Neo4j internal IDs are integers; toString(id(b)) produces a numeric string
    like '42'. The Clause branch must precede the ELSE so Clause nodes return
    their clause_id property, not a raw integer-like string.
    """
    from app.pipelines.graph_traversal import _Q_1HOP

    clause_branch = "WHEN 'Clause' IN labels(b) THEN b.clause_id"
    else_fallback = "ELSE toString(id(b))"
    assert clause_branch in _Q_1HOP, "Clause branch absent — dst falls back to numeric ID"
    clause_pos = _Q_1HOP.find(clause_branch)
    else_pos = _Q_1HOP.find(else_fallback)
    assert clause_pos < else_pos, "Clause branch must precede ELSE to avoid numeric fallback"


# Tests for Address node dst projection (T004)
# ---------------------------------------------------------------------------
# The risk: Address nodes lack a CASE branch in _Q_1HOP and _Q_2HOP dst
# expressions, so the ELSE toString(id(...)) fallback fires and dst becomes a
# numeric string like '42' instead of the city display label like 'Berlin'.
# These tests verify the Cypher constants directly — mock sessions cannot
# catch a missing branch because they return whatever value you provide.


def test_address_node_1hop_dst_returns_city() -> None:
    """_Q_1HOP dst CASE must have an Address branch returning b.city.

    Without this branch the ELSE fallback fires, returning toString(id(b)) —
    a numeric string instead of a city name like 'Berlin'.
    """
    from app.pipelines.graph_traversal import _Q_1HOP

    assert "WHEN 'Address' IN labels(b) THEN b.city" in _Q_1HOP


def test_address_node_2hop_dst_returns_city() -> None:
    """_Q_2HOP dst CASE must have an Address branch returning endNode(r).city.

    Without this branch, Address nodes in a 2-hop path return a numeric
    toString(id(endNode(r))) value instead of the city display label.
    """
    from app.pipelines.graph_traversal import _Q_2HOP

    assert (
        "WHEN 'Address' IN labels(endNode(r)) THEN endNode(r).city"
        in _Q_2HOP
    )


def test_address_node_1hop_dst_not_internal_id() -> None:
    """The Address branch must appear before the ELSE fallback in _Q_1HOP dst CASE.

    Neo4j internal IDs are integers; toString(id(b)) produces a numeric string
    like '42'. The Address branch must precede the ELSE so Address nodes return
    their city property, not a raw integer-like string.
    """
    from app.pipelines.graph_traversal import _Q_1HOP

    address_branch = "WHEN 'Address' IN labels(b) THEN b.city"
    else_fallback = "ELSE toString(id(b))"
    assert address_branch in _Q_1HOP, "Address branch absent — dst falls back to numeric ID"
    address_pos = _Q_1HOP.find(address_branch)
    else_pos = _Q_1HOP.find(else_fallback)
    assert address_pos < else_pos, "Address branch must precede ELSE to avoid numeric fallback"


def test_co_party_directors_deduped_against_director_of_in_traverse() -> None:
    """A triple from both expand_inbound_director_of and expand_co_party_directors appears once.

    Simulates the case where the same (person, DIRECTOR_OF, company) row is
    returned by both expansion functions — e.g. an unusual graph structure where
    the same person and company name combination surfaces from two different
    Cypher traversal paths.  The shared seen_triples guard in traverse_from_anchors
    must absorb the duplicate so only one Triple appears in the result.
    """
    shared_row = {
        "src": "Shared Director",
        "rel": "DIRECTOR_OF",
        "dst": "Overlap Corp",
        "chunk_id": None,
    }
    session = MagicMock()
    session.run.side_effect = [
        [],             # main hop query
        [shared_row],   # expand_inbound_director_of
        [],             # _expand_coparty
        [shared_row],   # expand_co_party_directors returns the same triple
        [],             # _collect_anchor_chunks: no chunks
    ]
    result = traverse_from_anchors(["ANCHOR_001"], session, max_hops=1)
    assert result.triples.count(
        Triple(src="Shared Director", rel="DIRECTOR_OF", dst="Overlap Corp")
    ) == 1


# ---------------------------------------------------------------------------
# Tests for _collect_anchor_chunks (T015)
# ---------------------------------------------------------------------------
# The risk: if the 5th session.run call is silently dropped, chunk_ids is always
# empty regardless of what the graph contains.  These tests verify the observable
# output (GraphTraversalResult.chunk_ids) when the 5th call returns chunk rows.
#
# Side-effect ordering is the hard part: all three tests that use non-empty
# node_ids need exactly 5 entries.  Fewer causes StopIteration on the 5th call.


def test_chunk_ids_non_empty_when_chunk_query_returns_rows() -> None:
    """5th session.run returns one chunk row → chunk_ids contains that chunk_id.

    The first four calls return empty lists so that any chunk_ids in the result
    can only come from _collect_anchor_chunks (the 5th call).  If the 5th call
    is ever dropped, chunk_ids stays empty and this test fails.
    """
    session = MagicMock()
    session.run.side_effect = [
        [],   # main hop query: no outbound edges
        [],   # expand_inbound_director_of: no directors
        [],   # expand_co_party_chain: no co-parties
        [],   # expand_co_party_directors: no co-party directors
        [{"chunk_id": "CL001_c0"}],  # _collect_anchor_chunks: one chunk
    ]
    result = traverse_from_anchors(["CONTRACT_001"], session)
    assert result.chunk_ids == ["CL001_c0"]


def test_chunk_ids_empty_when_node_ids_is_empty() -> None:
    """Empty node_ids triggers the early-return path — no DB calls, chunk_ids empty."""
    session = MagicMock()
    result = traverse_from_anchors([], session)
    assert result.chunk_ids == []
    session.run.assert_not_called()


def test_multi_anchor_chunk_ids_all_collected() -> None:
    """5th call returns two chunk rows from different anchors → both appear, no duplicates.

    Also verifies deduplication: if both rows carry the same chunk_id it must
    appear exactly once.  Two distinct IDs must both be present.
    """
    session = MagicMock()
    session.run.side_effect = [
        [],   # main hop query: no outbound edges
        [],   # expand_inbound_director_of: no directors
        [],   # expand_co_party_chain: no co-parties
        [],   # expand_co_party_directors: no co-party directors
        [     # _collect_anchor_chunks: two chunks from two different anchors
            {"chunk_id": "CL001_c0"},
            {"chunk_id": "CL002_c0"},
        ],
    ]
    result = traverse_from_anchors(["CONTRACT_001", "CONTRACT_002"], session)
    assert "CL001_c0" in result.chunk_ids
    assert "CL002_c0" in result.chunk_ids
    assert len(result.chunk_ids) == 2  # no duplicates


# ---------------------------------------------------------------------------
# Tests for Clause/Address dst string plumbing through traverse_from_anchors
# (T016 — verifies BUG 2 and BUG 3 Python-layer row→Triple mapping)
# ---------------------------------------------------------------------------
# NOTE on test strategy: the CASE expressions that select clause_id and city
# run inside Neo4j, not in Python.  MagicMock returns whatever string value we
# inject, so these tests cannot prove the Cypher constant is correct — they
# prove only that the Python plumbing (row["dst"] → Triple.dst) is transparent.
# Full regression proof requires either a Cypher integration test against a
# live Neo4j instance or code review of the _Q_1HOP / _Q_2HOP constants.
# The negative guard (`assert not Triple.dst.startswith('toString(')`) catches
# a future refactor that accidentally injects a toString(id(...)) literal as
# a mock return value, making a broken Cypher constant look like a passing test.


def test_clause_dst_returns_clause_id_string_in_one_hop() -> None:
    """1-hop traversal: a row with a clause_id string dst passes through unchanged.

    Simulates _Q_1HOP returning a Clause-typed row where the Cypher CASE
    expression has projected b.clause_id correctly.  The Python layer must
    not transform or discard the string; Triple.dst must equal the injected value.
    """
    session = MagicMock()
    session.run.side_effect = [
        # Call 1: main _Q_1HOP query — Clause node as dst
        [{"src": "Acme Corp", "rel": "HAS_CLAUSE", "dst": "CL-007", "chunk_id": None}],
        [],   # Call 2: expand_inbound_director_of
        [],   # Call 3: expand_co_party_chain
        [],   # Call 4: expand_co_party_directors
        [],   # Call 5: _collect_anchor_chunks
    ]
    result = traverse_from_anchors(["CONTRACT_001"], session, max_hops=1)
    assert len(result.triples) == 1
    triple = result.triples[0]
    assert triple.dst == "CL-007", "Clause node dst must be the clause_id string"
    # Guard: catches a refactor that re-introduces toString(id(...)) as injected value.
    # A real CASE-branch regression is only detectable via Cypher integration tests.
    assert not triple.dst.startswith("toString("), (
        "dst must not be a toString(id(...)) expression; "
        "full proof requires a Cypher integration test"
    )


def test_clause_dst_returns_clause_id_string_in_two_hop() -> None:
    """2-hop traversal: a row with a clause_id string dst passes through unchanged.

    _Q_1HOP and _Q_2HOP are independent Cypher constants; a missing Clause
    branch in _Q_2HOP would not be caught by the 1-hop test above.
    """
    session = MagicMock()
    session.run.side_effect = [
        # Call 1: main _Q_2HOP query — Clause node as dst at hop depth 1 or 2
        [{"src": "Beta Ltd", "rel": "HAS_CLAUSE", "dst": "CL-042", "chunk_id": None}],
        [],   # Call 2: expand_inbound_director_of
        [],   # Call 3: expand_co_party_chain
        [],   # Call 4: expand_co_party_directors
        [],   # Call 5: _collect_anchor_chunks
    ]
    result = traverse_from_anchors(["CONTRACT_002"], session, max_hops=2)
    assert len(result.triples) == 1
    triple = result.triples[0]
    assert triple.dst == "CL-042", "Clause node dst must be the clause_id string"
    assert not triple.dst.startswith("toString("), (
        "dst must not be a toString(id(...)) expression; "
        "full proof requires a Cypher integration test"
    )


def test_address_dst_returns_city_string_in_one_hop() -> None:
    """1-hop traversal: a row with a city string dst passes through unchanged.

    Simulates _Q_1HOP returning an Address-typed row where the Cypher CASE
    expression has projected b.city correctly.  The Python layer must not
    transform or discard the string; Triple.dst must equal the injected value.
    """
    session = MagicMock()
    session.run.side_effect = [
        # Call 1: main _Q_1HOP query — Address node as dst
        [{"src": "Gamma Corp", "rel": "REGISTERED_AT", "dst": "Berlin", "chunk_id": None}],
        [],   # Call 2: expand_inbound_director_of
        [],   # Call 3: expand_co_party_chain
        [],   # Call 4: expand_co_party_directors
        [],   # Call 5: _collect_anchor_chunks
    ]
    result = traverse_from_anchors(["COMP_GAMMA"], session, max_hops=1)
    assert len(result.triples) == 1
    triple = result.triples[0]
    assert triple.dst == "Berlin", "Address node dst must be the city string"
    # Guard: catches a refactor that re-introduces toString(id(...)) as injected value.
    # A real CASE-branch regression is only detectable via Cypher integration tests.
    assert not triple.dst.startswith("toString("), (
        "dst must not be a toString(id(...)) expression; "
        "full proof requires a Cypher integration test"
    )


def test_address_dst_returns_city_string_in_two_hop() -> None:
    """2-hop traversal: a row with a city string dst passes through unchanged.

    _Q_1HOP and _Q_2HOP are independent Cypher constants; a missing Address
    branch in _Q_2HOP would not be caught by the 1-hop test above.
    """
    session = MagicMock()
    session.run.side_effect = [
        # Call 1: main _Q_2HOP query — Address node as dst at hop depth 1 or 2
        [{"src": "Delta Inc", "rel": "REGISTERED_AT", "dst": "Amsterdam", "chunk_id": None}],
        [],   # Call 2: expand_inbound_director_of
        [],   # Call 3: expand_co_party_chain
        [],   # Call 4: expand_co_party_directors
        [],   # Call 5: _collect_anchor_chunks
    ]
    result = traverse_from_anchors(["COMP_DELTA"], session, max_hops=2)
    assert len(result.triples) == 1
    triple = result.triples[0]
    assert triple.dst == "Amsterdam", "Address node dst must be the city string"
    assert not triple.dst.startswith("toString("), (
        "dst must not be a toString(id(...)) expression; "
        "full proof requires a Cypher integration test"
    )
