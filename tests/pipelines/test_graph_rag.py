"""Tests for app.pipelines.graph_rag (T025.e).

Tests the run_graph_rag orchestrator by patching the four stage functions.
Each test verifies that run_graph_rag correctly wires stage outputs to stage
inputs and returns a GenerationResult.

Coverage map (15 tests):

entity_resolution (6):
  TestExactMatchFlowsToTraversal     — exact match node_id passed to traverse
  TestCaseInsensitiveMatchFlows      — case-insensitive match flows through
  TestPartialSubstringMatchFlows     — partial match score < 1.0 still flows
  TestZeroMatchesFallsBack           — empty entity list → fallback GenerationResult
  TestMultipleCandidatesTopK         — top_k_entities parameter forwarded
  TestSpecialCharactersInName        — special-char node_id flows through pipeline

graph_traversal (4):
  TestSingleHopTraversal             — 1-hop triples reach generate_answer
  TestTwoHopTraversal                — max_hops=2 forwarded to traverse_from_anchors
  TestNoPathReturnsEmptyTriples      — empty triples → generate_answer called with []
  TestIsolatedNodeTraversal          — node resolves, traverse returns empty result

constrained_vector_retrieval (3):
  TestChunksFilteredAdjacent  — chunk_ids from traversal passed to retrieve
  TestEmptyGraphEmptyChunks   — empty chunk_ids → retrieve returns []
  TestRankingPreservedInResult       — ranked chunks propagate to GenerationResult

integration_pipeline (2):
  TestEndToEndReturnsResult — all stages produce results → populated answer
  TestUnknownEntityFallsBack — resolve returns [] → fallback answer returned
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.pipelines.citation_generator import Citation, GenerationResult
from app.pipelines.constrained_retrieval import RankedChunk
from app.pipelines.entity_resolver import EntityMatch
from app.pipelines.graph_rag import run_graph_rag
from app.pipelines.graph_traversal import GraphTraversalResult, Triple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODULE = "app.pipelines.graph_rag"


def _entity(
    node_id: str, label: str = "Company", name: str = "Test Co", score: float = 1.0
) -> EntityMatch:
    return EntityMatch(node_id=node_id, label=label, name=name, score=score)


def _triple(src: str = "1", rel: str = "PARTY_TO", dst: str = "2") -> Triple:
    return Triple(src=src, rel=rel, dst=dst)


def _chunk(
    chunk_id: str = "CL001_c0", text: str = "Sample text.", score: float = 0.9
) -> RankedChunk:
    return RankedChunk(chunk_id=chunk_id, text=text, score=score)


def _traversal(
    chunk_ids: list[str] | None = None, triples: list[Triple] | None = None
) -> GraphTraversalResult:
    return GraphTraversalResult(
        chunk_ids=chunk_ids if chunk_ids is not None else [],
        triples=triples if triples is not None else [],
    )


def _gen_result(
    answer: str = "Answer.", citations: list[Citation] | None = None
) -> GenerationResult:
    return GenerationResult(
        answer=answer,
        text_citations=citations if citations is not None else [],
    )


# ---------------------------------------------------------------------------
# entity_resolution
# ---------------------------------------------------------------------------


class TestExactMatchFlowsToTraversal:
    def test_exact_match_node_id_passed_to_traverse(self) -> None:
        """Exact entity match: node_id is extracted and forwarded to traverse_from_anchors."""
        entity = _entity("COMP_001")
        traversal = _traversal(["CL001_c0"], [_triple()])
        chunk = _chunk()
        result = _gen_result("Query: test\nGraph context:", [Citation("CL001_c0", "Sample")])

        with (
            patch(f"{_MODULE}.resolve_entities", return_value=[entity]) as mock_resolve,
            patch(f"{_MODULE}.traverse_from_anchors", return_value=traversal) as mock_traverse,
            patch(f"{_MODULE}.retrieve_constrained", return_value=[chunk]),
            patch(f"{_MODULE}.generate_answer", return_value=result),
        ):
            out = run_graph_rag("Acme Corp contract", MagicMock(), MagicMock())

        mock_resolve.assert_called_once()
        _, traverse_args, _ = mock_traverse.mock_calls[0]
        assert "COMP_001" in traverse_args[0]
        assert isinstance(out, GenerationResult)
        assert len(out.text_citations) == 1


class TestCaseInsensitiveMatchFlows:
    def test_case_insensitive_entity_flows_through(self) -> None:
        """Entity resolved via case-insensitive lookup flows unchanged through pipeline."""
        entity = _entity("PERS_007", label="Person", name="Alice Nguyen", score=1.0)
        traversal = _traversal(["CL002_c0"], [_triple("7", "DIRECTOR_OF", "20")])
        chunk = _chunk("CL002_c0")
        result = _gen_result()

        with (
            patch(f"{_MODULE}.resolve_entities", return_value=[entity]),
            patch(f"{_MODULE}.traverse_from_anchors", return_value=traversal) as mock_traverse,
            patch(f"{_MODULE}.retrieve_constrained", return_value=[chunk]),
            patch(f"{_MODULE}.generate_answer", return_value=result),
        ):
            out = run_graph_rag("alice nguyen director", MagicMock(), MagicMock())

        _, traverse_args, _ = mock_traverse.mock_calls[0]
        assert traverse_args[0] == ["PERS_007"]
        assert isinstance(out, GenerationResult)


class TestPartialSubstringMatchFlows:
    def test_partial_match_flows_with_score_below_one(self) -> None:
        """Partial substring match (score < 1.0) still forwards node_id to traversal."""
        entity = _entity("COMP_002", name="Acme Corporation", score=4 / 16)
        traversal = _traversal([], [])
        result = _gen_result()

        with (
            patch(f"{_MODULE}.resolve_entities", return_value=[entity]),
            patch(f"{_MODULE}.traverse_from_anchors", return_value=traversal) as mock_traverse,
            patch(f"{_MODULE}.retrieve_constrained", return_value=[]),
            patch(f"{_MODULE}.generate_answer", return_value=result),
        ):
            out = run_graph_rag("Acme procurement", MagicMock(), MagicMock())

        _, traverse_args, _ = mock_traverse.mock_calls[0]
        assert "COMP_002" in traverse_args[0]
        assert isinstance(out, GenerationResult)


class TestZeroMatchesFallsBack:
    def test_empty_entity_list_produces_fallback_result(self) -> None:
        """Empty entity resolution: traverse and retrieve called with empty inputs."""
        traversal = _traversal([], [])
        fallback = _gen_result("No evidence retrieved for this query.")

        with (
            patch(f"{_MODULE}.resolve_entities", return_value=[]),
            patch(f"{_MODULE}.traverse_from_anchors", return_value=traversal) as mock_traverse,
            patch(f"{_MODULE}.retrieve_constrained", return_value=[]) as mock_retrieve,
            patch(f"{_MODULE}.generate_answer", return_value=fallback),
        ):
            out = run_graph_rag("what is the", MagicMock(), MagicMock())

        _, traverse_args, _ = mock_traverse.mock_calls[0]
        assert traverse_args[0] == []
        _, retrieve_args, _ = mock_retrieve.mock_calls[0]
        assert retrieve_args[1] == []  # allowed_chunk_ids is empty
        assert isinstance(out, GenerationResult)
        assert "No evidence" in out.answer


class TestMultipleCandidatesTopK:
    def test_top_k_entities_forwarded_to_resolve(self) -> None:
        """top_k_entities parameter is passed through to resolve_entities."""
        traversal = _traversal()
        result = _gen_result()

        with (
            patch(f"{_MODULE}.resolve_entities", return_value=[]) as mock_resolve,
            patch(f"{_MODULE}.traverse_from_anchors", return_value=traversal),
            patch(f"{_MODULE}.retrieve_constrained", return_value=[]),
            patch(f"{_MODULE}.generate_answer", return_value=result),
        ):
            run_graph_rag("NovaTech query", MagicMock(), MagicMock(), top_k_entities=3)

        _, _, resolve_kwargs = mock_resolve.mock_calls[0]
        assert resolve_kwargs.get("top_k") == 3


class TestSpecialCharactersInName:
    def test_special_char_node_id_flows_through(self) -> None:
        """Node ID with special-character content flows unchanged to traversal."""
        entity = _entity("C_BS", name="Bell & Sons Ltd", score=10 / 14)
        traversal = _traversal([], [])
        result = _gen_result()

        with (
            patch(f"{_MODULE}.resolve_entities", return_value=[entity]),
            patch(f"{_MODULE}.traverse_from_anchors", return_value=traversal) as mock_traverse,
            patch(f"{_MODULE}.retrieve_constrained", return_value=[]),
            patch(f"{_MODULE}.generate_answer", return_value=result),
        ):
            run_graph_rag('"Bell & Sons" contract', MagicMock(), MagicMock())

        _, traverse_args, _ = mock_traverse.mock_calls[0]
        assert traverse_args[0] == ["C_BS"]


# ---------------------------------------------------------------------------
# graph_traversal
# ---------------------------------------------------------------------------


class TestSingleHopTraversal:
    def test_single_hop_triples_reach_generate_answer(self) -> None:
        """1-hop traversal triples are forwarded to generate_answer."""
        entity = _entity("COMP_001")
        triple = _triple("10", "PARTY_TO", "20")
        traversal = _traversal(["CL001_c0"], [triple])
        chunk = _chunk("CL001_c0")
        result = _gen_result()

        with (
            patch(f"{_MODULE}.resolve_entities", return_value=[entity]),
            patch(f"{_MODULE}.traverse_from_anchors", return_value=traversal),
            patch(f"{_MODULE}.retrieve_constrained", return_value=[chunk]),
            patch(f"{_MODULE}.generate_answer", return_value=result) as mock_gen,
        ):
            run_graph_rag("test query", MagicMock(), MagicMock(), max_hops=1)

        _, gen_args, _ = mock_gen.mock_calls[0]
        triples_passed = gen_args[2]
        assert triple in triples_passed


class TestTwoHopTraversal:
    def test_max_hops_forwarded_to_traverse(self) -> None:
        """max_hops=2 is passed through to traverse_from_anchors."""
        entity = _entity("COMP_001")
        traversal = _traversal()
        result = _gen_result()

        with (
            patch(f"{_MODULE}.resolve_entities", return_value=[entity]),
            patch(f"{_MODULE}.traverse_from_anchors", return_value=traversal) as mock_traverse,
            patch(f"{_MODULE}.retrieve_constrained", return_value=[]),
            patch(f"{_MODULE}.generate_answer", return_value=result),
        ):
            run_graph_rag("test query", MagicMock(), MagicMock(), max_hops=2)

        _, _, traverse_kwargs = mock_traverse.mock_calls[0]
        assert traverse_kwargs.get("max_hops") == 2


class TestNoPathReturnsEmptyTriples:
    def test_empty_triples_forwarded_to_generate(self) -> None:
        """When traversal finds no paths, empty triples list reaches generate_answer."""
        entity = _entity("COMP_099")
        traversal = _traversal([], [])
        result = _gen_result()

        with (
            patch(f"{_MODULE}.resolve_entities", return_value=[entity]),
            patch(f"{_MODULE}.traverse_from_anchors", return_value=traversal),
            patch(f"{_MODULE}.retrieve_constrained", return_value=[]),
            patch(f"{_MODULE}.generate_answer", return_value=result) as mock_gen,
        ):
            run_graph_rag("disconnected query", MagicMock(), MagicMock())

        _, gen_args, _ = mock_gen.mock_calls[0]
        assert gen_args[2] == []


class TestIsolatedNodeTraversal:
    def test_isolated_node_no_chunk_ids_from_traversal(self) -> None:
        """Isolated node resolves but traversal returns empty chunk_ids and triples."""
        entity = _entity("ISO_001")
        traversal = _traversal([], [])
        result = _gen_result()

        with (
            patch(f"{_MODULE}.resolve_entities", return_value=[entity]),
            patch(f"{_MODULE}.traverse_from_anchors", return_value=traversal),
            patch(f"{_MODULE}.retrieve_constrained", return_value=[]) as mock_retrieve,
            patch(f"{_MODULE}.generate_answer", return_value=result),
        ):
            run_graph_rag("isolated entity query", MagicMock(), MagicMock())

        _, retrieve_args, _ = mock_retrieve.mock_calls[0]
        assert retrieve_args[1] == []  # empty chunk_ids from traversal


# ---------------------------------------------------------------------------
# constrained_vector_retrieval
# ---------------------------------------------------------------------------


class TestChunksFilteredAdjacent:
    def test_traversal_chunk_ids_passed_to_retrieve(self) -> None:
        """chunk_ids from traversal are forwarded to retrieve_constrained."""
        entity = _entity("COMP_001")
        traversal = _traversal(["CL001_c0", "CL001_c1"], [_triple()])
        chunk = _chunk("CL001_c0")
        result = _gen_result()

        with (
            patch(f"{_MODULE}.resolve_entities", return_value=[entity]),
            patch(f"{_MODULE}.traverse_from_anchors", return_value=traversal),
            patch(f"{_MODULE}.retrieve_constrained", return_value=[chunk]) as mock_retrieve,
            patch(f"{_MODULE}.generate_answer", return_value=result),
        ):
            run_graph_rag("filtered query", MagicMock(), MagicMock())

        _, retrieve_args, _ = mock_retrieve.mock_calls[0]
        assert retrieve_args[1] == ["CL001_c0", "CL001_c1"]


class TestEmptyGraphEmptyChunks:
    def test_empty_traversal_chunk_ids_reach_retrieve(self) -> None:
        """Empty traversal chunk_ids → retrieve_constrained called with empty list."""
        entity = _entity("COMP_001")
        traversal = _traversal([], [_triple()])
        result = _gen_result()

        with (
            patch(f"{_MODULE}.resolve_entities", return_value=[entity]),
            patch(f"{_MODULE}.traverse_from_anchors", return_value=traversal),
            patch(f"{_MODULE}.retrieve_constrained", return_value=[]) as mock_retrieve,
            patch(f"{_MODULE}.generate_answer", return_value=result),
        ):
            run_graph_rag("empty chunk query", MagicMock(), MagicMock())

        _, retrieve_args, _ = mock_retrieve.mock_calls[0]
        assert retrieve_args[1] == []


class TestRankingPreservedInResult:
    def test_ranked_chunks_flow_to_generation(self) -> None:
        """Chunks returned by retrieve_constrained are passed in order to generate_answer."""
        entity = _entity("COMP_001")
        traversal = _traversal(["c0", "c1", "c2"])
        chunks = [
            _chunk("c0", score=0.95),
            _chunk("c1", score=0.80),
            _chunk("c2", score=0.60),
        ]
        result = _gen_result()

        with (
            patch(f"{_MODULE}.resolve_entities", return_value=[entity]),
            patch(f"{_MODULE}.traverse_from_anchors", return_value=traversal),
            patch(f"{_MODULE}.retrieve_constrained", return_value=chunks),
            patch(f"{_MODULE}.generate_answer", return_value=result) as mock_gen,
        ):
            run_graph_rag("ranked query", MagicMock(), MagicMock())

        _, gen_args, _ = mock_gen.mock_calls[0]
        passed_chunks = gen_args[1]
        scores = [c.score for c in passed_chunks]
        assert scores == [0.95, 0.80, 0.60]


# ---------------------------------------------------------------------------
# integration_pipeline
# ---------------------------------------------------------------------------


class TestEndToEndReturnsResult:
    def test_full_pipeline_returns_populated_generation_result(self) -> None:
        """All stages produce results → GenerationResult has non-empty text_citations."""
        entity = _entity("COMP_001")
        triple = _triple("5", "PARTY_TO", "12")
        traversal = _traversal(["CT001_c0"], [triple])
        chunk = _chunk("CT001_c0", text="Contract clause text.")
        citation = Citation(chunk_id="CT001_c0", excerpt="Contract clause text.")
        answer_text = (
            "Query: test\nGraph context:\n  5 --[PARTY_TO]--> 12\n"
            "Supporting text:\n  [CT001_c0] Contract clause text."
        )
        result = GenerationResult(answer=answer_text, text_citations=[citation])

        with (
            patch(f"{_MODULE}.resolve_entities", return_value=[entity]),
            patch(f"{_MODULE}.traverse_from_anchors", return_value=traversal),
            patch(f"{_MODULE}.retrieve_constrained", return_value=[chunk]),
            patch(f"{_MODULE}.generate_answer", return_value=result),
        ):
            out = run_graph_rag("Acme Corp procurement", MagicMock(), MagicMock())

        assert isinstance(out, GenerationResult)
        assert len(out.text_citations) == 1
        assert out.text_citations[0].chunk_id == "CT001_c0"
        assert "PARTY_TO" in out.answer


class TestUnknownEntityFallsBack:
    def test_unknown_entity_returns_fallback_generation_result(self) -> None:
        """Unknown entity: resolve returns [] → pipeline returns fallback GenerationResult."""
        traversal = _traversal([], [])
        fallback_result = GenerationResult(
            answer="Query: Zxqrfoo unknown\nNo evidence retrieved for this query.",
            text_citations=[],
        )

        with (
            patch(f"{_MODULE}.resolve_entities", return_value=[]),
            patch(f"{_MODULE}.traverse_from_anchors", return_value=traversal),
            patch(f"{_MODULE}.retrieve_constrained", return_value=[]),
            patch(f"{_MODULE}.generate_answer", return_value=fallback_result),
        ):
            out = run_graph_rag("Zxqrfoo unknown entity", MagicMock(), MagicMock())

        assert isinstance(out, GenerationResult)
        assert out.text_citations == []
        assert "No evidence" in out.answer
