"""Unit tests for app.schemas.query (T030.a).

Test plan (13 tests):
  Request validation — unit (2):
    1. Empty string question raises ValidationError
    2. Missing question field raises ValidationError

  RetrievalDebug serialisation — plain-RAG context (3):
    3. All five keys present when all lists are empty
    4. graph_query=None serialises as a key (not omitted)
    5. Empty lists serialise as [] not as absent keys

  RetrievalDebug serialisation — graph-RAG context (3):
    6. Populated entity_matches serialises correctly
    7. Populated chunk_ids serialises correctly
    8. graph_query Cypher string round-trips

  graph_evidence format — AnswerSchema (3):
    9.  GraphFact entry has source_id key
    10. GraphFact entry has target_id key
    11. GraphFact entry has label key

  Happy-path AnswerSchema construction (2):
    12. mode='plain_rag' — all five top-level keys present
    13. mode='graph_rag'  — all five top-level keys present
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.query_schemas import QueryRequest
from graphrag_assistant.schemas import AnswerSchema, GraphFact, RetrievalDebug, TextCitation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPECTED_DEBUG_KEYS = {
    "graph_query", "entity_matches", "retrieved_node_ids", "chunk_ids", "timings"
}
_EXPECTED_ANSWER_KEYS = {"answer", "graph_evidence", "text_citations", "retrieval_debug", "mode"}


def _empty_debug() -> RetrievalDebug:
    # Canonical RetrievalDebug has no defaults — all five fields are required.
    return RetrievalDebug(
        graph_query=None,
        entity_matches=[],
        retrieved_node_ids=[],
        chunk_ids=[],
        timings={},
    )


def _populated_debug(cypher: str = "MATCH (n) RETURN n") -> RetrievalDebug:
    return RetrievalDebug(
        graph_query=cypher,
        entity_matches=["Acme Corp"],
        retrieved_node_ids=["company:1"],
        chunk_ids=["chunk:42"],
        timings={"graph_ms": 12.3, "embed_ms": 5.1},
    )


# ---------------------------------------------------------------------------
# 1-2  Request validation
# ---------------------------------------------------------------------------


def test_query_request_rejects_empty_string() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="")


def test_query_request_rejects_missing_question() -> None:
    with pytest.raises(ValidationError):
        QueryRequest()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 3-5  RetrievalDebug serialisation — plain-RAG (empty / None path)
# ---------------------------------------------------------------------------


def test_retrieval_debug_all_five_keys_present_when_empty() -> None:
    dumped = _empty_debug().model_dump()
    assert dumped.keys() == _EXPECTED_DEBUG_KEYS


def test_retrieval_debug_graph_query_none_serialises_as_key() -> None:
    dumped = _empty_debug().model_dump()
    assert "graph_query" in dumped
    assert dumped["graph_query"] is None


def test_retrieval_debug_empty_lists_present_not_absent() -> None:
    dumped = _empty_debug().model_dump()
    assert dumped["entity_matches"] == []
    assert dumped["retrieved_node_ids"] == []
    assert dumped["chunk_ids"] == []


# ---------------------------------------------------------------------------
# 6-8  RetrievalDebug serialisation — graph-RAG (populated path)
# ---------------------------------------------------------------------------


def test_retrieval_debug_entity_matches_round_trips() -> None:
    debug = _populated_debug()
    assert debug.model_dump()["entity_matches"] == ["Acme Corp"]


def test_retrieval_debug_chunk_ids_round_trips() -> None:
    debug = _populated_debug()
    assert debug.model_dump()["chunk_ids"] == ["chunk:42"]


def test_retrieval_debug_graph_query_string_round_trips() -> None:
    cypher = "MATCH (c:Company)-[:PARTY_TO]->(k:Contract) RETURN c, k"
    debug = _populated_debug(cypher)
    assert debug.model_dump()["graph_query"] == cypher


# ---------------------------------------------------------------------------
# 9-11  graph_evidence format
# ---------------------------------------------------------------------------


def _graph_fact() -> GraphFact:
    return GraphFact(source_id="company:1", target_id="contract:99", label="PARTY_TO")


def test_graph_evidence_entry_has_source_id() -> None:
    assert _graph_fact().model_dump()["source_id"] == "company:1"


def test_graph_evidence_entry_has_target_id() -> None:
    assert _graph_fact().model_dump()["target_id"] == "contract:99"


def test_graph_evidence_entry_has_label() -> None:
    assert _graph_fact().model_dump()["label"] == "PARTY_TO"


# ---------------------------------------------------------------------------
# 12-13  Happy-path AnswerSchema construction
# ---------------------------------------------------------------------------


def test_answer_schema_plain_rag_all_top_level_fields() -> None:
    schema = AnswerSchema(
        answer="Acme Corp is party to contract C-001.",
        graph_evidence=[],
        text_citations=[],
        retrieval_debug=_empty_debug(),
        mode="plain_rag",
    )
    dumped = schema.model_dump()
    assert dumped.keys() == _EXPECTED_ANSWER_KEYS
    assert dumped["mode"] == "plain_rag"


def test_retrieval_debug_class_is_from_graphrag_assistant_schemas() -> None:
    import graphrag_assistant.schemas as canonical

    assert RetrievalDebug is canonical.RetrievalDebug


def test_answer_schema_class_is_from_graphrag_assistant_schemas() -> None:
    import graphrag_assistant.schemas as canonical

    assert AnswerSchema is canonical.AnswerSchema


def test_answer_schema_graph_rag_all_top_level_fields() -> None:
    schema = AnswerSchema(
        answer="Acme Corp signed contract C-001 on 2024-01-15.",
        graph_evidence=[_graph_fact()],
        text_citations=[
            TextCitation(doc_id="doc:1", chunk_id="chunk:42", quote="signed on 2024-01-15")
        ],
        retrieval_debug=_populated_debug(),
        mode="graph_rag",
    )
    dumped = schema.model_dump()
    assert dumped.keys() == _EXPECTED_ANSWER_KEYS
    assert dumped["mode"] == "graph_rag"
    assert len(dumped["graph_evidence"]) == 1
    fact = dumped["graph_evidence"][0]
    assert {"source_id", "target_id", "label"} <= fact.keys()
