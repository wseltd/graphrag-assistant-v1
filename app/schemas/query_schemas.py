"""Shared Pydantic schemas for query endpoints (T030.a).

All fields carry explicit defaults so that partial debug is impossible at
the model layer — every serialised response always contains the full key set.

graph_query is the one nullable field (str | None = None) because a plain-RAG
path genuinely has no Cypher query; the default ensures the key is always
present in the serialised output even when the value is None.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from graphrag_assistant.schemas import GraphFact, TextCitation


class QueryRequest(BaseModel):
    """Request body for both query endpoints."""

    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1)

    def __repr__(self) -> str:
        return super().__repr__()


class RetrievalDebug(BaseModel):
    """Debug envelope attached to every query response.

    Every field has an explicit default so callers never receive a response
    with missing debug keys, even when retrieval returns nothing.
    """

    graph_query: str | None = None
    entity_matches: list = Field(default_factory=list)
    retrieved_node_ids: list = Field(default_factory=list)
    chunk_ids: list = Field(default_factory=list)
    timings: dict = Field(default_factory=dict)

    def __repr__(self) -> str:
        return super().__repr__()


class AnswerSchema(BaseModel):
    """Full response schema returned by both query endpoints."""

    answer: str = ""
    graph_evidence: list[GraphFact] = Field(default_factory=list)
    text_citations: list[TextCitation] = Field(default_factory=list)
    retrieval_debug: RetrievalDebug = Field(default_factory=RetrievalDebug)
    mode: str = ""

    def __repr__(self) -> str:
        return super().__repr__()
