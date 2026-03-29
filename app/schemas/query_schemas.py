"""Shared Pydantic schemas for query endpoints (T030.a).

AnswerSchema, RetrievalDebug, GraphFact, and TextCitation are the canonical
definitions from graphrag_assistant.schemas — re-exported here so that app
code imports from one place and isinstance checks are identity-safe.

QueryRequest is HTTP-layer only and lives solely in this module.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from graphrag_assistant.schemas import (  # noqa: F401  (re-export)
    AnswerSchema,
    GraphFact,
    RetrievalDebug,
    TextCitation,
)


class QueryRequest(BaseModel):
    """Request body for both query endpoints."""

    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1)

    def __repr__(self) -> str:
        return super().__repr__()
