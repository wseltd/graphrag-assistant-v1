from __future__ import annotations

from pydantic import BaseModel


class GraphFact(BaseModel):
    source_id: str
    target_id: str
    label: str

    def __repr__(self) -> str:
        return super().__repr__()


class TextCitation(BaseModel):
    doc_id: str
    chunk_id: str
    quote: str

    def __repr__(self) -> str:
        return super().__repr__()


class RetrievalDebug(BaseModel):
    graph_query: str | None
    entity_matches: list[str]
    retrieved_node_ids: list[str]
    chunk_ids: list[str]
    timings: dict[str, float]

    def __repr__(self) -> str:
        return super().__repr__()


class AnswerSchema(BaseModel):
    answer: str
    graph_evidence: list[GraphFact]
    text_citations: list[TextCitation]
    retrieval_debug: RetrievalDebug
    mode: str

    def __repr__(self) -> str:
        return super().__repr__()
