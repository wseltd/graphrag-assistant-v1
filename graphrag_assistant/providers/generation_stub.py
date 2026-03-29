"""Template-based stub implementing GenerationProvider.

Assembles a structurally valid AnswerSchema from graph facts and retrieved
chunks without calling any external service or model.  Intended for use by
benchmarking infrastructure before a real LLM provider is wired in.

Mode detection: the caller encodes the retrieval mode as the first token
of the prompt in the form ``[mode:plain_rag]`` or ``[mode:graph_rag]``.
Any unrecognised prefix defaults to ``plain_rag``.
"""
from __future__ import annotations

import re

from graphrag_assistant.providers.base import GenerationProvider
from graphrag_assistant.schemas import (
    AnswerSchema,
    GraphFact,
    RetrievalDebug,
    TextCitation,
)

_MODE_PREFIX_RE = re.compile(r"^\[mode:([^\]]+)\]")
_QUOTE_MAX = 200
_FALLBACK_ANSWER = (
    "No graph evidence or text chunks were retrieved for this query."
)
# Sentinel string used for graph_query in retrieval debug when this stub runs.
# A Python list repr (str(graph_facts)) would leak internal structure into a
# public schema field; a named constant signals which strategy was used instead.
_GRAPH_QUERY_STUB = "GRAPH_TRAVERSAL"


class TemplateGenerationProvider(GenerationProvider):
    """Generates answers by template assembly — no LLM calls."""

    def __repr__(self) -> str:
        return "TemplateGenerationProvider()"

    def generate(
        self,
        prompt: str,
        graph_facts: list[dict],
        chunks: list[dict],
    ) -> AnswerSchema:
        """Assemble and return a fully valid AnswerSchema.

        Args:
            prompt:      User question, optionally prefixed with
                         ``[mode:plain_rag]`` or ``[mode:graph_rag]``.
            graph_facts: Graph triples returned by the retrieval stage,
                         each a dict with at minimum
                         ``{"source_id", "target_id", "label"}``.
            chunks:      Retrieved text chunks, each a dict with at
                         minimum ``{"doc_id", "chunk_id", "text"}``.

        Returns:
            A fully populated AnswerSchema.
        """
        mode = _extract_mode(prompt)
        evidence = _build_graph_evidence(graph_facts)
        citations = _build_text_citations(chunks)
        answer = _build_answer(graph_facts, chunks)
        debug = _build_retrieval_debug(graph_facts, chunks)
        return AnswerSchema(
            answer=answer,
            graph_evidence=evidence,
            text_citations=citations,
            retrieval_debug=debug,
            mode=mode,
        )


def _extract_mode(prompt: str) -> str:
    match = _MODE_PREFIX_RE.match(prompt)
    if match:
        return match.group(1)
    return "plain_rag"


def _build_graph_evidence(graph_facts: list[dict]) -> list[GraphFact]:
    return [
        GraphFact(
            source_id=fact["source_id"],
            target_id=fact["target_id"],
            label=fact["label"],
        )
        for fact in graph_facts
    ]


def _build_text_citations(chunks: list[dict]) -> list[TextCitation]:
    return [
        TextCitation(
            doc_id=chunk["doc_id"],
            chunk_id=chunk["chunk_id"],
            quote=chunk["text"][:_QUOTE_MAX],
        )
        for chunk in chunks
    ]


def _build_answer(graph_facts: list[dict], chunks: list[dict]) -> str:
    if not graph_facts and not chunks:
        return _FALLBACK_ANSWER

    parts: list[str] = []

    if graph_facts:
        parts.append("Graph evidence:")
        for fact in graph_facts:
            src = fact["source_id"]
            tgt = fact["target_id"]
            lbl = fact["label"]
            parts.append(f"  {src} --[{lbl}]--> {tgt}")

    if chunks:
        parts.append("Supporting text:")
        for chunk in chunks:
            first = _first_sentence(chunk["text"])
            parts.append(f"  [{chunk['chunk_id']}] {first}")

    return "\n".join(parts)


def _first_sentence(text: str) -> str:
    """Return the first sentence of *text*, or the full text if no period."""
    end = text.find(".")
    if end == -1:
        return text
    return text[: end + 1]


def _build_retrieval_debug(
    graph_facts: list[dict],
    chunks: list[dict],
) -> RetrievalDebug:
    node_ids: list[str] = []
    for fact in graph_facts:
        if fact["source_id"] not in node_ids:
            node_ids.append(fact["source_id"])
        if fact["target_id"] not in node_ids:
            node_ids.append(fact["target_id"])
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    return RetrievalDebug(
        graph_query=_GRAPH_QUERY_STUB,
        entity_matches=list(node_ids),
        retrieved_node_ids=list(node_ids),
        chunk_ids=chunk_ids,
        timings={},
    )
