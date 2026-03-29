"""Citation generator pipeline module (T025.d).

generate_answer formats a deterministic answer string from graph triples and
ranked text chunks — no LLM call, no external dependency.

Answer structure
----------------
1. "Query: <query>"
2. "Graph context:" followed by one line per Triple:
   "  <src> --[<rel>]--> <dst>"
3. "Supporting text:" followed by one line per RankedChunk:
   "  [<chunk_id>] <first sentence of chunk text>"

When both triples and chunks are empty a single fallback line is appended.

text_citations
--------------
One Citation per RankedChunk in input order.  quote is the first
_EXCERPT_MAX characters of chunk.text (the full text when shorter).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.pipelines.constrained_retrieval import RankedChunk
from app.pipelines.graph_traversal import Triple

logger = logging.getLogger(__name__)

_EXCERPT_MAX = 200
_FALLBACK_LINE = "No evidence retrieved for this query."


@dataclass
class Citation:
    """A text citation linking a retrieved chunk to the answer.

    Attributes:
        chunk_id: Matches RankedChunk.chunk_id.
        quote:    First _EXCERPT_MAX characters of the chunk text.
        doc_id:   Document identifier forwarded from RankedChunk.doc_id.
                  Defaults to "" when the chunk carries no doc_id.
    """

    chunk_id: str
    quote: str
    doc_id: str = ""


@dataclass
class GenerationResult:
    """Output of generate_answer.

    Attributes:
        answer:          Formatted answer string with inline citation markers.
        text_citations:  One Citation per chunk passed to generate_answer.
    """

    answer: str
    text_citations: list[Citation]


def generate_answer(
    query: str,
    chunks: list[RankedChunk],
    triples: list[Triple],
) -> GenerationResult:
    """Build a deterministic answer from graph triples and ranked chunks.

    No LLM call is made.  Output is fully deterministic for identical inputs.

    Args:
        query:   Raw user question.
        chunks:  Ranked chunks from constrained retrieval (or plain-RAG
                 fallback).  Processed in input order.
        triples: Relationship triples from graph traversal.  Processed in
                 input order.

    Returns:
        GenerationResult with a populated answer string and text_citations.
        text_citations contains one Citation per chunk; empty when chunks=[].
    """
    parts: list[str] = [f"Query: {query}"]

    if triples:
        parts.append("Graph context:")
        for triple in triples:
            parts.append(f"  {triple.src} --[{triple.rel}]--> {triple.dst}")

    if chunks:
        parts.append("Supporting text:")
        for chunk in chunks:
            first = _first_sentence(chunk.text)
            parts.append(f"  [{chunk.chunk_id}] {first}")
    elif not triples:
        parts.append(_FALLBACK_LINE)

    answer = "\n".join(parts)

    text_citations = [
        Citation(
            chunk_id=chunk.chunk_id,
            quote=chunk.text[:_EXCERPT_MAX],
            doc_id=chunk.doc_id,
        )
        for chunk in chunks
    ]

    logger.info(
        "generate_answer: triples=%d chunks=%d citations=%d",
        len(triples),
        len(chunks),
        len(text_citations),
    )

    return GenerationResult(answer=answer, text_citations=text_citations)


def _first_sentence(text: str) -> str:
    """Return the first sentence of *text*, or the full text if no period."""
    end = text.find(".")
    if end == -1:
        return text
    return text[: end + 1]
