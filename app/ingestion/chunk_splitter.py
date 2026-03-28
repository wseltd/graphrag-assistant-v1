"""Chunk splitter for contract clauses (T020.b).

Public API
----------
split_clause_into_chunks(clause, party_company_ids, chunk_size=512, overlap=64)
chunk_clauses(clauses, party_company_ids, chunk_size=512, overlap=64)

Design
------
* Tokenisation is whitespace-based (str.split()); no external tokeniser required.
* Overlapping windows: stride = max(1, chunk_size - overlap).  Each window starts
  at i=0, i=stride, i=2*stride, ...  Iteration stops when i >= len(tokens).
* An empty or whitespace-only text produces exactly one chunk whose text is "".
* Each chunk dict carries: chunk_id, text, clause_id, clause_order, company_ids.
  company_ids is a shallow copy of party_company_ids so callers cannot mutate it.
"""
from __future__ import annotations


def split_clause_into_chunks(
    clause: dict,
    party_company_ids: list[str],
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[dict]:
    """Split a clause dict into overlapping token-window chunks.

    Args:
        clause:            Dict with at least 'clause_id', 'clause_order', 'text'.
        party_company_ids: Company IDs copied onto every produced chunk.
        chunk_size:        Maximum number of whitespace tokens per chunk.
        overlap:           Number of tokens shared between adjacent chunks.

    Returns:
        Non-empty list of chunk dicts, each with:
            chunk_id (str), text (str), clause_id (str),
            clause_order (int), company_ids (list[str]).
    """
    clause_id = clause["clause_id"]
    clause_order = clause["clause_order"]
    tokens = (clause.get("text") or "").split()

    if not tokens:
        return [
            {
                "chunk_id": f"{clause_id}_chunk_0",
                "text": "",
                "clause_id": clause_id,
                "clause_order": clause_order,
                "company_ids": list(party_company_ids),
            }
        ]

    stride = max(1, chunk_size - overlap)
    chunks: list[dict] = []
    i = 0
    while i < len(tokens):
        window = tokens[i : i + chunk_size]
        chunks.append(
            {
                "chunk_id": f"{clause_id}_chunk_{len(chunks)}",
                "text": " ".join(window),
                "clause_id": clause_id,
                "clause_order": clause_order,
                "company_ids": list(party_company_ids),
            }
        )
        i += stride

    return chunks


def chunk_clauses(
    clauses: list[dict],
    party_company_ids: list[str],
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[dict]:
    """Apply split_clause_into_chunks to every clause in a list.

    Args:
        clauses:           List of clause dicts (each as produced by parse_clauses).
        party_company_ids: Company IDs forwarded to split_clause_into_chunks.
        chunk_size:        Forwarded to split_clause_into_chunks.
        overlap:           Forwarded to split_clause_into_chunks.

    Returns:
        Flat list of all chunk dicts produced from all clauses, in order.
    """
    result: list[dict] = []
    for clause in clauses:
        result.extend(split_clause_into_chunks(clause, party_company_ids, chunk_size, overlap))
    return result
