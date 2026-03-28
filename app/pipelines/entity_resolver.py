"""Entity resolver pipeline module (T025.a).

Resolves a natural-language query to a ranked list of Neo4j node IDs using
case-insensitive Cypher CONTAINS lookups across all searchable node labels.

Scoring
-------
  exact match (candidate.lower() == name.lower()) → score = 1.0
  partial match (CONTAINS but not equal)          → score = len(candidate) / len(name)

The overlap-ratio score ensures that a shorter candidate matching inside a
longer name ranks below a candidate that matches the full name.  Multiple
candidates extracted from the same query can resolve to the same node; in
that case only the highest-scoring EntityMatch per node_id is kept.

Zero-candidate queries
----------------------
If no entity names can be extracted from the query text (before any DB call),
OR if all candidates produce zero graph matches, a WARNING is logged with
the raw query string so operators can identify un-anchored queries.

Cypher safety
-------------
User input is NEVER interpolated into Cypher strings.  The $candidate
parameter is always passed as a Neo4j parameter dict.  Node labels come
from the module-level _LABEL_QUERIES dict (compile-time constants only).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop-word set for candidate tokenisation
# ---------------------------------------------------------------------------
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "has", "have", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "that", "this", "these", "those", "it",
        "its", "which", "who", "whom", "what", "where", "when", "how",
        "between", "about", "under", "over", "after", "before", "during",
        "as", "not", "no", "nor", "so", "yet",
    }
)

# ---------------------------------------------------------------------------
# Per-label Cypher queries.
# Compile-time constants — never modified at runtime, never interpolated.
# Returns node_id (domain key) and name so the scorer can compare strings.
# Contract uses contract_id as key and title as the searchable name field.
# ---------------------------------------------------------------------------
_LABEL_QUERIES: dict[str, str] = {
    "Company": (
        "MATCH (n:Company) "
        "WHERE toLower(n.name) CONTAINS toLower($candidate) "
        "RETURN n.id AS node_id, n.name AS name"
    ),
    "Person": (
        "MATCH (n:Person) "
        "WHERE toLower(n.name) CONTAINS toLower($candidate) "
        "RETURN n.id AS node_id, n.name AS name"
    ),
    "Product": (
        "MATCH (n:Product) "
        "WHERE toLower(n.name) CONTAINS toLower($candidate) "
        "RETURN n.id AS node_id, n.name AS name"
    ),
    "Contract": (
        "MATCH (n:Contract) "
        "WHERE toLower(n.title) CONTAINS toLower($candidate) "
        "RETURN n.contract_id AS node_id, n.title AS name"
    ),
}


# ---------------------------------------------------------------------------
# Public type
# ---------------------------------------------------------------------------


@dataclass
class EntityMatch:
    """A single entity resolved from a query token to a Neo4j graph node.

    Attributes:
        node_id: Domain-level unique identifier (e.g. Company.id,
                 Contract.contract_id).  Used as *allowed_ids* in the
                 constrained retriever.
        label:   Neo4j label of the matched node (e.g. ``"Company"``).
        name:    The name/title property value from the matched node.
        score:   1.0 for exact matches; overlap ratio (len(candidate) /
                 len(name)) for partial CONTAINS matches.  Always in (0, 1].
    """

    node_id: str
    label: str
    name: str
    score: float


# ---------------------------------------------------------------------------
# Candidate extraction (pure Python, no Neo4j)
# ---------------------------------------------------------------------------


def _extract_quoted(query: str) -> tuple[list[str], str]:
    """Remove quoted phrases from *query* and return (phrases, remainder)."""
    pattern = re.compile(r"'([^']+)'|\"([^\"]+)\"")
    phrases: list[str] = []
    remainder = query
    for match in pattern.finditer(query):
        phrase = (match.group(1) or match.group(2)).strip()
        if phrase:
            phrases.append(phrase)
        remainder = remainder.replace(match.group(0), " ", 1)
    return phrases, remainder


def _flush_run(run: list[str], out: list[str]) -> None:
    if not run:
        return
    # Single-token candidates must be ≥3 chars to suppress pronouns and
    # common abbreviations.  Multi-token runs are always included.
    if len(run) > 1 or len(run[0]) >= 3:
        out.append(" ".join(run))


def _extract_capitalised_runs(text: str) -> list[str]:
    """Collect runs of capitalised, non-stop-word tokens from *text*."""
    tokens = text.split()
    candidates: list[str] = []
    run: list[str] = []
    for tok in tokens:
        bare = tok.rstrip(".,;:!?")
        cleaned = re.sub(r"[^A-Za-z0-9]", "", bare)
        if not cleaned:
            _flush_run(run, candidates)
            run = []
            continue
        is_stop = bare.lower() in _STOP_WORDS
        is_cap = cleaned[0].isupper()
        if is_cap and not is_stop:
            run.append(bare)
        else:
            _flush_run(run, candidates)
            run = []
    _flush_run(run, candidates)
    return candidates


def extract_candidates(query: str) -> list[str]:
    """Return deduplicated candidate entity strings from *query*.

    Priority order:
    1. Quoted strings (verbatim).
    2. Capitalised runs from the remainder.

    Args:
        query: Raw user query string.

    Returns:
        Ordered, deduplicated list of candidate strings (case-insensitive
        dedup; first occurrence wins).
    """
    quoted, remainder = _extract_quoted(query)
    cap_runs = _extract_capitalised_runs(remainder)
    seen: set[str] = set()
    result: list[str] = []
    for candidate in quoted + cap_runs:
        key = candidate.lower()
        if key not in seen and candidate.strip():
            seen.add(key)
            result.append(candidate)
    return result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score(candidate: str, name: str) -> float:
    """Return 1.0 for exact match, overlap ratio for partial CONTAINS match.

    Overlap ratio = len(candidate) / len(name).  Always in (0, 1] because
    a CONTAINS match guarantees len(candidate) <= len(name).
    """
    if candidate.lower() == name.lower():
        return 1.0
    name_len = len(name)
    if name_len == 0:
        return 0.0
    return min(1.0, len(candidate) / name_len)


# ---------------------------------------------------------------------------
# Neo4j CONTAINS lookup
# ---------------------------------------------------------------------------


def _match_candidate(session: Any, candidate: str) -> list[EntityMatch]:
    """Query every node label for *candidate* using CONTAINS.

    Args:
        session:   Open Neo4j session.
        candidate: Entity string from Stage 1 extraction.

    Returns:
        All EntityMatch objects across all labels for this candidate.
    """
    matches: list[EntityMatch] = []
    for label, cypher in _LABEL_QUERIES.items():
        result = session.run(cypher, candidate=candidate)
        for record in result:
            node_id = record["node_id"]
            name = record["name"]
            if node_id is None or name is None:
                continue
            matches.append(
                EntityMatch(
                    node_id=str(node_id),
                    label=label,
                    name=str(name),
                    score=_score(candidate, str(name)),
                )
            )
    return matches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_entities(
    query: str,
    session: Any,
    top_k: int = 5,
) -> list[EntityMatch]:
    """Extract named entities from *query* and resolve them to Neo4j node IDs.

    Steps:
    1. extract_candidates — tokenise query into candidate noun phrases.
    2. For each candidate, query all node labels via CONTAINS (Stage 2).
    3. Deduplicate by node_id (keep highest score per node).
    4. Sort by score descending; return up to *top_k* results.

    Logs a WARNING with the raw query string in two cases:
    - No candidates could be extracted from the query text.
    - Candidates were extracted but no graph nodes matched any of them.

    Args:
        query:   Raw user query string.
        session: Open Neo4j session (caller manages lifecycle).
        top_k:   Maximum number of EntityMatch results to return. Default 5.

    Returns:
        List of EntityMatch objects sorted by score descending, length <= top_k.
        Empty list when no recognisable entity name is found.
    """
    candidates = extract_candidates(query)
    if not candidates:
        logger.warning(
            "entity_resolver: no candidates extracted — query=%r", query
        )
        return []

    raw_matches: list[EntityMatch] = []
    for candidate in candidates:
        raw_matches.extend(_match_candidate(session, candidate))

    if not raw_matches:
        logger.warning(
            "entity_resolver: zero graph matches — query=%r", query
        )
        return []

    # Deduplicate by node_id, keeping the highest score per node.
    best: dict[str, EntityMatch] = {}
    for match in raw_matches:
        existing = best.get(match.node_id)
        if existing is None or match.score > existing.score:
            best[match.node_id] = match

    ranked = sorted(best.values(), key=lambda m: m.score, reverse=True)
    return ranked[:top_k]
