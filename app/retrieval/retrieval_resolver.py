"""Entity resolver: extract named entities from a query and match to Neo4j node IDs.

Two-stage resolver:

Stage 1 — Tokenise the query string into candidate noun phrases using stdlib re
only (no spaCy, no NLTK).
  - Quoted strings are extracted verbatim as single candidates (highest priority).
  - The remainder is split on stop-words; runs of capitalised tokens are
    collected as candidates.

Stage 2 — For each candidate, run a parameterised Cypher MATCH with
case-insensitive CONTAINS against name/title properties on every node label
(Company, Person, Product, Contract).  ALL matches are returned; callers
decide which labels to keep.  Disambiguation — the same candidate string may
match nodes of different labels — is a first-class requirement, not an edge case.

Scoring: score = float(len(candidate)).  Longer candidates rank higher, so an
exact full-name match outranks a partial-name substring match.

Cypher safety: user input is NEVER interpolated into Cypher strings.  The
$candidate parameter is always passed as a Neo4j parameter dict.  Node labels
come from the module-level _LABEL_QUERY dict (compile-time constants only).
"""
from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from neo4j import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop-word set used to split the query into noun-phrase candidates.
# Only add words here; removing entries could merge unrelated candidates.
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
# Labels are compile-time constants — never derived from user input.
# $candidate is always a parameterised placeholder.
# ---------------------------------------------------------------------------
_LABELS: tuple[str, ...] = ("Company", "Person", "Product", "Contract")

_LABEL_QUERY: dict[str, str] = {
    "Company": (
        "MATCH (n:Company) "
        "WHERE toLower(n.name) CONTAINS toLower($candidate) "
        "RETURN n.id AS node_id"
    ),
    "Person": (
        "MATCH (n:Person) "
        "WHERE toLower(n.name) CONTAINS toLower($candidate) "
        "RETURN n.id AS node_id"
    ),
    "Product": (
        "MATCH (n:Product) "
        "WHERE toLower(n.name) CONTAINS toLower($candidate) "
        "RETURN n.id AS node_id"
    ),
    "Contract": (
        "MATCH (n:Contract) "
        "WHERE toLower(n.title) CONTAINS toLower($candidate) "
        "RETURN n.contract_id AS node_id"
    ),
}


# ---------------------------------------------------------------------------
# Public type
# ---------------------------------------------------------------------------


class EntityMatch(BaseModel):
    """A single entity resolved from a candidate string to a graph node.

    Attributes:
        entity_string: The raw candidate text extracted from the query.
        node_id:       The graph node's unique identifier property.
        node_label:    The Neo4j label of the matched node.
        score:         Match score — currently float(len(entity_string)) so
                       longer candidates rank above shorter substring matches.
    """

    entity_string: str
    node_id: str
    node_label: str
    score: float

    def __repr__(self) -> str:
        return (
            f"EntityMatch("
            f"entity_string={self.entity_string!r}, "
            f"node_id={self.node_id!r}, "
            f"node_label={self.node_label!r}, "
            f"score={self.score!r})"
        )


# ---------------------------------------------------------------------------
# Stage 1 — candidate extraction (no Neo4j)
# ---------------------------------------------------------------------------


def _extract_quoted(query: str) -> tuple[list[str], str]:
    """Remove quoted phrases from *query* and return (phrases, remainder).

    Handles both single-quoted ('…') and double-quoted ("…") strings.
    The matched substrings are replaced with a space in the remainder so
    word-boundary detection in the capitalised-run pass is unaffected.
    """
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
    """Append the accumulated *run* to *out* if it meets the length threshold."""
    if not run:
        return
    # Single tokens must be ≥3 chars to suppress pronouns ("I"), articles
    # ("A"), and common abbreviations.  Multi-token runs are always included.
    if len(run) > 1 or len(run[0]) >= 3:
        out.append(" ".join(run))


def _extract_capitalised_runs(text: str) -> list[str]:
    """Collect runs of capitalised, non-stop tokens from *text*.

    A token is considered capitalised when its leading alphabetic character
    is uppercase (after stripping trailing punctuation).  Stop-word tokens
    break the current run.
    """
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
    """Return a deduplicated list of candidate entity strings from *query*.

    Extraction order:
    1. Quoted strings — kept verbatim and added first.
    2. Capitalised runs from the remainder of the query.

    Case-insensitive deduplication preserves first occurrence.
    Empty strings are never included.

    Args:
        query: The raw user query string.

    Returns:
        Ordered list of candidate strings, longest phrases tend to appear
        before their component substrings if quoted.
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
# Stage 2 — Neo4j CONTAINS lookup
# ---------------------------------------------------------------------------


def _match_candidate(session: Session, candidate: str) -> list[EntityMatch]:
    """Query every node label for *candidate* using CONTAINS.

    All matching nodes across all labels are returned.  The caller is
    responsible for disambiguation; this function never silently drops
    matches that happen to share a candidate string with another label.

    User input (*candidate*) is passed exclusively via the $candidate Neo4j
    parameter — it is never interpolated into the Cypher string.

    Args:
        session:   Open Neo4j session.
        candidate: The entity string extracted in Stage 1.

    Returns:
        List of EntityMatch objects (may be empty).
    """
    matches: list[EntityMatch] = []
    score = float(len(candidate))
    for label in _LABELS:
        cypher = _LABEL_QUERY[label]
        result = session.run(cypher, candidate=candidate)
        for record in result:
            node_id = record["node_id"]
            if node_id is None:
                continue
            matches.append(
                EntityMatch(
                    entity_string=candidate,
                    node_id=str(node_id),
                    node_label=label,
                    score=score,
                )
            )
    return matches


def resolve_entities(query: str, session: Session) -> list[EntityMatch]:
    """Extract entities from *query* and resolve them to Neo4j node IDs.

    Steps:
    1. extract_candidates(query) — Stage 1 tokenisation.
    2. For each candidate, query all node labels with CONTAINS — Stage 2.
    3. Merge all matches, sort by score descending, log timing, return.

    Returns all matches across all labels.  The same entity string may appear
    multiple times with different node_labels — this is correct and expected.

    Args:
        query:   The raw user query string.
        session: An open Neo4j session (caller manages lifecycle).

    Returns:
        List of EntityMatch objects sorted by score descending.
        Empty list when no recognisable entity is present.
    """
    start = time.monotonic()
    candidates = extract_candidates(query)
    all_matches: list[EntityMatch] = []
    for candidate in candidates:
        all_matches.extend(_match_candidate(session, candidate))
    elapsed_ms = (time.monotonic() - start) * 1000.0
    logger.info(
        "entity_resolver query=%r candidates=%d matches=%d elapsed_ms=%.1f",
        query,
        len(candidates),
        len(all_matches),
        elapsed_ms,
    )
    all_matches.sort(key=lambda m: m.score, reverse=True)
    return all_matches
