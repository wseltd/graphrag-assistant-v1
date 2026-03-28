"""Clause parser for contract text (T020.a).

Public API
----------
parse_clauses(contract_id, raw_text) -> list[dict]

Design
------
* Clause boundaries are detected by two regex patterns:
    - Markdown heading:   ``^## <title>``
    - Numbered section:   ``^\\d+\\.  [A-Z]<rest>``
* Each detected header starts a new clause; text before the first header
  (if any) is merged into the first clause.
* If no boundaries are found the entire text is returned as a single clause
  with clause_type='body' and clause_order=0.
* Header labels are normalised to clause_type by lowercasing, stripping
  leading digits/punctuation, and replacing runs of non-alphanumeric chars
  with underscores, then stripping leading/trailing underscores.
* The function never raises — empty strings, whitespace-only strings, and
  malformed markdown all produce the single-clause fallback.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Boundary patterns
# ---------------------------------------------------------------------------

# Matches "## Some Header" — group(1) captures the title text.
_MARKDOWN_HEADING = re.compile(r"^##\s+(.+)", re.MULTILINE)

# Matches "1. Title" or "12. PAYMENT TERMS" — group(1) captures title text.
_NUMBERED_SECTION = re.compile(r"^\d+\.\s+([A-Z].+)", re.MULTILINE)

# Combined: a line is a boundary if it matches either pattern.
_BOUNDARY = re.compile(
    r"^(?:##\s+(.+)|\d+\.\s+([A-Z].+))",
    re.MULTILINE,
)

# Characters to collapse into underscores for clause_type normalisation.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _header_to_type(header: str) -> str:
    """Normalise a raw header string to a snake_case clause_type token.

    Examples:
        "## Payment Terms"  -> "payment_terms"
        "1. CONFIDENTIALITY" -> "confidentiality"
        "##"                -> "body"
    """
    # Strip leading markdown hashes and digit+dot prefixes.
    cleaned = re.sub(r"^#+\s*|\d+\.\s*", "", header).strip().lower()
    normalised = _NON_ALNUM.sub("_", cleaned).strip("_")
    return normalised or "body"


def _split_on_boundaries(raw_text: str) -> list[tuple[str, str]]:
    """Return a list of (header, body_text) pairs split at boundary lines.

    If no boundary is found, returns an empty list (caller handles fallback).
    """
    matches = list(_BOUNDARY.finditer(raw_text))
    if not matches:
        return []

    segments: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        header = (match.group(1) or match.group(2) or "").strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end].strip()
        segments.append((header, body))

    return segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_clauses(contract_id: str, raw_text: str) -> list[dict]:
    """Split *raw_text* into clauses and return structured dicts.

    Each returned dict has:
        clause_id    str   f"{contract_id}_{index}"
        clause_type  str   inferred from header label (snake_case)
        clause_order int   0-based position in document
        text         str   clause body text (may be empty string)

    If no boundary headers are detected, a single clause with
    clause_type='body' and clause_order=0 is returned.

    This function never raises — any exception from regex or string
    operations is caught and causes a single-clause fallback.

    Args:
        contract_id: Identifier prefix used to form clause_id values.
        raw_text:    Raw contract text (may be empty or malformed).

    Returns:
        Non-empty list of clause dicts (at least one element).
    """
    try:
        return _parse(contract_id, raw_text)
    except Exception:  # noqa: BLE001
        return [
            {
                "clause_id": f"{contract_id}_0",
                "clause_type": "body",
                "clause_order": 0,
                "text": raw_text if isinstance(raw_text, str) else "",
            }
        ]


def _parse(contract_id: str, raw_text: str) -> list[dict]:
    """Internal implementation — may propagate exceptions (wrapped by caller)."""
    if not isinstance(raw_text, str):
        raw_text = ""

    text = raw_text.strip()

    segments = _split_on_boundaries(text)
    if not segments:
        return [
            {
                "clause_id": f"{contract_id}_0",
                "clause_type": "body",
                "clause_order": 0,
                "text": text,
            }
        ]

    return [
        {
            "clause_id": f"{contract_id}_{i}",
            "clause_type": _header_to_type(header),
            "clause_order": i,
            "text": body,
        }
        for i, (header, body) in enumerate(segments)
    ]
