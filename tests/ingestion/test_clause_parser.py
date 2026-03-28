"""Tests for graphrag_assistant.ingestion.clause_parser (T020.a).

Unit tests (15)
---------------
Parser and chunker are the riskiest paths — 9 of 15 tests cover them.

Parser tests (9)
  1. test_markdown_headers_split_into_correct_clause_count
  2. test_markdown_header_clause_type_normalised
  3. test_numbered_section_splits_correctly
  4. test_numbered_section_clause_type_normalised
  5. test_no_header_produces_single_body_clause
  6. test_empty_string_produces_single_clause_no_raise
  7. test_whitespace_only_produces_single_clause_no_raise
  8. test_malformed_markdown_single_hash_no_split
  9. test_clause_ids_use_contract_id_prefix

Remaining unit tests (6)
 10. test_clause_order_is_sequential
 11. test_text_field_contains_body_content
 12. test_single_header_at_start_has_empty_preamble
 13. test_mixed_header_types_all_detected
 14. test_clause_type_strips_special_characters
 15. test_none_input_does_not_raise

Integration tests (2) — labelled with pytest.mark.integration
 16. test_integration_clause_count_matches_headers
 17. test_integration_single_clause_fallback_for_plain_text
"""
from __future__ import annotations

import pytest

from graphrag_assistant.ingestion.clause_parser import parse_clauses

# ---------------------------------------------------------------------------
# Fixtures / shared data
# ---------------------------------------------------------------------------

CONTRACT_ID = "CTR-001"

MULTI_HEADER_TEXT = """\
## Payment Terms

Payment is due within 30 days of invoice date.

## Confidentiality

All information shared under this agreement is confidential.

## Termination

Either party may terminate with 30 days written notice.
"""

NUMBERED_TEXT = """\
1. PAYMENT TERMS

Payment is due within 30 days of invoice.

2. CONFIDENTIALITY

Information shall remain confidential.
"""

NO_HEADER_TEXT = "This is a simple contract body with no headers at all."

EMPTY_TEXT = ""

WHITESPACE_TEXT = "   \n\t\n   "

MALFORMED_TEXT = "# Single hash does not match\n# another single hash line"

MIXED_TEXT = """\
## Definitions

Terms used in this agreement are defined below.

1. PAYMENT TERMS

Payment is due on receipt.
"""


# ---------------------------------------------------------------------------
# Parser tests (9) — highest risk
# ---------------------------------------------------------------------------


def test_markdown_headers_split_into_correct_clause_count():
    result = parse_clauses(CONTRACT_ID, MULTI_HEADER_TEXT)
    assert len(result) == 3


def test_markdown_header_clause_type_normalised():
    result = parse_clauses(CONTRACT_ID, MULTI_HEADER_TEXT)
    types = [c["clause_type"] for c in result]
    assert types == ["payment_terms", "confidentiality", "termination"]


def test_numbered_section_splits_correctly():
    result = parse_clauses(CONTRACT_ID, NUMBERED_TEXT)
    assert len(result) == 2


def test_numbered_section_clause_type_normalised():
    result = parse_clauses(CONTRACT_ID, NUMBERED_TEXT)
    types = [c["clause_type"] for c in result]
    assert types == ["payment_terms", "confidentiality"]


def test_no_header_produces_single_body_clause():
    result = parse_clauses(CONTRACT_ID, NO_HEADER_TEXT)
    assert len(result) == 1
    assert result[0]["clause_type"] == "body"
    assert result[0]["clause_order"] == 0


def test_empty_string_produces_single_clause_no_raise():
    result = parse_clauses(CONTRACT_ID, EMPTY_TEXT)
    assert len(result) == 1
    assert result[0]["clause_type"] == "body"


def test_whitespace_only_produces_single_clause_no_raise():
    result = parse_clauses(CONTRACT_ID, WHITESPACE_TEXT)
    assert len(result) == 1
    assert result[0]["clause_type"] == "body"


def test_malformed_markdown_single_hash_no_split():
    # Single # does not match ^##\s+ so falls back to one clause.
    result = parse_clauses(CONTRACT_ID, MALFORMED_TEXT)
    assert len(result) == 1
    assert result[0]["clause_type"] == "body"


def test_clause_ids_use_contract_id_prefix():
    result = parse_clauses(CONTRACT_ID, MULTI_HEADER_TEXT)
    for i, clause in enumerate(result):
        assert clause["clause_id"] == f"{CONTRACT_ID}_{i}"


# ---------------------------------------------------------------------------
# Remaining unit tests (6)
# ---------------------------------------------------------------------------


def test_clause_order_is_sequential():
    result = parse_clauses(CONTRACT_ID, MULTI_HEADER_TEXT)
    orders = [c["clause_order"] for c in result]
    assert orders == list(range(len(result)))


def test_text_field_contains_body_content():
    result = parse_clauses(CONTRACT_ID, MULTI_HEADER_TEXT)
    assert "30 days" in result[0]["text"]
    assert "confidential" in result[1]["text"]


def test_single_header_at_start_has_correct_text():
    text = "## Introduction\n\nThis contract governs the relationship."
    result = parse_clauses(CONTRACT_ID, text)
    assert len(result) == 1
    assert result[0]["clause_type"] == "introduction"
    assert "governs" in result[0]["text"]


def test_mixed_header_types_all_detected():
    result = parse_clauses(CONTRACT_ID, MIXED_TEXT)
    assert len(result) == 2
    types = [c["clause_type"] for c in result]
    assert "definitions" in types
    assert "payment_terms" in types


def test_clause_type_strips_special_characters():
    text = "## Liability & Indemnification\n\nText here."
    result = parse_clauses(CONTRACT_ID, text)
    assert result[0]["clause_type"] == "liability_indemnification"


def test_none_input_does_not_raise():
    # Passing None (non-string) must not raise.
    result = parse_clauses(CONTRACT_ID, None)  # type: ignore[arg-type]
    assert len(result) == 1
    assert result[0]["clause_type"] == "body"


# ---------------------------------------------------------------------------
# Integration tests (2)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_clause_count_matches_headers():
    """End-to-end: a realistic contract with 5 ## headers yields 5 clauses."""
    contract_text = """\
## Parties

Alpha Corp and Beta Ltd enter this agreement.

## Scope of Work

Alpha Corp shall provide software development services.

## Payment Terms

Fees are payable monthly in arrears within 14 days.

## Confidentiality

Both parties agree to keep all materials confidential.

## Termination

Either party may terminate with 60 days written notice.
"""
    result = parse_clauses("INT-001", contract_text)
    assert len(result) == 5
    types = [c["clause_type"] for c in result]
    assert "payment_terms" in types
    assert "confidentiality" in types


@pytest.mark.integration
def test_integration_single_clause_fallback_for_plain_text():
    """End-to-end: plain prose with no headers produces exactly one clause."""
    plain_text = (
        "This agreement is entered into between Alpha Corp and Beta Ltd. "
        "The terms and conditions set out herein govern the relationship between "
        "the parties. Both parties agree to act in good faith at all times."
    )
    result = parse_clauses("INT-002", plain_text)
    assert len(result) == 1
    assert result[0]["clause_type"] == "body"
    assert result[0]["clause_order"] == 0
    assert "Alpha Corp" in result[0]["text"]
