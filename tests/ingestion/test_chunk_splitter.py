"""Tests for app.ingestion.chunk_splitter (T020.b).

Unit tests (15)
---------------
Chunker core (9) — highest risk: boundary and overlap logic
  1. test_text_shorter_than_chunk_size_produces_one_chunk
  2. test_text_exactly_one_token_produces_one_chunk
  3. test_empty_text_produces_one_chunk_with_empty_string
  4. test_chunk_count_for_text_longer_than_chunk_size
  5. test_overlap_tokens_appear_in_adjacent_chunks
  6. test_zero_overlap_no_shared_tokens
  7. test_chunk_id_format
  8. test_overlap_equals_chunk_size_minus_one_uses_stride_one
  9. test_last_chunk_may_be_shorter_than_chunk_size

Other unit tests (6)
 10. test_company_ids_copied_to_each_chunk
 11. test_clause_id_preserved_in_all_chunks
 12. test_clause_order_preserved_in_all_chunks
 13. test_company_ids_mutation_does_not_affect_chunk
 14. test_chunk_clauses_maps_over_multiple_clauses
 15. test_chunk_clauses_empty_clauses_list_returns_empty

Integration tests (2) — labelled with pytest.mark.integration
 16. test_integration_realistic_clause_produces_correct_chunk_count
 17. test_integration_chunk_clauses_all_chunks_carry_party_ids
"""
from __future__ import annotations

import pytest

from app.ingestion.chunk_splitter import chunk_clauses, split_clause_into_chunks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clause(
    clause_id: str = "CTR-001_0",
    clause_order: int = 0,
    text: str = "",
) -> dict:
    return {"clause_id": clause_id, "clause_order": clause_order, "text": text}


def _words(n: int) -> str:
    """Return a string of n distinct whitespace-separated tokens."""
    return " ".join(f"word{i}" for i in range(n))


# ---------------------------------------------------------------------------
# Chunker core (9) — highest risk
# ---------------------------------------------------------------------------


def test_text_shorter_than_chunk_size_produces_one_chunk():
    # 5 tokens, chunk_size=10, overlap=2, stride=8 → 5 < 8 → 1 chunk
    clause = _clause(text=_words(5))
    chunks = split_clause_into_chunks(clause, [], chunk_size=10, overlap=2)
    assert len(chunks) == 1


def test_text_exactly_one_token_produces_one_chunk():
    clause = _clause(text="singleton")
    chunks = split_clause_into_chunks(clause, [], chunk_size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "singleton"


def test_empty_text_produces_one_chunk_with_empty_string():
    clause = _clause(text="")
    chunks = split_clause_into_chunks(clause, [], chunk_size=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0]["text"] == ""


def test_chunk_count_for_text_longer_than_chunk_size():
    # 10 tokens, chunk_size=4, overlap=1, stride=3
    # i=0 (0..3), i=3 (3..6), i=6 (6..9), i=9 (9..12→9) → 4 chunks
    clause = _clause(text=_words(10))
    chunks = split_clause_into_chunks(clause, [], chunk_size=4, overlap=1)
    assert len(chunks) == 4


def test_overlap_tokens_appear_in_adjacent_chunks():
    # "w0 w1 w2 w3", chunk_size=4, overlap=2, stride=2
    # i=0: "w0 w1 w2 w3" → chunk_0
    # i=2: "w2 w3"        → chunk_1  (w2, w3 shared)
    # i=4: 4 >= 4 → stop
    tokens = ["w0", "w1", "w2", "w3"]
    clause = _clause(text=" ".join(tokens))
    chunks = split_clause_into_chunks(clause, [], chunk_size=4, overlap=2)
    assert len(chunks) == 2
    c0_tokens = chunks[0]["text"].split()
    c1_tokens = chunks[1]["text"].split()
    # Last overlap tokens of chunk_0 equal first overlap tokens of chunk_1
    assert c0_tokens[-2:] == c1_tokens[:2]


def test_zero_overlap_no_shared_tokens():
    # 6 tokens, chunk_size=3, overlap=0, stride=3 → 2 non-overlapping chunks
    clause = _clause(text=_words(6))
    chunks = split_clause_into_chunks(clause, [], chunk_size=3, overlap=0)
    assert len(chunks) == 2
    set0 = set(chunks[0]["text"].split())
    set1 = set(chunks[1]["text"].split())
    assert set0.isdisjoint(set1)


def test_chunk_id_format():
    clause = _clause(clause_id="CTR-042_3")
    chunks = split_clause_into_chunks(clause, [], chunk_size=3, overlap=0)
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_id"] == f"CTR-042_3_chunk_{i}"


def test_overlap_equals_chunk_size_minus_one_uses_stride_one():
    # overlap = chunk_size - 1 → stride = 1 (minimum)
    # 3 tokens, chunk_size=3, overlap=2, stride=1
    # i=0 (w0 w1 w2), i=1 (w1 w2 w3 → w1 w2 truncated), i=2 (w2), stop
    clause = _clause(text="w0 w1 w2")
    chunks = split_clause_into_chunks(clause, [], chunk_size=3, overlap=2)
    assert len(chunks) == 3
    assert chunks[0]["text"] == "w0 w1 w2"
    assert chunks[1]["text"] == "w1 w2"
    assert chunks[2]["text"] == "w2"


def test_last_chunk_may_be_shorter_than_chunk_size():
    # 7 tokens, chunk_size=4, overlap=0, stride=4
    # chunk_0: 4 tokens, chunk_1: 3 tokens
    clause = _clause(text=_words(7))
    chunks = split_clause_into_chunks(clause, [], chunk_size=4, overlap=0)
    assert len(chunks) == 2
    assert len(chunks[1]["text"].split()) == 3


# ---------------------------------------------------------------------------
# Other unit tests (6)
# ---------------------------------------------------------------------------


def test_company_ids_copied_to_each_chunk():
    company_ids = ["CO-001", "CO-002"]
    clause = _clause(text=_words(10))
    chunks = split_clause_into_chunks(clause, company_ids, chunk_size=4, overlap=1)
    for chunk in chunks:
        assert chunk["company_ids"] == ["CO-001", "CO-002"]


def test_clause_id_preserved_in_all_chunks():
    clause = _clause(clause_id="CTR-999_7", text=_words(10))
    chunks = split_clause_into_chunks(clause, [], chunk_size=4, overlap=1)
    for chunk in chunks:
        assert chunk["clause_id"] == "CTR-999_7"


def test_clause_order_preserved_in_all_chunks():
    clause = _clause(clause_order=5, text=_words(10))
    chunks = split_clause_into_chunks(clause, [], chunk_size=4, overlap=1)
    for chunk in chunks:
        assert chunk["clause_order"] == 5


def test_company_ids_mutation_does_not_affect_chunk():
    company_ids = ["CO-001"]
    clause = _clause(text="hello world")
    chunks = split_clause_into_chunks(clause, company_ids, chunk_size=512, overlap=64)
    company_ids.append("CO-999")  # mutate original list after call
    assert chunks[0]["company_ids"] == ["CO-001"]


def test_chunk_clauses_maps_over_multiple_clauses():
    clauses = [
        _clause(clause_id="CTR-001_0", clause_order=0, text=_words(6)),
        _clause(clause_id="CTR-001_1", clause_order=1, text=_words(9)),
    ]
    # chunk_size=3, overlap=0, stride=3
    # clause 0: 6 tokens → 2 chunks; clause 1: 9 tokens → 3 chunks
    chunks = chunk_clauses(clauses, ["CO-001"], chunk_size=3, overlap=0)
    assert len(chunks) == 5
    assert chunks[0]["clause_id"] == "CTR-001_0"
    assert chunks[2]["clause_id"] == "CTR-001_1"


def test_chunk_clauses_empty_clauses_list_returns_empty():
    chunks = chunk_clauses([], ["CO-001"])
    assert chunks == []


# ---------------------------------------------------------------------------
# Integration tests (2)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_realistic_clause_produces_correct_chunk_count():
    """A realistic payment-terms clause (>512 tokens when duplicated) produces
    the expected number of overlapping chunks at default parameters."""
    # Build a clause text that is clearly longer than 512 tokens.
    base = (
        "Payment is due within thirty days of the invoice date. "
        "Late payments shall incur interest at the rate of two percent per month. "
        "All amounts are stated in United States dollars and are exclusive of tax. "
    )
    # Repeat until we have ~540 tokens (36 words × 15 = 540).
    long_text = (base * 15).strip()
    tokens = long_text.split()
    assert len(tokens) > 512, "fixture must exceed chunk_size"

    clause = _clause(clause_id="INT-001_0", clause_order=0, text=long_text)
    chunks = split_clause_into_chunks(clause, ["CO-ALPHA", "CO-BETA"], chunk_size=512, overlap=64)

    # With stride=448 and len>512, we expect at least 2 chunks.
    assert len(chunks) >= 2

    # Every chunk must reference the correct clause_id and both company_ids.
    for chunk in chunks:
        assert chunk["clause_id"] == "INT-001_0"
        assert chunk["company_ids"] == ["CO-ALPHA", "CO-BETA"]

    # Verify overlap: last 64 tokens of chunk_0 must equal first 64 tokens of chunk_1.
    c0_tokens = chunks[0]["text"].split()
    c1_tokens = chunks[1]["text"].split()
    assert c0_tokens[-64:] == c1_tokens[:64]


@pytest.mark.integration
def test_integration_chunk_clauses_all_chunks_carry_party_ids():
    """chunk_clauses over a multi-clause contract assigns party_ids to every chunk."""
    party_ids = ["ALPHA-CORP", "BETA-LTD"]
    clauses = [
        {
            "clause_id": f"INT-002_{i}",
            "clause_order": i,
            "text": f"Clause {i} text with enough words to be meaningful " * 3,
        }
        for i in range(5)
    ]
    chunks = chunk_clauses(clauses, party_ids, chunk_size=20, overlap=4)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk["company_ids"] == party_ids
        # chunk_id must encode both the clause_id and a numeric suffix
        assert "_chunk_" in chunk["chunk_id"]
        assert chunk["clause_id"].startswith("INT-002_")
