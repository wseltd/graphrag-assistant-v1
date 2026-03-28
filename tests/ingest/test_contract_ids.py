"""Tests for app/ingest/contract_ids.py.

8 tests as specified in T029.a:
  Unit — happy path (2), empty/malformed input (2)
  Integration-style pure-function checks — idempotency × 3, multi-file × 1
"""

import hashlib

from app.ingest.contract_ids import make_chunk_id, normalise_contract_id

# ---------------------------------------------------------------------------
# Unit: happy-path (2)
# ---------------------------------------------------------------------------


def test_normalise_strips_path_and_extension():
    result = normalise_contract_id("contracts/ACME_Corp-2024.md")
    assert result == "acme_corp_2024_md"


def test_make_chunk_id_is_sha256_hex():
    contract_id = "acme_corp_2024_md"
    chunk_index = 3
    expected = hashlib.sha256(f"{contract_id}:{chunk_index}".encode()).hexdigest()
    assert make_chunk_id(contract_id, chunk_index) == expected
    assert len(make_chunk_id(contract_id, chunk_index)) == 64


# ---------------------------------------------------------------------------
# Unit: empty / malformed input (2)
# ---------------------------------------------------------------------------


def test_normalise_empty_string_returns_empty():
    # An empty filename has no alphanumeric characters; stripping underscores
    # from an empty string is still empty.
    assert normalise_contract_id("") == ""


def test_normalise_only_special_chars_returns_empty():
    # A filename consisting entirely of non-alphanumeric characters should
    # yield an empty string after stripping.
    assert normalise_contract_id("---...---") == ""


# ---------------------------------------------------------------------------
# Integration-style: idempotency — same input always produces same output (3)
# ---------------------------------------------------------------------------


def test_normalise_idempotent_first_call():
    """First call: establishes the expected contract_id."""
    assert normalise_contract_id("order_001.txt") == "order_001_txt"


def test_normalise_idempotent_second_call():
    """Second call with identical input returns identical contract_id."""
    first = normalise_contract_id("order_001.txt")
    second = normalise_contract_id("order_001.txt")
    assert first == second


def test_make_chunk_id_idempotent_after_repeated_calls():
    """Chunk ID is stable across three calls — simulates three ingests."""
    cid = normalise_contract_id("order_001.txt")
    ids = [make_chunk_id(cid, 0) for _ in range(3)]
    assert ids[0] == ids[1] == ids[2]


# ---------------------------------------------------------------------------
# Integration-style: multi-file — five distinct filenames → five distinct IDs
# ---------------------------------------------------------------------------


def test_five_distinct_filenames_produce_distinct_contract_ids():
    filenames = [
        "contracts/alpha.md",
        "contracts/beta.md",
        "contracts/gamma.txt",
        "data/delta_2024.md",
        "EPSILON-Contract.md",
    ]
    ids = [normalise_contract_id(f) for f in filenames]
    assert len(set(ids)) == len(ids), f"Expected 5 distinct IDs, got: {ids}"
