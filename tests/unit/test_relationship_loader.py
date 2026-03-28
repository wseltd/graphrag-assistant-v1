"""Unit tests for graphrag_assistant.loaders.relationship_loader (T016).

9 tests:
  Idempotency — MERGE returns 0 created on second call (mocked driver):
    1. test_director_of_zero_created_on_second_call
    2. test_registered_at_zero_created_on_second_call
    3. test_supplies_zero_created_on_second_call

  DataIntegrityError — raised when MATCH returns no matching nodes:
    4. test_data_integrity_error_missing_source_person
    5. test_data_integrity_error_missing_target_company
    6. test_data_integrity_error_both_endpoints_missing

  Property mapping — type coercion per relationship type:
    7. test_director_of_is_active_bool_coercion
    8. test_supplies_volume_per_year_float_coercion
    9. test_registered_at_since_date_passthrough
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from graphrag_assistant.loaders.relationship_loader import (
    DataIntegrityError,
    _load_director_of,
    _load_registered_at,
    _load_supplies,
    _map_directorship,
    _map_registered_at,
    _map_supplies,
)

# ---------------------------------------------------------------------------
# Mock driver helpers
# ---------------------------------------------------------------------------

# CSV row counts (must match data/raw/ files written by T016):
_DIRECTOR_OF_ROW_COUNT = 20
_REGISTERED_AT_ROW_COUNT = 15  # one per company
_SUPPLIES_ROW_COUNT = 15  # one per product


def _make_driver(
    src_missing: bool = False,
    tgt_missing: bool = False,
    relationships_created: int = 0,
) -> MagicMock:
    """Build a mock neo4j.Driver for unit tests.

    session.run() dispatches on the 'ids' vs 'rows' parameter key:
    - {"ids": [...]}  → validation query: returns all IDs as found, OR empty if
                        src_missing (first call) / tgt_missing (second call).
    - {"rows": [...]} → MERGE query: returns a summary with the given
                        relationships_created counter.

    ids_call_count tracks whether we're on the first (source) or second (target)
    validation call so src_missing and tgt_missing can be applied independently.
    """
    summary = MagicMock()
    summary.counters.relationships_created = relationships_created
    merge_result = MagicMock()
    merge_result.consume.return_value = summary

    ids_call_count: list[int] = [0]

    def run_side_effect(cypher: str, params: dict | None = None) -> MagicMock:
        params = params or {}
        if "ids" in params:
            ids_call_count[0] += 1
            is_first = ids_call_count[0] == 1
            should_miss = (is_first and src_missing) or (not is_first and tgt_missing)
            found_ids = [] if should_miss else list(params["ids"])
            r = MagicMock()
            r.__iter__ = MagicMock(return_value=iter([{"id": id_} for id_ in found_ids]))
            return r
        # rows in params → MERGE call
        return merge_result

    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.run.side_effect = run_side_effect

    driver = MagicMock()
    driver.session.return_value = session
    return driver


# ---------------------------------------------------------------------------
# Idempotency tests
# ---------------------------------------------------------------------------


def test_director_of_zero_created_on_second_call() -> None:
    """When MERGE reports 0 relationships_created, all rows are treated as merged."""
    driver = _make_driver(relationships_created=0)
    created, merged = _load_director_of(driver)
    assert created == 0
    assert merged == _DIRECTOR_OF_ROW_COUNT


def test_registered_at_zero_created_on_second_call() -> None:
    """When MERGE reports 0 relationships_created, all rows are treated as merged."""
    driver = _make_driver(relationships_created=0)
    created, merged = _load_registered_at(driver)
    assert created == 0
    assert merged == _REGISTERED_AT_ROW_COUNT


def test_supplies_zero_created_on_second_call() -> None:
    """When MERGE reports 0 relationships_created, all rows are treated as merged."""
    driver = _make_driver(relationships_created=0)
    created, merged = _load_supplies(driver)
    assert created == 0
    assert merged == _SUPPLIES_ROW_COUNT


# ---------------------------------------------------------------------------
# DataIntegrityError tests
# ---------------------------------------------------------------------------


def test_data_integrity_error_missing_source_person() -> None:
    """DataIntegrityError is raised when source Person nodes are absent."""
    driver = _make_driver(src_missing=True)
    with pytest.raises(DataIntegrityError) as exc_info:
        _load_director_of(driver)
    # At least one source ID must appear in missing_ids
    assert len(exc_info.value.missing_ids) > 0


def test_data_integrity_error_missing_target_company() -> None:
    """DataIntegrityError is raised when target Company nodes are absent."""
    driver = _make_driver(tgt_missing=True)
    with pytest.raises(DataIntegrityError) as exc_info:
        _load_director_of(driver)
    assert len(exc_info.value.missing_ids) > 0


def test_data_integrity_error_both_endpoints_missing() -> None:
    """DataIntegrityError collects missing IDs from both source and target checks."""
    driver = _make_driver(src_missing=True, tgt_missing=True)
    with pytest.raises(DataIntegrityError) as exc_info:
        _load_director_of(driver)
    # Both source and target IDs are missing, so missing_ids should contain
    # entries from both validation queries.
    assert len(exc_info.value.missing_ids) >= 2


# ---------------------------------------------------------------------------
# Property mapping / type coercion tests
# ---------------------------------------------------------------------------


def test_director_of_is_active_bool_coercion() -> None:
    """'true'/'false' strings are coerced to Python booleans."""
    row_true = {
        "person_id": "P001",
        "company_id": "C001",
        "role": "Director",
        "appointed_date": "2020-01-01",
        "is_active": "true",
    }
    mapped_true = _map_directorship(row_true)
    assert mapped_true["is_active"] is True

    row_false = {**row_true, "is_active": "false"}
    mapped_false = _map_directorship(row_false)
    assert mapped_false["is_active"] is False

    # Case-insensitive
    row_upper = {**row_true, "is_active": "True"}
    mapped_upper = _map_directorship(row_upper)
    assert mapped_upper["is_active"] is True


def test_supplies_volume_per_year_float_coercion() -> None:
    """volume_per_year string is coerced to float."""
    row = {
        "company_id": "C002",
        "product_id": "PR001",
        "contract_id": "K001",
        "since_date": "2020-01-15",
        "volume_per_year": "12.5",
    }
    mapped = _map_supplies(row)
    assert isinstance(mapped["volume_per_year"], float)
    assert mapped["volume_per_year"] == pytest.approx(12.5)

    # Integer string also coerces to float
    row_int = {**row, "volume_per_year": "50"}
    mapped_int = _map_supplies(row_int)
    assert isinstance(mapped_int["volume_per_year"], float)
    assert mapped_int["volume_per_year"] == pytest.approx(50.0)


def test_registered_at_since_date_passthrough() -> None:
    """since_date is passed through as-is from registered_since column."""
    row = {
        "company_id": "C001",
        "name": "Meridian Holdings",
        "type": "Ltd",
        "registration_number": "12847391",
        "country": "UK",
        "registered_address_id": "A001",
        "registered_since": "2015-03-12",
    }
    mapped = _map_registered_at(row)
    assert mapped["since_date"] == "2015-03-12"
    assert mapped["company_id"] == "C001"
    assert mapped["address_id"] == "A001"
    # No extra fields
    assert set(mapped.keys()) == {"company_id", "address_id", "since_date"}
