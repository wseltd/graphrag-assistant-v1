"""Unit tests for graphrag_assistant.loaders.entity_loader (T015).

9 tests:
  Idempotency — ON CREATE / ON MATCH present in Cypher:
    1. test_company_merge_has_on_create_and_on_match
    2. test_person_merge_has_on_create_and_on_match
    3. test_address_and_product_merge_have_on_create_and_on_match

  Property mapping — CSV columns map to correct node property names:
    4. test_company_property_mapping_all_csv_columns
    5. test_person_property_mapping_all_csv_columns
    6. test_address_and_product_property_mapping_all_csv_columns

  Constraint creation — all four IF NOT EXISTS statements issued before MERGE:
    7. test_exactly_four_constraint_stmts_defined
    8. test_all_constraint_stmts_contain_if_not_exists
    9. test_constraints_issued_before_merge_calls
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from graphrag_assistant.loaders.entity_loader import (
    _CONSTRAINT_STMTS,
    _MERGE_ADDRESS,
    _MERGE_COMPANY,
    _MERGE_PERSON,
    _MERGE_PRODUCT,
    _map_addresses,
    _map_companies,
    _map_people,
    _map_products,
    load_entities,
)

# ---------------------------------------------------------------------------
# Helper — build a mock neo4j.Driver
# ---------------------------------------------------------------------------


def _make_driver(nodes_created: int = 0) -> MagicMock:
    """Return a mock Driver whose session().run().consume().counters.nodes_created
    always returns *nodes_created*."""
    mock_summary = MagicMock()
    mock_summary.counters.nodes_created = nodes_created

    mock_result = MagicMock()
    mock_result.consume.return_value = mock_summary

    mock_session = MagicMock()
    mock_session.run.return_value = mock_result
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session
    return mock_driver


# ---------------------------------------------------------------------------
# Idempotency: ON CREATE SET and ON MATCH SET both present
# ---------------------------------------------------------------------------


def test_company_merge_has_on_create_and_on_match() -> None:
    assert "ON CREATE SET" in _MERGE_COMPANY
    assert "ON MATCH SET" in _MERGE_COMPANY


def test_person_merge_has_on_create_and_on_match() -> None:
    assert "ON CREATE SET" in _MERGE_PERSON
    assert "ON MATCH SET" in _MERGE_PERSON


def test_address_and_product_merge_have_on_create_and_on_match() -> None:
    assert "ON CREATE SET" in _MERGE_ADDRESS
    assert "ON MATCH SET" in _MERGE_ADDRESS
    assert "ON CREATE SET" in _MERGE_PRODUCT
    assert "ON MATCH SET" in _MERGE_PRODUCT


# ---------------------------------------------------------------------------
# Property mapping
# ---------------------------------------------------------------------------


def test_company_property_mapping_all_csv_columns() -> None:
    row = {
        "company_id": "C001",
        "name": "ACME Ltd",
        "type": "Ltd",
        "registration_number": "12345678",
        "country": "UK",
        "registered_address_id": "A001",  # FK — must be excluded
    }
    mapped = _map_companies([row])[0]
    assert mapped["id"] == "C001"
    assert mapped["name"] == "ACME Ltd"
    assert mapped["type"] == "Ltd"
    assert mapped["registration_number"] == "12345678"
    assert mapped["country"] == "UK"
    assert "registered_address_id" not in mapped


def test_person_property_mapping_all_csv_columns() -> None:
    row = {
        "person_id": "P001",
        "full_name": "Jane Doe",
        "title": "Dr",
        "nationality": "British",
        "job_title": "CTO",
        "company_id": "C001",  # FK — must be excluded
        "address_id": "A001",  # FK — must be excluded
        "email": "jane@example.com",
    }
    mapped = _map_people([row])[0]
    assert mapped["id"] == "P001"
    assert mapped["name"] == "Jane Doe"  # full_name → name
    assert mapped["title"] == "Dr"
    assert mapped["nationality"] == "British"
    assert mapped["job_title"] == "CTO"
    assert mapped["email"] == "jane@example.com"
    assert "full_name" not in mapped
    assert "company_id" not in mapped
    assert "address_id" not in mapped


def test_address_and_product_property_mapping_all_csv_columns() -> None:
    addr_row = {
        "address_id": "A001",
        "street": "1 Main St",
        "city": "London",
        "postcode": "EC1A 1BB",
        "country": "UK",
    }
    mapped_addr = _map_addresses([addr_row])[0]
    assert mapped_addr["id"] == "A001"
    assert mapped_addr["street"] == "1 Main St"
    assert mapped_addr["city"] == "London"
    assert mapped_addr["postcode"] == "EC1A 1BB"
    assert mapped_addr["country"] == "UK"
    assert "address_id" not in mapped_addr

    prod_row = {
        "product_id": "PR001",
        "name": "Widget",
        "category": "Hardware",
        "unit": "unit",
        "unit_price_gbp": "99.99",
        "supplier_company_id": "C001",  # FK — must be excluded
    }
    mapped_prod = _map_products([prod_row])[0]
    assert mapped_prod["id"] == "PR001"
    assert mapped_prod["name"] == "Widget"
    assert mapped_prod["category"] == "Hardware"
    assert mapped_prod["unit"] == "unit"
    assert mapped_prod["unit_price"] == pytest.approx(99.99)
    assert "unit_price_gbp" not in mapped_prod  # renamed to unit_price
    assert "supplier_company_id" not in mapped_prod
    assert "product_id" not in mapped_prod


# ---------------------------------------------------------------------------
# Constraint creation
# ---------------------------------------------------------------------------


def test_exactly_four_constraint_stmts_defined() -> None:
    assert len(_CONSTRAINT_STMTS) == 4


def test_all_constraint_stmts_contain_if_not_exists() -> None:
    for stmt in _CONSTRAINT_STMTS:
        assert "IF NOT EXISTS" in stmt.upper(), (
            f"Constraint statement missing IF NOT EXISTS: {stmt!r}"
        )


def test_constraints_issued_before_merge_calls() -> None:
    """All four CREATE CONSTRAINT calls must appear before any MERGE/UNWIND in
    the sequence of session.run() invocations made by load_entities()."""
    driver = _make_driver()
    load_entities(driver)

    # All run() calls accumulate on the same mock session (driver.session always
    # returns the same context-manager mock whose __enter__ returns itself).
    session_mock = driver.session.return_value
    all_cypher: list[str] = [
        c.args[0] if c.args else ""
        for c in session_mock.run.call_args_list
    ]

    constraint_indices = [i for i, q in enumerate(all_cypher) if "CREATE CONSTRAINT" in q]
    merge_indices = [i for i, q in enumerate(all_cypher) if "UNWIND" in q and "MERGE" in q]

    assert len(constraint_indices) == 4, (
        f"Expected 4 schema-setup calls, got {len(constraint_indices)}; all calls: {all_cypher}"
    )
    assert len(merge_indices) == 4, (
        f"Expected 4 MERGE calls, got {len(merge_indices)}: {all_cypher}"
    )
    assert max(constraint_indices) < min(merge_indices), (
        "All CREATE CONSTRAINT calls must precede any MERGE call"
    )
