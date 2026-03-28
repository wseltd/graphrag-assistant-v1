from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

COMPANY_TYPE_ALLOWLIST = {"Ltd", "PLC", "GmbH", "LLC", "SA"}


def _load_companies() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "companies.csv", dtype=str)


def _load_people() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "people.csv", dtype=str)


def _load_addresses() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "addresses.csv", dtype=str)


def _load_products() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "products.csv", dtype=str)


def test_required_columns_present() -> None:
    companies = _load_companies()
    assert {"company_id", "name", "type", "registration_number", "country"}.issubset(
        companies.columns
    )

    people = _load_people()
    assert {"person_id", "full_name", "title", "nationality"}.issubset(people.columns)

    addresses = _load_addresses()
    assert {"address_id", "street", "city", "postcode", "country"}.issubset(
        addresses.columns
    )

    products = _load_products()
    assert {"product_id", "name", "category", "unit_price_gbp"}.issubset(
        products.columns
    )


def test_company_id_uniqueness() -> None:
    companies = _load_companies()
    assert not companies["company_id"].duplicated().any(), (
        "Duplicate company_id values found"
    )


def test_person_id_uniqueness() -> None:
    people = _load_people()
    assert not people["person_id"].duplicated().any(), (
        "Duplicate person_id values found"
    )


def test_address_id_uniqueness() -> None:
    addresses = _load_addresses()
    assert not addresses["address_id"].duplicated().any(), (
        "Duplicate address_id values found"
    )


def test_product_id_uniqueness() -> None:
    products = _load_products()
    assert not products["product_id"].duplicated().any(), (
        "Duplicate product_id values found"
    )


def test_no_null_ids() -> None:
    checks = [
        (_load_companies(), "company_id"),
        (_load_people(), "person_id"),
        (_load_addresses(), "address_id"),
        (_load_products(), "product_id"),
    ]
    for df, col in checks:
        assert df[col].notna().all(), f"Null values found in {col}"
        assert (df[col].str.strip() != "").all(), f"Empty strings found in {col}"


def test_unit_price_numeric_and_positive() -> None:
    products = _load_products()
    prices = pd.to_numeric(products["unit_price_gbp"], errors="coerce")
    assert prices.notna().all(), "unit_price_gbp contains non-numeric values"
    assert (prices > 0).all(), "unit_price_gbp contains non-positive values"


def test_company_type_enum() -> None:
    companies = _load_companies()
    invalid = set(companies["type"].unique()) - COMPANY_TYPE_ALLOWLIST
    assert not invalid, f"Unexpected company type values: {invalid}"
