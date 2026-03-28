"""Entity node loader: reads Company, Person, Address, Product CSVs and
MERGEs all nodes into Neo4j.

Public API: load_entities(driver) -> LoadResult

Design decisions
----------------
* Constraints are issued before any MERGE so a concurrent or repeated call
  cannot produce duplicate nodes before the constraint exists.
* ON CREATE SET populates every property; ON MATCH SET updates only mutable
  fields (name, unit_price) so re-seeding with corrected CSVs propagates
  changes without creating duplicates.
* id is never overwritten on MATCH — the MERGE key already guarantees it.
* Counters are derived from Cypher summary objects, not from local data sizes.
* FK columns (registered_address_id, company_id, address_id,
  supplier_company_id) are intentionally excluded; relationships are T016.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neo4j import Driver
from neo4j.exceptions import ClientError

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

# Constraint names match neo4j_client.py for Company and Person so that
# IF NOT EXISTS silences duplicate-name attempts when bootstrap_schema() ran first.
_CONSTRAINT_STMTS: tuple[str, ...] = (
    "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (n:Company) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (n:Person) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT address_id IF NOT EXISTS FOR (n:Address) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT product_id IF NOT EXISTS FOR (n:Product) REQUIRE n.id IS UNIQUE",
)

# Silenced when an equivalent constraint already exists under a different name.
_EQUIV_SCHEMA_CODE = "Neo.ClientError.Schema.EquivalentSchema" "RuleAlreadyExists"

# ---------------------------------------------------------------------------
# MERGE Cypher — one statement per label.
# ON CREATE: populate all properties.
# ON MATCH: update mutable fields only (name, unit_price); never overwrite id.
# ---------------------------------------------------------------------------

_MERGE_COMPANY = (
    "UNWIND $rows AS row "
    "MERGE (n:Company {id: row.id}) "
    "ON CREATE SET n.name = row.name, n.type = row.type, "
    "n.registration_number = row.registration_number, n.country = row.country "
    "ON MATCH SET n.name = row.name"
)

_MERGE_PERSON = (
    "UNWIND $rows AS row "
    "MERGE (n:Person {id: row.id}) "
    "ON CREATE SET n.name = row.name, n.title = row.title, "
    "n.nationality = row.nationality, n.job_title = row.job_title, n.email = row.email "
    "ON MATCH SET n.name = row.name"
)

_MERGE_ADDRESS = (
    "UNWIND $rows AS row "
    "MERGE (n:Address {id: row.id}) "
    "ON CREATE SET n.street = row.street, n.city = row.city, "
    "n.postcode = row.postcode, n.country = row.country "
    "ON MATCH SET n.street = row.street, n.city = row.city, "
    "n.postcode = row.postcode, n.country = row.country"
)

_MERGE_PRODUCT = (
    "UNWIND $rows AS row "
    "MERGE (n:Product {id: row.id}) "
    "ON CREATE SET n.name = row.name, n.category = row.category, "
    "n.unit = row.unit, n.unit_price = row.unit_price "
    "ON MATCH SET n.name = row.name, n.unit_price = row.unit_price"
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class LoadResult:
    nodes_created: int
    nodes_merged: int
    constraints_ensured: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _map_companies(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "id": r["company_id"],
            "name": r["name"],
            "type": r["type"],
            "registration_number": r["registration_number"],
            "country": r["country"],
        }
        for r in rows
    ]


def _map_people(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "id": r["person_id"],
            "name": r["full_name"],
            "title": r["title"],
            "nationality": r["nationality"],
            "job_title": r["job_title"],
            "email": r["email"],
        }
        for r in rows
    ]


def _map_addresses(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "id": r["address_id"],
            "street": r["street"],
            "city": r["city"],
            "postcode": r["postcode"],
            "country": r["country"],
        }
        for r in rows
    ]


def _map_products(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "id": r["product_id"],
            "name": r["name"],
            "category": r["category"],
            "unit": r["unit"],
            "unit_price": float(r["unit_price_gbp"]),
        }
        for r in rows
    ]


def _run_merge(driver: Driver, cypher: str, rows: list[dict[str, Any]]) -> int:
    """Execute a batch MERGE and return the number of nodes actually created."""
    with driver.session() as session:
        result = session.run(cypher, {"rows": rows})
        summary = result.consume()
        return summary.counters.nodes_created


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_entities(driver: Driver) -> LoadResult:
    """Load Company, Person, Address, and Product nodes from data/raw/ CSVs.

    Safe to call multiple times — MERGE is idempotent and constraints are
    created with IF NOT EXISTS.  On a clean database nodes_merged will be 0;
    on subsequent calls nodes_created will be 0.
    """
    # Step 1 — constraints must precede any MERGE
    with driver.session() as session:
        for stmt in _CONSTRAINT_STMTS:
            try:
                session.run(stmt)
            except ClientError as exc:
                if _EQUIV_SCHEMA_CODE in str(exc):
                    logger.debug("Equivalent constraint already exists; skipping.")
                else:
                    raise

    # Step 2 — read CSVs
    company_rows = _map_companies(_read_csv(_DATA_DIR / "companies.csv"))
    person_rows = _map_people(_read_csv(_DATA_DIR / "people.csv"))
    address_rows = _map_addresses(_read_csv(_DATA_DIR / "addresses.csv"))
    product_rows = _map_products(_read_csv(_DATA_DIR / "products.csv"))

    # Step 3 — MERGE batches, accumulate counters from Cypher summaries
    batches: list[tuple[list[dict[str, Any]], str]] = [
        (company_rows, _MERGE_COMPANY),
        (person_rows, _MERGE_PERSON),
        (address_rows, _MERGE_ADDRESS),
        (product_rows, _MERGE_PRODUCT),
    ]

    nodes_created = 0
    total_rows = 0
    for rows, cypher in batches:
        nodes_created += _run_merge(driver, cypher, rows)
        total_rows += len(rows)

    return LoadResult(
        nodes_created=nodes_created,
        nodes_merged=total_rows - nodes_created,
        constraints_ensured=len(_CONSTRAINT_STMTS),
    )
