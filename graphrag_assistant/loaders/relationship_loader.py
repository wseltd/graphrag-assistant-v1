"""Relationship loader: reads directorships.csv, supplies.csv, and companies.csv,
then MERGEs DIRECTOR_OF, REGISTERED_AT, and SUPPLIES edges between already-loaded
nodes in Neo4j.

Public API:  load_relationships(driver) -> LoadResult

Design
------
Each relationship type runs three phases:
  Phase 1  UNWIND + MATCH (not MERGE) — find which input IDs have matching nodes.
           Runs as two separate queries: one for source label, one for target label.
  Phase 2  Python set-difference check — any input ID absent from phase-1 results
           is collected; if the set is non-empty, DataIntegrityError is raised before
           any edges are written (all-or-nothing per relationship type).
  Phase 3  UNWIND + MERGE (src)-[r:REL_TYPE]->(tgt) ON CREATE SET … — writes edges
           only when all endpoints are confirmed present.

Silent skips are explicitly prohibited.  A missing endpoint raises DataIntegrityError
with the list of missing IDs so the caller can act on actionable information.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neo4j import Driver

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class DataIntegrityError(Exception):
    """Raised when a required endpoint node is absent from Neo4j.

    Attributes
    ----------
    missing_ids:
        The IDs that could not be matched to existing nodes.
    """

    def __init__(self, missing_ids: list[str]) -> None:
        self.missing_ids = missing_ids
        super().__init__(f"Missing endpoint nodes: {missing_ids!r}")

    def __repr__(self) -> str:
        return f"DataIntegrityError(missing_ids={self.missing_ids!r})"


@dataclass
class LoadResult:
    edges_created: int
    edges_merged: int


# ---------------------------------------------------------------------------
# Cypher: MERGE statements (phase 3)
# ---------------------------------------------------------------------------

_MERGE_DIRECTOR_OF = (
    "UNWIND $rows AS row "
    "MATCH (p:Person {id: row.person_id}) "
    "MATCH (c:Company {id: row.company_id}) "
    "MERGE (p)-[r:DIRECTOR_OF]->(c) "
    "ON CREATE SET r.role = row.role, r.appointed_date = row.appointed_date, "
    "r.is_active = row.is_active"
)

_MERGE_REGISTERED_AT = (
    "UNWIND $rows AS row "
    "MATCH (c:Company {id: row.company_id}) "
    "MATCH (a:Address {id: row.address_id}) "
    "MERGE (c)-[r:REGISTERED_AT]->(a) "
    "ON CREATE SET r.since_date = row.since_date"
)

_MERGE_SUPPLIES = (
    "UNWIND $rows AS row "
    "MATCH (c:Company {id: row.company_id}) "
    "MATCH (p:Product {id: row.product_id}) "
    "MERGE (c)-[r:SUPPLIES]->(p) "
    "ON CREATE SET r.contract_id = row.contract_id, r.since_date = row.since_date, "
    "r.volume_per_year = row.volume_per_year"
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _find_missing_ids(session: Any, label: str, ids: list[str]) -> list[str]:
    """Return every id in *ids* that has no matching node with label *label* in Neo4j.

    Uses MATCH (not OPTIONAL MATCH) so only existing nodes appear in results.
    The set-difference between input and found IDs gives the missing ones.
    """
    if not ids:
        return []
    cypher = (
        "UNWIND $ids AS lookup_id "
        f"MATCH (n:{label} {{id: lookup_id}}) "
        "RETURN n.id AS id"
    )
    found = {record["id"] for record in session.run(cypher, {"ids": ids})}
    return [id_ for id_ in ids if id_ not in found]


def _map_directorship(row: dict[str, str]) -> dict[str, Any]:
    return {
        "person_id": row["person_id"],
        "company_id": row["company_id"],
        "role": row["role"],
        "appointed_date": row["appointed_date"],
        "is_active": row["is_active"].strip().lower() == "true",
    }


def _map_registered_at(row: dict[str, str]) -> dict[str, Any]:
    return {
        "company_id": row["company_id"],
        "address_id": row["registered_address_id"],
        "since_date": row["registered_since"],
    }


def _map_supplies(row: dict[str, str]) -> dict[str, Any]:
    return {
        "company_id": row["company_id"],
        "product_id": row["product_id"],
        "contract_id": row["contract_id"],
        "since_date": row["since_date"],
        "volume_per_year": float(row["volume_per_year"]),
    }


# ---------------------------------------------------------------------------
# Per-type loaders (phases 1–3 each)
# ---------------------------------------------------------------------------


def _load_director_of(driver: Driver) -> tuple[int, int]:
    """MERGE DIRECTOR_OF edges from directorships.csv.

    Returns (edges_created, edges_merged).
    Raises DataIntegrityError if any person_id or company_id is absent.
    """
    rows_raw = _read_csv(_DATA_DIR / "directorships.csv")
    rows = [_map_directorship(r) for r in rows_raw]
    person_ids = [r["person_id"] for r in rows]
    company_ids = [r["company_id"] for r in rows]

    with driver.session() as session:
        missing: list[str] = []
        missing += _find_missing_ids(session, "Person", person_ids)
        missing += _find_missing_ids(session, "Company", company_ids)
        if missing:
            raise DataIntegrityError(missing)
        result = session.run(_MERGE_DIRECTOR_OF, {"rows": rows})
        created = result.consume().counters.relationships_created

    merged = len(rows) - created
    logger.info("DIRECTOR_OF: created=%d merged=%d", created, merged)
    return created, merged


def _load_registered_at(driver: Driver) -> tuple[int, int]:
    """MERGE REGISTERED_AT edges from companies.csv (registered_address_id column).

    Returns (edges_created, edges_merged).
    Raises DataIntegrityError if any company_id or address_id is absent.
    """
    rows_raw = _read_csv(_DATA_DIR / "companies.csv")
    rows = [_map_registered_at(r) for r in rows_raw]
    company_ids = [r["company_id"] for r in rows]
    address_ids = [r["address_id"] for r in rows]

    with driver.session() as session:
        missing: list[str] = []
        missing += _find_missing_ids(session, "Company", company_ids)
        missing += _find_missing_ids(session, "Address", address_ids)
        if missing:
            raise DataIntegrityError(missing)
        result = session.run(_MERGE_REGISTERED_AT, {"rows": rows})
        created = result.consume().counters.relationships_created

    merged = len(rows) - created
    logger.info("REGISTERED_AT: created=%d merged=%d", created, merged)
    return created, merged


def _load_supplies(driver: Driver) -> tuple[int, int]:
    """MERGE SUPPLIES edges from supplies.csv.

    Returns (edges_created, edges_merged).
    Raises DataIntegrityError if any company_id or product_id is absent.
    """
    rows_raw = _read_csv(_DATA_DIR / "supplies.csv")
    rows = [_map_supplies(r) for r in rows_raw]
    company_ids = [r["company_id"] for r in rows]
    product_ids = [r["product_id"] for r in rows]

    with driver.session() as session:
        missing: list[str] = []
        missing += _find_missing_ids(session, "Company", company_ids)
        missing += _find_missing_ids(session, "Product", product_ids)
        if missing:
            raise DataIntegrityError(missing)
        result = session.run(_MERGE_SUPPLIES, {"rows": rows})
        created = result.consume().counters.relationships_created

    merged = len(rows) - created
    logger.info("SUPPLIES: created=%d merged=%d", created, merged)
    return created, merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_relationships(driver: Driver) -> LoadResult:
    """Load DIRECTOR_OF, REGISTERED_AT, and SUPPLIES edges from data/raw/ CSVs.

    All endpoint nodes must already exist (loaded by load_entities).
    Each relationship type is validated all-or-nothing: if any endpoint node is
    missing, DataIntegrityError is raised and no edges for that type are written.
    MERGE is fully idempotent; a second call produces zero new edges.

    Raises
    ------
    DataIntegrityError
        If any source or target node referenced by a CSV row is absent from Neo4j.
    """
    total_created = 0
    total_merged = 0
    for loader in (_load_director_of, _load_registered_at, _load_supplies):
        created, merged = loader(driver)
        total_created += created
        total_merged += merged
    return LoadResult(edges_created=total_created, edges_merged=total_merged)
