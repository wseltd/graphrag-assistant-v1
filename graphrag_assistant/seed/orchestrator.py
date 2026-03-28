"""Seed orchestrator: coordinates the full graph seeding pipeline (T019.b).

Public API
----------
seed(driver, reset=False)
    Optionally reset the graph, then load base entities, contracts/clauses,
    and chunks in fixed order. Returns a summary dict of written counts.

Design
------
* Three loader wrappers (load_base_entities, load_contracts_and_clauses,
  load_chunks_and_edges) each accept only a driver and handle their own I/O.
* seed() calls them in fixed order; any exception propagates immediately.
* No parallelism, no retry, no progress tracking.
* The embedding provider is instantiated inside load_chunks_and_edges so
  construction cost is paid once per seed call, not at import time.
* load_contracts_and_clauses uses execute_write for the Contract/Clause/
  HAS_CLAUSE batch, then runs PARTY_TO in the same session (post-commit)
  so that the party-existence MATCH sees the freshly committed nodes.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import neo4j

from graphrag_assistant.loaders.chunk_reader import load_chunks, load_chunks_to_neo4j
from graphrag_assistant.loaders.contract_loader import (
    load_clauses,
    load_contracts,
    load_has_clause_edges,
    load_party_to_edges,
)
from graphrag_assistant.loaders.entity_loader import load_entities
from graphrag_assistant.loaders.relationship_loader import load_relationships
from graphrag_assistant.providers.embedding import SentenceTransformerProvider
from graphrag_assistant.seed.reset import reset_graph

logger = logging.getLogger(__name__)

_DATA_RAW = Path(__file__).parent.parent.parent / "data" / "raw"
_CHUNKS_PATH = str(
    Path(__file__).parent.parent.parent / "data" / "processed" / "chunks.jsonl"
)


def _read_csv(filename: str) -> list[dict[str, Any]]:
    path = _DATA_RAW / filename
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_base_entities(driver: neo4j.Driver) -> dict[str, int]:
    """Load Company, Person, Address, Product nodes and base relationships.

    Returns total nodes and edges processed (created + merged = CSV row count)
    so the counts are stable across repeated calls without reset.
    """
    entity_result = load_entities(driver)
    rel_result = load_relationships(driver)
    return {
        "nodes_written": entity_result.nodes_created + entity_result.nodes_merged,
        "edges_written": rel_result.edges_created + rel_result.edges_merged,
    }


def load_contracts_and_clauses(driver: neo4j.Driver) -> dict[str, int]:
    """Load Contract and Clause nodes, HAS_CLAUSE edges, and PARTY_TO edges.

    Reads contracts.csv and contract_clauses.csv from data/raw/.
    Contract/Clause/HAS_CLAUSE writes run inside a single execute_write
    transaction. PARTY_TO edges run in the same session after commit so
    that the MATCH on party nodes sees the committed data.
    """
    contract_rows = _read_csv("contracts.csv")
    clause_rows = _read_csv("contract_clauses.csv")

    def _write(tx: Any) -> None:
        load_contracts(tx, contract_rows)
        load_clauses(tx, clause_rows)
        load_has_clause_edges(tx, clause_rows)

    with driver.session() as session:
        session.execute_write(_write)
        party_to_count = load_party_to_edges(session, contract_rows)

    return {
        "nodes_written": len(contract_rows) + len(clause_rows),
        "edges_written": len(clause_rows) + party_to_count,
    }


def load_chunks_and_edges(driver: neo4j.Driver) -> dict[str, int]:
    """Load Chunk nodes and FROM_CONTRACT / ABOUT_COMPANY / RELATED_TO edges.

    Reads data/processed/chunks.jsonl and embeds texts with the local model.
    Returns loaded chunk count and relationship counts from the Cypher summaries.
    """
    chunks = load_chunks(_CHUNKS_PATH)
    provider = SentenceTransformerProvider()
    result = load_chunks_to_neo4j(chunks, driver, provider)
    return {
        "nodes_written": result.chunks_loaded,
        "edges_written": (
            result.from_contract_edges
            + result.about_company_edges
            + result.related_to_edges
        ),
    }


def seed(driver: neo4j.Driver, reset: bool = False) -> dict[str, int]:
    """Run the full graph seeding pipeline.

    When *reset* is True the graph is cleared before any loader runs.
    Loaders execute in fixed order: base entities → contracts/clauses → chunks.
    Any exception from a loader propagates immediately without suppression.

    Args:
        driver: Live Neo4j driver.
        reset:  Clear the graph before loading when True.

    Returns:
        Dict with 'nodes_written' and 'edges_written' totals across all loaders.
    """
    if reset:
        reset_graph(driver)
        logger.info("seed: graph cleared")

    total_nodes = 0
    total_edges = 0

    for loader in (load_base_entities, load_contracts_and_clauses, load_chunks_and_edges):
        counts = loader(driver)
        total_nodes += counts["nodes_written"]
        total_edges += counts["edges_written"]
        logger.info(
            "seed: %s → nodes=%d edges=%d",
            getattr(loader, "__name__", repr(loader)),
            counts["nodes_written"],
            counts["edges_written"],
        )

    return {"nodes_written": total_nodes, "edges_written": total_edges}
