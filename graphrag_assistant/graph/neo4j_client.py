from __future__ import annotations

import logging
from typing import Any

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import ClientError

logger = logging.getLogger(__name__)

# Neo4j error code returned when an identical schema rule already exists.
# Used to make bootstrap_schema idempotent — see _run_ddl.
_EQUIV_SCHEMA_CODE = "Neo.ClientError.Schema.EquivalentSchema" "RuleAlreadyExists"

# Static DDL strings — schema operations cannot be parameterised in Cypher.
# No runtime string formatting is used here.
_DDL_CHUNK_VECTOR_INDEX = (
    "CREATE VECTOR INDEX chunk_embedding_idx IF NOT EXISTS"
    " FOR (c:Chunk) ON (c.embedding)"
    " OPTIONS {indexConfig: {"
    "`vector.dimensions`: $dims,"
    " `vector.similarity_function`: 'cosine'"
    "}}"
)

_DDL_CONSTRAINTS: tuple[str, ...] = (
    "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (n:Company) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (n:Person) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT contract_id IF NOT EXISTS FOR (n:Contract) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE",
)


class Neo4jClient:
    """Thin wrapper around the Neo4j Python driver."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        """Close the underlying driver connection."""
        self._driver.close()

    def run_query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a read Cypher query and return results as a list of dicts.

        Dynamic values must be passed via *parameters*, never via string
        interpolation in *cypher*.
        """
        with self._driver.session() as session:
            result = session.run(cypher, parameters or {})
            return [dict(record) for record in result]

    def run_write(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Execute a write Cypher statement. No return value.

        Dynamic values must be passed via *parameters*, never via string
        interpolation in *cypher*.
        """
        with self._driver.session() as session:
            session.run(cypher, parameters or {})

    def verify_connectivity(self) -> None:
        """Raise if the driver cannot reach the configured Neo4j instance."""
        self._driver.verify_connectivity()

    def bootstrap_schema(self, embedding_dims: int = 384) -> None:
        """Create indexes and constraints; safe to call on an already-bootstrapped graph."""
        for ddl in _DDL_CONSTRAINTS:
            self._run_ddl(ddl)
        self._run_ddl(_DDL_CHUNK_VECTOR_INDEX, {"dims": embedding_dims})

    def _run_ddl(
        self,
        statement: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Execute a DDL statement, silencing duplicate-schema errors."""
        try:
            self.run_write(statement, parameters)
        except ClientError as exc:
            if _EQUIV_SCHEMA_CODE in str(exc):
                logger.debug("Schema rule already exists; skipping creation.")
            else:
                raise
