"""Neo4j native vector-index implementation of VectorProvider.

Design:
- One Cypher CALL to db.index.vector.queryNodes; results mapped to ChunkResult.
- Raises ProviderError (not raw Neo4j exceptions) so callers have a stable contract.
- No re-ranking, no hybrid search, no index creation, no caching.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from neo4j import Driver
from neo4j.exceptions import ClientError, ServiceUnavailable

from graphrag_assistant.providers.base import VectorProvider

logger = logging.getLogger(__name__)

# Cypher uses $index_name so the index name is never string-interpolated into Cypher.
_CYPHER = (
    "CALL db.index.vector.queryNodes($index_name, $k, $embedding) "
    "YIELD node, score "
    "RETURN node.chunk_id AS chunk_id, node.doc_id AS doc_id, "
    "node.text AS text, score "
    "ORDER BY score DESC"
)


class ProviderError(Exception):
    """Raised when Neo4jVectorProvider cannot complete a vector query."""

    def __repr__(self) -> str:
        msg = self.args[0] if self.args else ""
        return f"{self.__class__.__name__}({msg!r})"


@dataclass
class ChunkResult:
    """A single result returned by Neo4jVectorProvider.query."""

    chunk_id: str
    doc_id: str
    score: float
    text: str

    def __repr__(self) -> str:
        return (
            f"ChunkResult(chunk_id={self.chunk_id!r}, doc_id={self.doc_id!r}, "
            f"score={self.score:.4f})"
        )


class Neo4jVectorProvider(VectorProvider):
    """VectorProvider backed by Neo4j 5.x native vector index.

    The index name is injected at construction time; no module-level globals.
    Vector index creation is the responsibility of the ingestion layer.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(index_name={self._index_name!r})"

    def __init__(self, driver: Driver, index_name: str) -> None:
        self._driver = driver
        self._index_name = index_name

    def query(self, embedding: list[float], k: int) -> list[ChunkResult]:
        """Query the vector index; return at most k ChunkResult objects ordered descending by score.

        Returns an empty list when the index is empty.  Raises ProviderError
        (never a raw Neo4j exception) for all failure modes.
        """
        try:
            with self._driver.session() as session:
                result = session.run(
                    _CYPHER,
                    {"index_name": self._index_name, "k": k, "embedding": embedding},
                )
                records = list(result)
        except ServiceUnavailable as exc:
            raise ProviderError(f"Neo4j service unavailable: {exc}") from exc
        except ClientError as exc:
            msg = str(exc)
            if self._index_name in msg or "index" in msg.lower():
                raise ProviderError(
                    f"Vector index '{self._index_name}' does not exist or is unavailable: {exc}"
                ) from exc
            raise ProviderError(f"Neo4j query failed: {exc}") from exc

        results: list[ChunkResult] = []
        for record in records:
            try:
                chunk_id = record["chunk_id"]
                doc_id = record["doc_id"]
                score = float(record["score"])
                text = record["text"]
            except (KeyError, TypeError) as exc:
                raise ProviderError(
                    f"Chunk node is missing required properties: {exc}"
                ) from exc
            if chunk_id is None or doc_id is None or text is None:
                raise ProviderError(
                    "Chunk node returned null for a required property (chunk_id, doc_id, or text)"
                )
            results.append(ChunkResult(chunk_id=chunk_id, doc_id=doc_id, score=score, text=text))
        return results

    def search(
        self,
        vector: list[float],
        top_k: int,
        node_ids: list[str] | None = None,
    ) -> list[dict]:
        """Satisfy VectorProvider ABC; delegates to query.

        node_ids graph-constrained filtering is deferred to the graph-retrieval ticket.
        """
        results = self.query(vector, top_k)
        return [
            {"chunk_id": r.chunk_id, "doc_id": r.doc_id, "score": r.score, "text": r.text}
            for r in results
        ]
