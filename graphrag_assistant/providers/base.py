"""Abstract base classes for pluggable providers.

Three contracts are defined here and nowhere else:

- EmbeddingProvider  — text -> dense vector
- GenerationProvider — graph facts + chunks -> AnswerSchema
- VectorProvider     — vector -> top-k matching chunks, optionally
                       constrained to a set of graph-resolved node IDs

The node_ids parameter on VectorProvider.search is the join-point between
graph retrieval and vector retrieval.  Implementations must honour it by
restricting candidates to those node IDs, enabling graph-first retrieval.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from graphrag_assistant.schemas import AnswerSchema


class EmbeddingProvider(ABC):
    """Converts raw text into dense embedding vectors."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of float vectors in the same order as *texts*.
        """
        pass


class GenerationProvider(ABC):
    """Synthesises a final answer from graph evidence and retrieved chunks."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        graph_facts: list[dict],
        chunks: list[dict],
    ) -> AnswerSchema:
        """Generate an answer and populate the full AnswerSchema.

        Args:
            prompt:      The original user question.
            graph_facts: Graph triples returned by the retrieval stage,
                         each a dict with at minimum
                         {"source_id", "target_id", "label"}.
            chunks:      Retrieved text chunks, each a dict with at
                         minimum {"doc_id", "chunk_id", "text"}.

        Returns:
            A fully populated AnswerSchema including graph_evidence,
            text_citations, retrieval_debug, and mode.
        """
        pass


class VectorProvider(ABC):
    """Retrieves semantically similar chunks from a vector index."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def search(
        self,
        vector: list[float],
        top_k: int,
        node_ids: list[str] | None = None,
    ) -> list[dict]:
        """Return the top-k most similar chunks.

        Args:
            vector:   Query embedding produced by EmbeddingProvider.
            top_k:    Maximum number of results to return.
            node_ids: When supplied, restrict candidates to chunks whose
                      associated node ID is in this list.  This is the
                      graph-constrained filter: implementations MUST
                      honour it so that graph-first retrieval can narrow
                      the vector search space.  When None, search is
                      unconstrained.

        Returns:
            List of chunk dicts ordered by descending similarity score.
            Each dict contains at minimum
            {"chunk_id", "doc_id", "text", "score"}.
        """
        pass
