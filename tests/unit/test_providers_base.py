"""Unit tests for graphrag_assistant.providers.base ABCs.

Six tests covering:
  - Cannot instantiate EmbeddingProvider directly (TypeError)
  - Cannot instantiate GenerationProvider directly (TypeError)
  - Cannot instantiate VectorProvider directly (TypeError)
  - Partial subclass (missing methods) raises TypeError at instantiation
  - Complete concrete subclass instantiates without error
  - VectorProvider.search signature has node_ids param with default None
"""
from __future__ import annotations

import inspect

import pytest

from graphrag_assistant.providers.base import (
    EmbeddingProvider,
    GenerationProvider,
    VectorProvider,
)
from graphrag_assistant.schemas import AnswerSchema, RetrievalDebug


def test_embedding_provider_not_instantiable() -> None:
    with pytest.raises(TypeError):
        EmbeddingProvider()  # type: ignore[abstract]


def test_generation_provider_not_instantiable() -> None:
    with pytest.raises(TypeError):
        GenerationProvider()  # type: ignore[abstract]


def test_vector_provider_not_instantiable() -> None:
    with pytest.raises(TypeError):
        VectorProvider()  # type: ignore[abstract]


def test_partial_subclass_raises_type_error() -> None:
    class _Partial(EmbeddingProvider):
        pass  # embed is not implemented

    with pytest.raises(TypeError):
        _Partial()  # type: ignore[abstract]


def test_complete_subclass_instantiates() -> None:
    class _Complete(EmbeddingProvider, GenerationProvider, VectorProvider):
        def embed(self, texts: list[str]) -> list[list[float]]:
            return []

        def generate(
            self,
            prompt: str,
            graph_facts: list[dict],
            chunks: list[dict],
        ) -> AnswerSchema:
            return AnswerSchema(
                answer="",
                graph_evidence=[],
                text_citations=[],
                retrieval_debug=RetrievalDebug(
                    graph_query="",
                    entity_matches=[],
                    retrieved_node_ids=[],
                    chunk_ids=[],
                    timings={},
                ),
                mode="plain_rag",
            )

        def search(
            self,
            vector: list[float],
            top_k: int,
            node_ids: list[str] | None = None,
        ) -> list[dict]:
            return []

    instance = _Complete()
    assert instance is not None


def test_vector_provider_search_has_node_ids_param() -> None:
    sig = inspect.signature(VectorProvider.search)
    assert "node_ids" in sig.parameters
    assert sig.parameters["node_ids"].default is None
