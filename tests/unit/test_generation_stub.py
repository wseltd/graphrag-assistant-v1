"""Unit tests for TemplateGenerationProvider."""
from __future__ import annotations

import pytest

from graphrag_assistant.providers.base import GenerationProvider
from graphrag_assistant.providers.generation_stub import (
    TemplateGenerationProvider,
)
from graphrag_assistant.schemas import AnswerSchema

_FACT = {
    "source_id": "company:1",
    "target_id": "company:2",
    "label": "SUPPLIES",
}
_CHUNK = {
    "doc_id": "doc:1",
    "chunk_id": "chunk:1",
    "text": "Acme Corp supplies widgets to Beta Inc under contract C-001.",
}


@pytest.fixture()
def provider() -> TemplateGenerationProvider:
    return TemplateGenerationProvider()


def test_both_populated_returns_valid_schema(provider: TemplateGenerationProvider) -> None:
    result = provider.generate(
        "[mode:graph_rag] Which company supplies widgets?",
        graph_facts=[_FACT],
        chunks=[_CHUNK],
    )
    assert isinstance(result, AnswerSchema)
    result.model_validate(result.model_dump())


def test_empty_graph_facts_returns_valid_schema(
    provider: TemplateGenerationProvider,
) -> None:
    result = provider.generate(
        "[mode:graph_rag] Query",
        graph_facts=[],
        chunks=[_CHUNK],
    )
    assert isinstance(result, AnswerSchema)


def test_empty_graph_facts_has_empty_graph_evidence(
    provider: TemplateGenerationProvider,
) -> None:
    result = provider.generate(
        "[mode:graph_rag] Query",
        graph_facts=[],
        chunks=[_CHUNK],
    )
    assert result.graph_evidence == []


def test_empty_chunks_returns_valid_schema(
    provider: TemplateGenerationProvider,
) -> None:
    result = provider.generate(
        "[mode:graph_rag] Query",
        graph_facts=[_FACT],
        chunks=[],
    )
    assert isinstance(result, AnswerSchema)


def test_empty_chunks_has_empty_text_citations(
    provider: TemplateGenerationProvider,
) -> None:
    result = provider.generate(
        "[mode:graph_rag] Query",
        graph_facts=[_FACT],
        chunks=[],
    )
    assert result.text_citations == []


def test_both_empty_returns_valid_schema(
    provider: TemplateGenerationProvider,
) -> None:
    result = provider.generate(
        "[mode:graph_rag] Query",
        graph_facts=[],
        chunks=[],
    )
    assert isinstance(result, AnswerSchema)


def test_both_empty_answer_is_non_empty(
    provider: TemplateGenerationProvider,
) -> None:
    result = provider.generate(
        "[mode:graph_rag] Query",
        graph_facts=[],
        chunks=[],
    )
    assert len(result.answer) > 0


def test_mode_plain_rag(provider: TemplateGenerationProvider) -> None:
    result = provider.generate(
        "[mode:plain_rag] Query",
        graph_facts=[],
        chunks=[],
    )
    assert result.mode == "plain_rag"


def test_mode_graph_rag(provider: TemplateGenerationProvider) -> None:
    result = provider.generate(
        "[mode:graph_rag] Query",
        graph_facts=[],
        chunks=[],
    )
    assert result.mode == "graph_rag"


def test_citation_quote_truncated_at_120_chars(
    provider: TemplateGenerationProvider,
) -> None:
    long_text = "x" * 200
    chunk = {"doc_id": "doc:1", "chunk_id": "chunk:1", "text": long_text}
    result = provider.generate(
        "[mode:plain_rag] Query",
        graph_facts=[],
        chunks=[chunk],
    )
    assert len(result.text_citations[0].quote) == 120


def test_isinstance_check(provider: TemplateGenerationProvider) -> None:
    assert isinstance(provider, GenerationProvider)
