"""Unit tests for graphrag_assistant.providers.neo4j_vector."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from neo4j.exceptions import ClientError, ServiceUnavailable

from graphrag_assistant.providers.neo4j_vector import (
    ChunkResult,
    Neo4jVectorProvider,
    ProviderError,
)

_INDEX = "chunk_embedding_idx"
_EMB = [0.1, 0.2, 0.3, 0.4]


def _make_provider(mock_driver: MagicMock, index_name: str = _INDEX) -> Neo4jVectorProvider:
    return Neo4jVectorProvider(mock_driver, index_name)


def _wire_session(mock_driver: MagicMock, records: list) -> MagicMock:
    """Attach a mock session that returns *records* from session.run()."""
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = records
    return mock_session


def _make_record(chunk_id: str, doc_id: str, text: str, score: float) -> MagicMock:
    data = {"chunk_id": chunk_id, "doc_id": doc_id, "text": text, "score": score}
    record = MagicMock()
    record.__getitem__.side_effect = data.__getitem__
    return record


# ---------------------------------------------------------------------------
# Happy path: 3 results in descending score order
# ---------------------------------------------------------------------------


def test_query_happy_path_returns_three_results() -> None:
    mock_driver = MagicMock()
    records = [
        _make_record("c1", "d1", "alpha text", 0.95),
        _make_record("c2", "d2", "beta text", 0.80),
        _make_record("c3", "d3", "gamma text", 0.60),
    ]
    _wire_session(mock_driver, records)
    provider = _make_provider(mock_driver)

    results = provider.query(_EMB, k=3)

    assert len(results) == 3
    assert results[0].score == pytest.approx(0.95)
    assert results[1].score == pytest.approx(0.80)
    assert results[2].score == pytest.approx(0.60)
    assert all(isinstance(r, ChunkResult) for r in results)


# ---------------------------------------------------------------------------
# Empty result — no exception raised
# ---------------------------------------------------------------------------


def test_query_empty_result_returns_empty_list() -> None:
    mock_driver = MagicMock()
    _wire_session(mock_driver, [])
    provider = _make_provider(mock_driver)

    results = provider.query(_EMB, k=5)

    assert results == []


# ---------------------------------------------------------------------------
# k=1 boundary
# ---------------------------------------------------------------------------


def test_query_k_one_returns_exactly_one_result() -> None:
    mock_driver = MagicMock()
    _wire_session(mock_driver, [_make_record("c1", "d1", "only result", 0.75)])
    provider = _make_provider(mock_driver)

    results = provider.query(_EMB, k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "c1"


# ---------------------------------------------------------------------------
# Index not found → ProviderError containing the index name
# ---------------------------------------------------------------------------


def test_index_missing_raises_provider_error_with_index_name() -> None:
    index_name = "chunk_embedding_idx"
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
    mock_session.run.side_effect = ClientError(
        {
            "code": "Neo.ClientError.Procedure.ProcedureCallFailed",
            "message": f"There is no such vector schema index: {index_name}",
        }
    )
    provider = Neo4jVectorProvider(mock_driver, index_name)

    with pytest.raises(ProviderError) as exc_info:
        provider.query(_EMB, k=3)

    assert index_name in str(exc_info.value)


# ---------------------------------------------------------------------------
# ServiceUnavailable → ProviderError
# ---------------------------------------------------------------------------


def test_service_unavailable_raises_provider_error() -> None:
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
    mock_session.run.side_effect = ServiceUnavailable("Connection refused")
    provider = _make_provider(mock_driver)

    with pytest.raises(ProviderError) as exc_info:
        provider.query(_EMB, k=3)

    assert "unavailable" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Partial node properties → ProviderError (not a raw KeyError)
# ---------------------------------------------------------------------------


def test_missing_text_property_raises_provider_error() -> None:
    mock_driver = MagicMock()
    incomplete = MagicMock()
    # "text" key is absent — accessing it raises KeyError
    _stub = {"chunk_id": "c1", "doc_id": "d1", "score": 0.9}
    incomplete.__getitem__.side_effect = _stub.__getitem__
    _wire_session(mock_driver, [incomplete])
    provider = _make_provider(mock_driver)

    with pytest.raises(ProviderError):
        provider.query(_EMB, k=1)
