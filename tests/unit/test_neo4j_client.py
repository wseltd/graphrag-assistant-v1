"""Unit tests for graphrag_assistant.graph.neo4j_client."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from neo4j.exceptions import ClientError

from graphrag_assistant.graph.neo4j_client import (
    _EQUIV_SCHEMA_CODE,
    Neo4jClient,
)


@pytest.fixture()
def mock_driver() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def client(mock_driver: MagicMock) -> Neo4jClient:
    with patch(
        "graphrag_assistant.graph.neo4j_client.GraphDatabase.driver",
        return_value=mock_driver,
    ):
        return Neo4jClient("bolt://localhost:7687", "user", "test")


def _attach_session(mock_driver: MagicMock) -> MagicMock:
    """Wire a mock session onto the driver and return it."""
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return mock_session


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


def test_close_delegates_to_driver(client: Neo4jClient, mock_driver: MagicMock) -> None:
    client.close()
    assert mock_driver.close.call_count == 1


# ---------------------------------------------------------------------------
# run_query
# ---------------------------------------------------------------------------


def test_run_query_returns_list(client: Neo4jClient, mock_driver: MagicMock) -> None:
    mock_session = _attach_session(mock_driver)
    mock_session.run.return_value = []

    result = client.run_query("MATCH (n) RETURN n")

    assert isinstance(result, list)


def test_run_query_passes_parameters_to_session(
    client: Neo4jClient,
    mock_driver: MagicMock,
) -> None:
    mock_session = _attach_session(mock_driver)
    mock_session.run.return_value = []

    cypher = "MATCH (n {name: $name}) RETURN n"
    params = {"name": "ACME Corp"}
    client.run_query(cypher, params)

    assert mock_session.run.call_count == 1
    mock_session.run.assert_called_once_with(cypher, params)


def test_run_query_uses_empty_dict_when_no_params(
    client: Neo4jClient,
    mock_driver: MagicMock,
) -> None:
    mock_session = _attach_session(mock_driver)
    mock_session.run.return_value = []

    cypher = "MATCH (n) RETURN count(n)"
    client.run_query(cypher)

    assert mock_session.run.call_count == 1
    mock_session.run.assert_called_once_with(cypher, {})


# ---------------------------------------------------------------------------
# run_write
# ---------------------------------------------------------------------------


def test_run_write_calls_session_run(client: Neo4jClient, mock_driver: MagicMock) -> None:
    mock_session = _attach_session(mock_driver)

    cypher = "CREATE (n:Test {id: $id})"
    client.run_write(cypher, {"id": "t1"})

    assert mock_session.run.call_count == 1
    mock_session.run.assert_called_once_with(cypher, {"id": "t1"})


def test_run_write_uses_empty_dict_when_no_params(
    client: Neo4jClient,
    mock_driver: MagicMock,
) -> None:
    mock_session = _attach_session(mock_driver)

    cypher = "CREATE (n:Tag)"
    client.run_write(cypher)

    assert mock_session.run.call_count == 1
    mock_session.run.assert_called_once_with(cypher, {})


# ---------------------------------------------------------------------------
# bootstrap_schema / _run_ddl
# ---------------------------------------------------------------------------


def test_bootstrap_schema_swallows_equiv_schema_error(client: Neo4jClient) -> None:
    """Equivalent-schema errors must be silenced so bootstrap is idempotent."""
    exc = ClientError({"code": _EQUIV_SCHEMA_CODE, "message": "already exists"})
    completed = False
    with patch.object(client, "run_write", side_effect=exc):
        client.bootstrap_schema()
        completed = True
    assert completed


def test_bootstrap_silences_equivalent_schema_rule(client: Neo4jClient) -> None:
    exc = ClientError({"code": _EQUIV_SCHEMA_CODE, "message": "already exists"})
    completed = False
    with patch.object(client, "run_write", side_effect=exc):
        client.bootstrap_schema()
        completed = True
    assert completed


def test_bootstrap_silences_equivalent_schema_rule_still_closes_session(
    client: Neo4jClient,
    mock_driver: MagicMock,
) -> None:
    exc = ClientError({"code": _EQUIV_SCHEMA_CODE, "message": "already exists"})
    sessions_entered: list[MagicMock] = []

    original_session = mock_driver.session

    def _track_session() -> MagicMock:
        ctx = original_session()
        sessions_entered.append(ctx)
        return ctx

    mock_driver.session.side_effect = _track_session

    with patch.object(client, "run_write", side_effect=exc):
        client.bootstrap_schema()

    assert len(sessions_entered) == 0  # run_write is fully mocked; no real sessions opened


def test_bootstrap_creates_missing_index(client: Neo4jClient) -> None:
    calls: list[str] = []

    def _record(cypher: str, params: dict | None = None) -> None:
        calls.append(cypher)

    with patch.object(client, "run_write", side_effect=_record):
        client.bootstrap_schema()

    assert any("VECTOR INDEX" in c.upper() or "vector index" in c.lower() for c in calls)


def test_bootstrap_schema_reraises_other_client_errors(client: Neo4jClient) -> None:
    other_exc = ClientError({"code": "Neo.ClientError.Other.SomethingElse", "message": "boom"})
    with patch.object(client, "run_write", side_effect=other_exc):
        with pytest.raises(ClientError):
            client.bootstrap_schema()
