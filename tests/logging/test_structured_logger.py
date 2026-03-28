"""Unit tests for StructuredLogger (T035)."""
from __future__ import annotations

import json
from datetime import datetime
from io import StringIO

from app.log_config.structured_logger import StructuredLogger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logger() -> tuple[StructuredLogger, StringIO]:
    buf = StringIO()
    return StructuredLogger(output=buf), buf


def _flush_and_parse(logger: StructuredLogger, buf: StringIO) -> dict:
    logger.flush()
    line = buf.getvalue()
    assert line.endswith("\n"), "emitted line must end with newline"
    return json.loads(line.strip())


# ---------------------------------------------------------------------------
# Test 1 — normal flush emits all required fields
# ---------------------------------------------------------------------------


def test_normal_flush_emits_all_required_fields() -> None:
    logger, buf = _make_logger()

    logger.set_question("Who supplies Acme Corp?")
    logger.set_entities(["Acme Corp"])
    logger.set_cypher("MATCH (n) RETURN n")
    logger.set_resolved_node_ids(["company-1", "company-2"])
    logger.set_chunk_ids(["chunk-a", "chunk-b"])
    logger.record_stage("entity_resolution", 12.5)
    logger.record_stage("graph_traversal", 34.1)
    logger.set_mode("graph_rag")
    logger.set_answer_len(250)

    record = _flush_and_parse(logger, buf)

    assert record["question"] == "Who supplies Acme Corp?"
    assert record["entities"] == ["Acme Corp"]
    assert record["cypher"] == "MATCH (n) RETURN n"
    assert record["resolved_node_ids"] == ["company-1", "company-2"]
    assert record["chunk_ids"] == ["chunk-a", "chunk-b"]
    assert record["timings"] == {"entity_resolution": 12.5, "graph_traversal": 34.1}
    assert record["mode"] == "graph_rag"
    assert record["answer_len"] == 250
    assert "ts" in record
    assert "request_id" in record
    # ts must be ISO-8601 with timezone offset
    parsed_ts = datetime.fromisoformat(record["ts"])
    assert parsed_ts.tzinfo is not None


# ---------------------------------------------------------------------------
# Test 2 — flush with missing optional fields uses safe defaults
# ---------------------------------------------------------------------------


def test_flush_with_missing_optional_fields_uses_defaults() -> None:
    logger, buf = _make_logger()
    # Only set question; leave everything else at defaults.
    logger.set_question("bare minimum")

    record = _flush_and_parse(logger, buf)

    assert record["question"] == "bare minimum"
    assert record["cypher"] is None
    assert record["entities"] == []
    assert record["resolved_node_ids"] == []
    assert record["chunk_ids"] == []
    assert record["timings"] == {}
    assert record["mode"] is None
    assert record["answer_len"] == 0
    # All ten required keys must be present even when most are unset.
    required_keys = {
        "ts", "request_id", "question", "entities", "cypher",
        "resolved_node_ids", "chunk_ids", "timings", "mode", "answer_len",
    }
    assert required_keys.issubset(record.keys())


# ---------------------------------------------------------------------------
# Test 3 — flush error does not propagate to caller
# ---------------------------------------------------------------------------


def test_flush_error_does_not_propagate() -> None:
    class RaisingStream:
        def write(self, s: str) -> None:
            raise RuntimeError("simulated disk-full error")

        def flush(self) -> None:
            pass

    logger = StructuredLogger(output=RaisingStream())  # type: ignore[arg-type]
    logger.set_question("will fail silently")
    logger.record_stage("entity_resolution", 5.0)

    # Must not raise — best-effort contract.
    logger.flush()

    # _flushed is set True even when the write fails, preserving idempotency.
    assert logger._flushed is True


# ---------------------------------------------------------------------------
# Test 4 — duplicate stage name gets _2 suffix; flush is idempotent
# ---------------------------------------------------------------------------


def test_duplicate_stage_name_gets_suffix_and_flush_is_idempotent() -> None:
    logger, buf = _make_logger()

    logger.record_stage("retrieve", 10.0)
    logger.record_stage("retrieve", 20.0)  # duplicate → stored as retrieve_2

    logger.flush()  # first flush — emits one line
    logger.flush()  # second flush — no-op

    output = buf.getvalue()
    lines = [ln for ln in output.splitlines() if ln.strip()]
    assert len(lines) == 1, "flush() must be idempotent — only one line emitted"

    record = json.loads(lines[0])
    assert record["timings"]["retrieve"] == 10.0
    assert record["timings"]["retrieve_2"] == 20.0
