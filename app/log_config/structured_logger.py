"""Request-scoped structured logger for the GraphRAG query pipeline.

Accumulates fields for one request and emits a single JSON line to stderr
(or a configured stream) when flush() is called. All methods are best-effort:
exceptions are caught and a WARNING line is written to stderr so logger
failures never disrupt responses.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import UTC, datetime
from typing import IO


def _warn(context: str, exc: Exception) -> None:
    """Write a single WARNING line to stderr without using the logging module."""
    try:
        sys.stderr.write(f"WARNING: StructuredLogger error in {context}: {exc}\n")
    except Exception:  # noqa: BLE001  # absolute last resort — stderr itself failed
        pass


class StructuredLogger:
    """Accumulates request lifecycle fields and emits one JSON line on flush().

    Usage::

        logger = StructuredLogger()
        logger.set_question("Who supplies Acme Corp?")
        logger.set_entities(["Acme Corp"])
        logger.record_stage("entity_resolution", 12.4)
        logger.record_stage("graph_traversal", 34.1)
        logger.set_mode("graph_rag")
        logger.set_answer_len(420)
        logger.flush()  # emits one JSON line to stderr

    flush() is idempotent: calling it a second time does nothing.
    record_stage() with a duplicate name appends ``_2`` to avoid silent overwrites.
    """

    def __init__(self, output: IO[str] | None = None) -> None:
        self._out: IO[str] = output if output is not None else sys.stderr
        self._flushed: bool = False
        self._request_id: str = str(uuid.uuid4())
        self._question: str = ""
        self._entities: list[str] = []
        self._cypher: str | None = None
        self._resolved_node_ids: list[str] = []
        self._chunk_ids: list[str] = []
        self._timings: dict[str, float] = {}
        self._mode: str | None = None
        self._answer_len: int = 0

    # ------------------------------------------------------------------
    # Setters — all best-effort, never raise
    # ------------------------------------------------------------------

    def set_question(self, question: str) -> None:
        try:
            self._question = str(question)
        except Exception as exc:
            _warn("set_question", exc)

    def set_entities(self, entities: list[str]) -> None:
        try:
            self._entities = list(entities)
        except Exception as exc:
            _warn("set_entities", exc)

    def set_cypher(self, cypher: str | None) -> None:
        try:
            self._cypher = str(cypher) if cypher is not None else None
        except Exception as exc:
            _warn("set_cypher", exc)

    def set_resolved_node_ids(self, node_ids: list[str]) -> None:
        try:
            self._resolved_node_ids = list(node_ids)
        except Exception as exc:
            _warn("set_resolved_node_ids", exc)

    def set_chunk_ids(self, chunk_ids: list[str]) -> None:
        try:
            self._chunk_ids = list(chunk_ids)
        except Exception as exc:
            _warn("set_chunk_ids", exc)

    def set_mode(self, mode: str) -> None:
        try:
            self._mode = str(mode)
        except Exception as exc:
            _warn("set_mode", exc)

    def set_answer_len(self, answer_len: int) -> None:
        try:
            self._answer_len = int(answer_len)
        except Exception as exc:
            _warn("set_answer_len", exc)

    # ------------------------------------------------------------------
    # Stage timing — appends _2 suffix on duplicate name
    # ------------------------------------------------------------------

    def record_stage(self, name: str, elapsed_ms: float) -> None:
        """Record elapsed time for a pipeline stage.

        If *name* has already been recorded, the second value is stored under
        ``{name}_2`` rather than silently overwriting the first.
        """
        try:
            key = f"{name}_2" if name in self._timings else name
            self._timings[key] = float(elapsed_ms)
        except Exception as exc:
            _warn("record_stage", exc)

    # ------------------------------------------------------------------
    # Flush — emits exactly one JSON line; idempotent; never raises
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Emit one JSON log line.  No-op if already called."""
        if self._flushed:
            return
        self._flushed = True
        try:
            record = {
                "ts": datetime.now(UTC).isoformat(),
                "request_id": self._request_id,
                "question": self._question,
                "entities": self._entities,
                "cypher": self._cypher,
                "resolved_node_ids": self._resolved_node_ids,
                "chunk_ids": self._chunk_ids,
                "timings": self._timings,
                "mode": self._mode,
                "answer_len": self._answer_len,
            }
            self._out.write(json.dumps(record) + "\n")
            self._out.flush()
        except Exception as exc:
            _warn("flush", exc)
