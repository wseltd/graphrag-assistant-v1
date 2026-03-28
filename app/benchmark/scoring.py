from __future__ import annotations

import re

from pydantic import BaseModel

STOP_WORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "is", "in", "of", "to", "and",
        "or", "for", "with", "that", "this", "it", "be",
        "as", "at", "by", "from",
    }
)


class ExpectedAnswer(BaseModel):
    query_id: str
    answer: str
    chunk_ids: list[str] = []

    def __repr__(self) -> str:
        return f"ExpectedAnswer(query_id={self.query_id!r}, chunk_ids={self.chunk_ids!r})"


class AnswerResult(BaseModel):
    run_id: str
    query_id: str
    answer: str
    text_citations: list[dict] = []
    latency_ms: float = 0.0
    mode: str = "graph_rag"

    def __repr__(self) -> str:
        return (
            f"AnswerResult(run_id={self.run_id!r}, query_id={self.query_id!r},"
            f" mode={self.mode!r})"
        )


class ScoredRun(BaseModel):
    run_id: str
    accuracy: float
    latency_ms: float

    def __repr__(self) -> str:
        return (
            f"ScoredRun(run_id={self.run_id!r}, accuracy={self.accuracy},"
            f" latency_ms={self.latency_ms})"
        )


def _tokenize(s: str) -> set[str]:
    cleaned = re.sub(r"[^\w\s]", "", s.lower())
    return {t for t in cleaned.split() if t not in STOP_WORDS}


def score_keyword_overlap(expected: str, actual: str) -> float:
    expected_tokens = _tokenize(expected)
    actual_tokens = _tokenize(actual)
    return len(expected_tokens & actual_tokens) / max(len(expected_tokens), 1)


def score_citation_coverage(
    expected_chunk_ids: list[str],
    text_citations: list[dict],
) -> float:
    expected_set = set(expected_chunk_ids)
    actual_set = {c["chunk_id"] for c in text_citations if "chunk_id" in c}
    return len(expected_set & actual_set) / max(len(expected_set), 1)


def score_query(expected: dict, result: dict, latency: float) -> dict:
    accuracy = score_keyword_overlap(
        expected.get("answer", ""),
        result.get("answer", ""),
    )
    citation_coverage = score_citation_coverage(
        expected.get("chunk_ids", []),
        result.get("text_citations", []),
    )
    return {
        "accuracy": accuracy,
        "citation_coverage": citation_coverage,
        "latency_seconds": latency,
    }
