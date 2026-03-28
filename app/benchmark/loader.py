from __future__ import annotations

import json

_QUERY_KEYS: frozenset[str] = frozenset({"query_id", "query"})
_ANSWER_KEYS: frozenset[str] = frozenset({"answer", "chunk_ids", "query_id"})


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_benchmark_data(
    queries_path: str = "data/processed/benchmark_queries.jsonl",
    answers_path: str = "data/expected/benchmark_answers.jsonl",
) -> tuple[list[dict], list[dict]]:
    queries = load_jsonl(queries_path)
    answers = load_jsonl(answers_path)

    if len(queries) != len(answers):
        raise ValueError(
            f"queries and answers length mismatch: {len(queries)} vs {len(answers)}"
        )

    for i, q in enumerate(queries):
        missing = _QUERY_KEYS - q.keys()
        if missing:
            raise ValueError(f"query entry {i} missing keys: {missing}")

    for i, a in enumerate(answers):
        missing = _ANSWER_KEYS - a.keys()
        if missing:
            raise ValueError(f"answer entry {i} missing keys: {missing}")

    return queries, answers
