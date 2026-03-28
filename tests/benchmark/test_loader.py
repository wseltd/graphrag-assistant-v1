from __future__ import annotations

import json

import pytest

from app.benchmark.loader import load_benchmark_data, load_jsonl


def _write_jsonl(path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _make_queries(n: int = 2) -> list[dict]:
    return [{"query_id": f"q{i}", "query": f"question {i}"} for i in range(n)]


def _make_answers(n: int = 2) -> list[dict]:
    return [
        {"query_id": f"q{i}", "answer": f"answer {i}", "chunk_ids": [f"c{i}"]}
        for i in range(n)
    ]


# --- load_jsonl ---


def test_load_jsonl_returns_list_of_dicts(tmp_path):
    p = tmp_path / "data.jsonl"
    _write_jsonl(p, [{"a": 1}, {"b": 2}])
    result = load_jsonl(str(p))
    assert result == [{"a": 1}, {"b": 2}]


def test_load_jsonl_empty_file_returns_empty_list(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert load_jsonl(str(p)) == []


def test_load_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "blanks.jsonl"
    p.write_text('{"x": 1}\n\n{"x": 2}\n', encoding="utf-8")
    assert load_jsonl(str(p)) == [{"x": 1}, {"x": 2}]


def test_load_jsonl_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_jsonl("/nonexistent/path/file.jsonl")


def test_load_jsonl_raises_on_invalid_json(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("not json\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_jsonl(str(p))


# --- load_benchmark_data ---


def test_load_benchmark_data_returns_queries_and_answers(tmp_path):
    qp = tmp_path / "q.jsonl"
    ap = tmp_path / "a.jsonl"
    _write_jsonl(qp, _make_queries(3))
    _write_jsonl(ap, _make_answers(3))
    queries, answers = load_benchmark_data(str(qp), str(ap))
    assert len(queries) == 3
    assert len(answers) == 3


def test_load_benchmark_data_length_mismatch_raises(tmp_path):
    qp = tmp_path / "q.jsonl"
    ap = tmp_path / "a.jsonl"
    _write_jsonl(qp, _make_queries(2))
    _write_jsonl(ap, _make_answers(3))
    with pytest.raises(ValueError, match="length mismatch"):
        load_benchmark_data(str(qp), str(ap))


def test_load_benchmark_data_missing_query_key_raises(tmp_path):
    qp = tmp_path / "q.jsonl"
    ap = tmp_path / "a.jsonl"
    _write_jsonl(qp, [{"query_id": "q0"}])  # missing 'query'
    _write_jsonl(ap, _make_answers(1))
    with pytest.raises(ValueError, match="missing keys"):
        load_benchmark_data(str(qp), str(ap))


def test_load_benchmark_data_missing_answer_key_raises(tmp_path):
    qp = tmp_path / "q.jsonl"
    ap = tmp_path / "a.jsonl"
    _write_jsonl(qp, _make_queries(1))
    _write_jsonl(ap, [{"query_id": "q0", "answer": "x"}])  # missing 'chunk_ids'
    with pytest.raises(ValueError, match="missing keys"):
        load_benchmark_data(str(qp), str(ap))


def test_load_benchmark_data_missing_query_id_in_query_raises(tmp_path):
    qp = tmp_path / "q.jsonl"
    ap = tmp_path / "a.jsonl"
    _write_jsonl(qp, [{"query": "what?"}])  # missing 'query_id'
    _write_jsonl(ap, _make_answers(1))
    with pytest.raises(ValueError, match="missing keys"):
        load_benchmark_data(str(qp), str(ap))


def test_load_benchmark_data_missing_query_id_in_answer_raises(tmp_path):
    qp = tmp_path / "q.jsonl"
    ap = tmp_path / "a.jsonl"
    _write_jsonl(qp, _make_queries(1))
    _write_jsonl(ap, [{"answer": "x", "chunk_ids": []}])  # missing 'query_id'
    with pytest.raises(ValueError, match="missing keys"):
        load_benchmark_data(str(qp), str(ap))


def test_load_benchmark_data_single_entry(tmp_path):
    qp = tmp_path / "q.jsonl"
    ap = tmp_path / "a.jsonl"
    _write_jsonl(qp, _make_queries(1))
    _write_jsonl(ap, _make_answers(1))
    queries, answers = load_benchmark_data(str(qp), str(ap))
    assert queries[0]["query_id"] == "q0"
    assert answers[0]["chunk_ids"] == ["c0"]


def test_load_benchmark_data_preserves_extra_fields(tmp_path):
    qp = tmp_path / "q.jsonl"
    ap = tmp_path / "a.jsonl"
    _write_jsonl(qp, [{"query_id": "q0", "query": "x", "extra": 42}])
    _write_jsonl(ap, [{"query_id": "q0", "answer": "y", "chunk_ids": [], "note": "ok"}])
    queries, answers = load_benchmark_data(str(qp), str(ap))
    assert queries[0]["extra"] == 42
    assert answers[0]["note"] == "ok"
