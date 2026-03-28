"""Unit tests for seed_benchmark.py.

6 tests total — no integration tests, all data is static in-memory.

Tests:
  1. test_query_schema_completeness      — every query has required fields with correct types
  2. test_query_category_distribution    — 10 multi_hop_graph, 5 graph_plus_text, 5 distractor
  3. test_answer_schema_completeness     — every answer has required fields with correct types
  4. test_answer_evidence_by_category    — multi_hop_graph/graph_plus_text have non-empty evidence
  5. test_required_entity_ids_exist      — every required_entity_id maps to a known seed entity
  6. test_required_chunk_ids_exist       — every required_chunk_id maps to a known chunk pair
"""

from __future__ import annotations

from scripts.seed.seed_benchmark import ANSWERS, QUERIES

# ---------------------------------------------------------------------------
# In-memory seed fixtures — avoids reading files, derived from T010 CSVs
# ---------------------------------------------------------------------------

_SEED_ENTITY_IDS: frozenset[str] = frozenset(
    [
        # companies.csv
        "C001", "C002", "C003", "C004", "C005", "C006", "C007", "C008",
        "C009", "C010", "C011", "C012", "C013", "C014", "C015",
        # people.csv
        "P001", "P002", "P003", "P004", "P005", "P006", "P007", "P008",
        "P009", "P010", "P011", "P012", "P013", "P014", "P015",
        # addresses.csv
        "A001", "A002", "A003", "A004", "A005", "A006", "A007", "A008",
        "A009", "A010", "A011", "A012", "A013", "A014", "A015",
        # products.csv
        "PR001", "PR002", "PR003", "PR004", "PR005", "PR006", "PR007",
        "PR008", "PR009", "PR010", "PR011", "PR012", "PR013", "PR014",
        "PR015",
    ]
)

# (doc_id, chunk_id) pairs present in data/processed/chunks.jsonl, derived
# by running chunk_contracts.py over K001–K006 with WINDOW=400 OVERLAP=80.
# K001: 33 chunks (0–32), K002: 29 (0–28), K003: 26 (0–25),
# K004: 33 (0–32), K005: 33 (0–32), K006: 29 (0–28).
_SEED_CHUNKS: frozenset[tuple[str, int]] = frozenset(
    [("K001", i) for i in range(33)]
    + [("K002", i) for i in range(29)]
    + [("K003", i) for i in range(26)]
    + [("K004", i) for i in range(33)]
    + [("K005", i) for i in range(33)]
    + [("K006", i) for i in range(29)]
)

_VALID_CATEGORIES: frozenset[str] = frozenset(
    {"multi_hop_graph", "graph_plus_text", "distractor"}
)

_QUERY_REQUIRED_FIELDS: dict[str, type] = {
    "query_id": str,
    "question": str,
    "category": str,
    "required_entity_ids": list,
    "required_chunk_ids": list,
}

_ANSWER_REQUIRED_FIELDS: dict[str, type] = {
    "query_id": str,
    "answer": str,
    "graph_evidence": list,
    "text_citations": list,
    "mode": str,
}


# ---------------------------------------------------------------------------
# Test 1 — query schema completeness
# ---------------------------------------------------------------------------


def test_query_schema_completeness() -> None:
    """Every query record has all required fields with the correct Python types."""
    assert len(QUERIES) == 20, f"Expected 20 queries, got {len(QUERIES)}"
    for q in QUERIES:
        for field, expected_type in _QUERY_REQUIRED_FIELDS.items():
            assert field in q, f"Query {q.get('query_id', '?')!r} missing field {field!r}"
            assert isinstance(q[field], expected_type), (
                f"Query {q['query_id']!r} field {field!r}: "
                f"expected {expected_type.__name__}, got {type(q[field]).__name__}"
            )
        assert q["category"] in _VALID_CATEGORIES, (
            f"Query {q['query_id']!r} has invalid category {q['category']!r}"
        )
        assert q["query_id"].startswith("BQ"), (
            f"query_id {q['query_id']!r} does not start with 'BQ'"
        )
        assert q["question"].strip(), (
            f"Query {q['query_id']!r} has an empty question"
        )
        for eid in q["required_entity_ids"]:
            assert isinstance(eid, str), (
                f"Query {q['query_id']!r}: required_entity_ids must contain strings"
            )
        for cid in q["required_chunk_ids"]:
            assert isinstance(cid, str), (
                f"Query {q['query_id']!r}: required_chunk_ids must contain strings"
            )


# ---------------------------------------------------------------------------
# Test 2 — category distribution
# ---------------------------------------------------------------------------


def test_query_category_distribution() -> None:
    """Distribution must be exactly 10 multi_hop_graph, 5 graph_plus_text, 5 distractor."""
    from collections import Counter

    counts = Counter(q["category"] for q in QUERIES)
    assert counts["multi_hop_graph"] == 10, (
        f"Expected 10 multi_hop_graph, got {counts['multi_hop_graph']}"
    )
    assert counts["graph_plus_text"] == 5, (
        f"Expected 5 graph_plus_text, got {counts['graph_plus_text']}"
    )
    assert counts["distractor"] == 5, (
        f"Expected 5 distractor, got {counts['distractor']}"
    )
    query_ids = [q["query_id"] for q in QUERIES]
    assert len(set(query_ids)) == 20, "Duplicate query_ids detected in QUERIES"


# ---------------------------------------------------------------------------
# Test 3 — answer schema completeness
# ---------------------------------------------------------------------------


def test_answer_schema_completeness() -> None:
    """Every answer record has all required fields with the correct Python types."""
    assert len(ANSWERS) == 20, f"Expected 20 answers, got {len(ANSWERS)}"
    query_ids = {q["query_id"] for q in QUERIES}
    for a in ANSWERS:
        for field, expected_type in _ANSWER_REQUIRED_FIELDS.items():
            assert field in a, f"Answer {a.get('query_id', '?')!r} missing field {field!r}"
            assert isinstance(a[field], expected_type), (
                f"Answer {a['query_id']!r} field {field!r}: "
                f"expected {expected_type.__name__}, got {type(a[field]).__name__}"
            )
        assert a["query_id"] in query_ids, (
            f"Answer query_id {a['query_id']!r} has no matching query"
        )
        assert a["answer"].strip(), f"Answer {a['query_id']!r} has an empty answer string"
        assert a["mode"] in ("plain_rag", "graph_rag"), (
            f"Answer {a['query_id']!r} has invalid mode {a['mode']!r}"
        )
        for ge in a["graph_evidence"]:
            assert {"source_id", "target_id", "label"} == set(ge.keys()), (
                f"Answer {a['query_id']!r} graph_evidence entry missing keys: {ge}"
            )
        for tc in a["text_citations"]:
            assert {"doc_id", "chunk_id", "quote"} == set(tc.keys()), (
                f"Answer {a['query_id']!r} text_citations entry missing keys: {tc}"
            )
    answer_ids = [a["query_id"] for a in ANSWERS]
    assert len(set(answer_ids)) == 20, "Duplicate query_ids detected in ANSWERS"


# ---------------------------------------------------------------------------
# Test 4 — evidence presence by category
# ---------------------------------------------------------------------------


def test_answer_evidence_by_category() -> None:
    """multi_hop_graph answers must have non-empty graph_evidence.
    graph_plus_text answers must have non-empty graph_evidence AND text_citations.
    distractor answers may have empty evidence (no match expected).
    """
    category_by_id = {q["query_id"]: q["category"] for q in QUERIES}
    for a in ANSWERS:
        cat = category_by_id[a["query_id"]]
        if cat == "multi_hop_graph":
            assert a["graph_evidence"], (
                f"Answer {a['query_id']!r} (multi_hop_graph) must have non-empty "
                f"graph_evidence"
            )
        elif cat == "graph_plus_text":
            assert a["graph_evidence"], (
                f"Answer {a['query_id']!r} (graph_plus_text) must have non-empty "
                f"graph_evidence"
            )
            assert a["text_citations"], (
                f"Answer {a['query_id']!r} (graph_plus_text) must have non-empty "
                f"text_citations"
            )


# ---------------------------------------------------------------------------
# Test 5 — required_entity_ids cross-reference
# ---------------------------------------------------------------------------


def test_required_entity_ids_exist() -> None:
    """Every required_entity_id in QUERIES appears in the in-memory seed entity fixture."""
    for q in QUERIES:
        for eid in q["required_entity_ids"]:
            assert eid in _SEED_ENTITY_IDS, (
                f"Query {q['query_id']!r}: entity_id {eid!r} not found in seed data "
                f"(companies, people, addresses, products)"
            )


# ---------------------------------------------------------------------------
# Test 6 — required_chunk_ids cross-reference
# ---------------------------------------------------------------------------


def test_required_chunk_ids_exist() -> None:
    """Every required_chunk_id '<doc_id>:<chunk_id>' in QUERIES maps to a record
    in the in-memory chunks fixture derived from chunks.jsonl.
    """
    for q in QUERIES:
        for chunk_ref in q["required_chunk_ids"]:
            assert ":" in chunk_ref, (
                f"Query {q['query_id']!r}: chunk_ref {chunk_ref!r} must use "
                f"'<doc_id>:<chunk_id>' format"
            )
            doc_id, raw_cid = chunk_ref.rsplit(":", 1)
            assert raw_cid.isdigit(), (
                f"Query {q['query_id']!r}: chunk_ref {chunk_ref!r} — chunk_id part "
                f"must be a non-negative integer"
            )
            pair = (doc_id, int(raw_cid))
            assert pair in _SEED_CHUNKS, (
                f"Query {q['query_id']!r}: chunk ({doc_id!r}, {int(raw_cid)}) not "
                f"found in seed chunks"
            )
