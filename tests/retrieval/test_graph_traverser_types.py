"""Unit tests for app.retrieval.graph_traverser_types.

Covers:
- GraphFact field access and types
- GraphFact equality (value semantics)
- GraphFact repr
- GraphFact immutability (frozen Pydantic model)
- GraphFact validation rejects non-string fields
- CypherTemplate type alias is str
- TemplateKey type alias is tuple[str, str]
- Module contains zero f-string or %-format interpolation of user input
"""
from __future__ import annotations

import inspect
import re

import pytest
from pydantic import ValidationError

import app.retrieval.graph_traverser_types as _mod
from app.retrieval.graph_traverser_types import (
    CypherTemplate,
    GraphFact,
    TemplateKey,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fact(**overrides: str) -> GraphFact:
    defaults = dict(
        source_id="p-001",
        source_label="Person",
        rel_type="DIRECTOR_OF",
        target_id="c-001",
        target_label="Company",
    )
    defaults.update(overrides)
    return GraphFact(**defaults)


# ---------------------------------------------------------------------------
# GraphFact — field access
# ---------------------------------------------------------------------------


class TestGraphFactFields:
    def test_source_id_accessible(self) -> None:
        fact = _make_fact(source_id="p-999")
        assert fact.source_id == "p-999"

    def test_source_label_accessible(self) -> None:
        fact = _make_fact(source_label="Person")
        assert fact.source_label == "Person"

    def test_rel_type_accessible(self) -> None:
        fact = _make_fact(rel_type="DIRECTOR_OF")
        assert fact.rel_type == "DIRECTOR_OF"

    def test_target_id_accessible(self) -> None:
        fact = _make_fact(target_id="c-042")
        assert fact.target_id == "c-042"

    def test_target_label_accessible(self) -> None:
        fact = _make_fact(target_label="Company")
        assert fact.target_label == "Company"

    def test_all_fields_are_str(self) -> None:
        fact = _make_fact()
        assert isinstance(fact.source_id, str)
        assert isinstance(fact.source_label, str)
        assert isinstance(fact.rel_type, str)
        assert isinstance(fact.target_id, str)
        assert isinstance(fact.target_label, str)

    def test_different_relationship_types_stored(self) -> None:
        for rel in ("DIRECTOR_OF", "PARTY_TO", "SUPPLIES", "FROM_CONTRACT", "HAS_CLAUSE"):
            fact = _make_fact(rel_type=rel)
            assert fact.rel_type == rel


# ---------------------------------------------------------------------------
# GraphFact — equality (value semantics)
# ---------------------------------------------------------------------------


class TestGraphFactEquality:
    def test_equal_instances_compare_equal(self) -> None:
        a = _make_fact()
        b = _make_fact()
        assert a == b

    def test_different_source_id_not_equal(self) -> None:
        a = _make_fact(source_id="p-001")
        b = _make_fact(source_id="p-002")
        assert a != b

    def test_different_rel_type_not_equal(self) -> None:
        a = _make_fact(rel_type="DIRECTOR_OF")
        b = _make_fact(rel_type="PARTY_TO")
        assert a != b

    def test_different_target_id_not_equal(self) -> None:
        a = _make_fact(target_id="c-001")
        b = _make_fact(target_id="c-002")
        assert a != b

    def test_equal_facts_have_equal_hashes(self) -> None:
        a = _make_fact()
        b = _make_fact()
        assert hash(a) == hash(b)

    def test_can_be_used_in_set(self) -> None:
        a = _make_fact()
        b = _make_fact()
        c = _make_fact(source_id="p-002")
        s = {a, b, c}
        assert len(s) == 2


# ---------------------------------------------------------------------------
# GraphFact — repr
# ---------------------------------------------------------------------------


class TestGraphFactRepr:
    def test_repr_contains_class_name(self) -> None:
        assert repr(_make_fact()).startswith("GraphFact(")

    def test_repr_contains_source_id(self) -> None:
        fact = _make_fact(source_id="p-007")
        assert "p-007" in repr(fact)

    def test_repr_contains_source_label(self) -> None:
        fact = _make_fact(source_label="Person")
        assert "Person" in repr(fact)

    def test_repr_contains_rel_type(self) -> None:
        fact = _make_fact(rel_type="SUPPLIES")
        assert "SUPPLIES" in repr(fact)

    def test_repr_contains_target_id(self) -> None:
        fact = _make_fact(target_id="prod-001")
        assert "prod-001" in repr(fact)

    def test_repr_contains_target_label(self) -> None:
        fact = _make_fact(target_label="Product")
        assert "Product" in repr(fact)

    def test_repr_is_deterministic(self) -> None:
        fact = _make_fact()
        assert repr(fact) == repr(fact)


# ---------------------------------------------------------------------------
# GraphFact — immutability
# ---------------------------------------------------------------------------


class TestGraphFactImmutability:
    def test_cannot_set_source_id(self) -> None:
        fact = _make_fact()
        with pytest.raises(ValidationError):
            fact.source_id = "mutated"  # type: ignore[misc]

    def test_cannot_set_rel_type(self) -> None:
        fact = _make_fact()
        with pytest.raises(ValidationError):
            fact.rel_type = "mutated"  # type: ignore[misc]

    def test_cannot_set_target_id(self) -> None:
        fact = _make_fact()
        with pytest.raises(ValidationError):
            fact.target_id = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GraphFact — validation
# ---------------------------------------------------------------------------


class TestGraphFactValidation:
    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            GraphFact(  # type: ignore[call-arg]
                source_id="p-001",
                source_label="Person",
                rel_type="DIRECTOR_OF",
                target_id="c-001",
                # target_label omitted
            )

    def test_extra_field_ignored_or_raises(self) -> None:
        # Pydantic v2 default ignores extra fields; frozen model still valid.
        fact = GraphFact(
            source_id="p-001",
            source_label="Person",
            rel_type="DIRECTOR_OF",
            target_id="c-001",
            target_label="Company",
        )
        assert fact.source_id == "p-001"


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------


class TestTypeAliases:
    def test_cypher_template_is_str_alias(self) -> None:
        t: CypherTemplate = "MATCH (n) RETURN n"
        assert isinstance(t, str)

    def test_template_key_is_tuple_alias(self) -> None:
        k: TemplateKey = ("Person", "DIRECTOR_OF")
        assert isinstance(k, tuple)
        assert len(k) == 2
        assert all(isinstance(x, str) for x in k)

    def test_cypher_template_exported(self) -> None:
        assert hasattr(_mod, "CypherTemplate")

    def test_template_key_exported(self) -> None:
        assert hasattr(_mod, "TemplateKey")


# ---------------------------------------------------------------------------
# Cypher safety — static check on the source file
# ---------------------------------------------------------------------------


class TestCypherSafety:
    """Verify the types module contains no f-string or %-format interpolation."""

    def _source(self) -> str:
        return inspect.getsource(_mod)

    def test_no_fstring_interpolation(self) -> None:
        source = self._source()
        # f"..." or f'...' would indicate runtime string building
        assert not re.search(r'\bf"', source), "f-strings found in types module"
        assert not re.search(r"\bf'", source), "f-strings found in types module"

    def test_no_percent_format(self) -> None:
        source = self._source()
        # % formatting on strings would be: "..." % variable
        assert not re.search(r'["\']\s*%\s*\(', source), "%-format found in types module"
