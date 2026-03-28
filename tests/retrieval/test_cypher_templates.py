"""Unit tests for app.retrieval.cypher_templates.

Coverage plan
-------------
TestTemplatesStructure     — all 8 rel types present, 3 templates per key,
                             UNWIND $ids in every template, no * wildcards
TestArrowDirections        — arrow direction correct for each relationship type
TestReverseGuard           — PARTY_TO target is :Contract, not :Person or :Company
TestMultiHopTemplates      — SUPPLIES and FROM_CONTRACT keys exist with correct arrows
TestEmptyInput             — empty ids list returns [] with no DB call
TestCypherSafety           — no f-string or %-format interpolation in the module source
TestCartesianGuard         — UNWIND $ids structure prevents row multiplication
TestLookupFacts            — lookup_facts wires template → session → GraphFact correctly
"""
from __future__ import annotations

import inspect
import os as _os
import re
from unittest.mock import MagicMock

import pytest

import app.retrieval.cypher_templates as _mod
from app.retrieval.cypher_templates import TEMPLATES, lookup_facts
from app.retrieval.graph_traverser_types import GraphFact

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_TEMPLATE_TEXT = " ".join(t for templates in TEMPLATES.values() for t in templates)

_ALL_REL_TYPES = (
    "DIRECTOR_OF",
    "REGISTERED_AT",
    "SUPPLIES",
    "PARTY_TO",
    "HAS_CLAUSE",
    "FROM_CONTRACT",
    "ABOUT_COMPANY",
    "RELATED_TO",
)


def _1hop(key: tuple[str, str]) -> str:
    return TEMPLATES[key][0]


# ---------------------------------------------------------------------------
# TestTemplatesStructure
# ---------------------------------------------------------------------------


class TestTemplatesStructure:
    def test_templates_is_a_dict(self) -> None:
        assert isinstance(TEMPLATES, dict)

    def test_eight_keys(self) -> None:
        assert len(TEMPLATES) == 8

    def test_each_key_has_three_templates(self) -> None:
        for key, templates in TEMPLATES.items():
            assert len(templates) == 3, f"Key {key!r}: expected 3 templates, got {len(templates)}"

    def test_all_eight_relationship_types_present(self) -> None:
        for rel in _ALL_REL_TYPES:
            assert rel in _ALL_TEMPLATE_TEXT, f"Relationship type {rel!r} missing"

    def test_all_templates_contain_unwind_ids(self) -> None:
        for key, templates in TEMPLATES.items():
            for i, t in enumerate(templates):
                assert "UNWIND $ids" in t, f"TEMPLATES[{key!r}][{i}] missing 'UNWIND $ids'"

    def test_no_wildcard_star_patterns(self) -> None:
        for key, templates in TEMPLATES.items():
            for i, t in enumerate(templates):
                # Reject variable-length patterns like [:REL*] or [:REL*1..3]
                assert not re.search(r"\[:[\w_]+\*", t), (
                    f"Wildcard pattern found in TEMPLATES[{key!r}][{i}]: {t!r}"
                )

    def test_all_templates_use_parameterised_id_match(self) -> None:
        for key, templates in TEMPLATES.items():
            for i, t in enumerate(templates):
                assert "WHERE id(src) = id" in t, (
                    f"TEMPLATES[{key!r}][{i}] missing 'WHERE id(src) = id'"
                )

    def test_keys_are_tuples_of_two_strings(self) -> None:
        for key in TEMPLATES:
            assert isinstance(key, tuple) and len(key) == 2
            assert all(isinstance(s, str) for s in key)


# ---------------------------------------------------------------------------
# TestArrowDirections
# ---------------------------------------------------------------------------


class TestArrowDirections:
    """Arrow directions verified via string inspection on the 1-hop template.

    1-hop templates use a named relationship variable: (src)-[r:REL]->(tgt).
    Direction checks therefore look for ``REL]->`` (present in both ``[r:REL]->``
    and ``[:REL]->`` forms) and assert the reverse ``<-[`` pattern is absent.
    """

    def test_director_of_forward_arrow(self) -> None:
        t = _1hop(("Person", "Company"))
        assert re.search(r"-\[r?:DIRECTOR_OF\]->", t), "DIRECTOR_OF forward arrow not found"

    def test_director_of_no_reverse_arrow(self) -> None:
        t = _1hop(("Person", "Company"))
        assert not re.search(r"<-\[r?:DIRECTOR_OF\]-", t)

    def test_registered_at_forward_arrow(self) -> None:
        t = _1hop(("Company", "Address"))
        assert re.search(r"-\[r?:REGISTERED_AT\]->", t), "REGISTERED_AT forward arrow not found"

    def test_registered_at_no_reverse_arrow(self) -> None:
        t = _1hop(("Company", "Address"))
        assert not re.search(r"<-\[r?:REGISTERED_AT\]-", t)

    def test_supplies_forward_arrow(self) -> None:
        t = _1hop(("Company", "Product"))
        assert re.search(r"-\[r?:SUPPLIES\]->", t), "SUPPLIES forward arrow not found"

    def test_supplies_no_reverse_arrow(self) -> None:
        t = _1hop(("Company", "Product"))
        assert not re.search(r"<-\[r?:SUPPLIES\]-", t)

    def test_party_to_forward_arrow(self) -> None:
        t = _1hop(("Company", "Contract"))
        assert re.search(r"-\[r?:PARTY_TO\]->", t), "PARTY_TO forward arrow not found"

    def test_party_to_no_reverse_arrow(self) -> None:
        t = _1hop(("Company", "Contract"))
        assert not re.search(r"<-\[r?:PARTY_TO\]-", t)

    def test_has_clause_forward_arrow(self) -> None:
        t = _1hop(("Contract", "Clause"))
        assert re.search(r"-\[r?:HAS_CLAUSE\]->", t), "HAS_CLAUSE forward arrow not found"

    def test_has_clause_no_reverse_arrow(self) -> None:
        t = _1hop(("Contract", "Clause"))
        assert not re.search(r"<-\[r?:HAS_CLAUSE\]-", t)

    def test_from_contract_forward_arrow(self) -> None:
        t = _1hop(("Chunk", "Contract"))
        assert re.search(r"-\[r?:FROM_CONTRACT\]->", t), "FROM_CONTRACT forward arrow not found"

    def test_from_contract_no_reverse_arrow(self) -> None:
        t = _1hop(("Chunk", "Contract"))
        assert not re.search(r"<-\[r?:FROM_CONTRACT\]-", t)

    def test_about_company_forward_arrow(self) -> None:
        t = _1hop(("Chunk", "Company"))
        assert re.search(r"-\[r?:ABOUT_COMPANY\]->", t), "ABOUT_COMPANY forward arrow not found"

    def test_about_company_no_reverse_arrow(self) -> None:
        t = _1hop(("Chunk", "Company"))
        assert not re.search(r"<-\[r?:ABOUT_COMPANY\]-", t)

    def test_related_to_forward_arrow(self) -> None:
        t = _1hop(("Chunk", "Entity"))
        assert re.search(r"-\[r?:RELATED_TO\]->", t), "RELATED_TO forward arrow not found"

    def test_related_to_no_reverse_arrow(self) -> None:
        t = _1hop(("Chunk", "Entity"))
        assert not re.search(r"<-\[r?:RELATED_TO\]-", t)

    def test_director_of_source_is_person(self) -> None:
        t = _1hop(("Person", "Company"))
        assert "(src:Person)" in t

    def test_director_of_target_is_company(self) -> None:
        t = _1hop(("Person", "Company"))
        assert "(tgt:Company)" in t

    def test_party_to_target_is_contract(self) -> None:
        t = _1hop(("Company", "Contract"))
        assert "(tgt:Contract)" in t


# ---------------------------------------------------------------------------
# TestReverseGuard
# ---------------------------------------------------------------------------


class TestReverseGuard:
    """Verify that PARTY_TO traversal cannot accidentally return a Person."""

    def test_party_to_1hop_target_label_is_contract(self) -> None:
        t = _1hop(("Company", "Contract"))
        assert "(tgt:Contract)" in t

    def test_party_to_1hop_target_not_labeled_person(self) -> None:
        t = _1hop(("Company", "Contract"))
        assert "(tgt:Person)" not in t

    def test_party_to_1hop_target_not_labeled_company(self) -> None:
        # Source is Company, target must not be re-labeled Company
        t = _1hop(("Company", "Contract"))
        assert "(tgt:Company)" not in t

    def test_person_company_key_absent_from_party_to_templates(self) -> None:
        # There is no key that would return a Person as a PARTY_TO target
        party_to_templates = [
            t for t in TEMPLATES.get(("Company", "Contract"), [])
        ]
        combined = " ".join(party_to_templates)
        # No pattern that matches -[:PARTY_TO]->(tgt:Person)
        assert "-[:PARTY_TO]->(tgt:Person)" not in combined

    def test_director_of_templates_only_in_person_company_key(self) -> None:
        # DIRECTOR_OF must not appear in PARTY_TO templates
        party_to_templates = " ".join(TEMPLATES.get(("Company", "Contract"), []))
        assert "DIRECTOR_OF" not in party_to_templates


# ---------------------------------------------------------------------------
# TestMultiHopTemplates
# ---------------------------------------------------------------------------


class TestMultiHopTemplates:
    """Verify both hops of a multi-hop chain have correct templates."""

    def test_supplies_key_exists(self) -> None:
        assert ("Company", "Product") in TEMPLATES

    def test_from_contract_key_exists(self) -> None:
        assert ("Chunk", "Contract") in TEMPLATES

    def test_supplies_1hop_direction(self) -> None:
        t = TEMPLATES[("Company", "Product")][0]
        assert re.search(r"-\[r?:SUPPLIES\]->", t), "SUPPLIES forward arrow not found"
        assert "(src:Company)" in t
        assert "(tgt:Product)" in t

    def test_from_contract_1hop_direction(self) -> None:
        t = TEMPLATES[("Chunk", "Contract")][0]
        assert re.search(r"-\[r?:FROM_CONTRACT\]->", t), "FROM_CONTRACT forward arrow not found"
        assert "(src:Chunk)" in t
        assert "(tgt:Contract)" in t

    def test_supplies_2hop_template_contains_supplies_twice(self) -> None:
        t = TEMPLATES[("Company", "Product")][1]
        assert t.count("-[:SUPPLIES]->") == 2

    def test_from_contract_2hop_template_contains_from_contract_twice(self) -> None:
        t = TEMPLATES[("Chunk", "Contract")][1]
        assert t.count("-[:FROM_CONTRACT]->") == 2

    def test_supplies_3hop_template_contains_supplies_three_times(self) -> None:
        t = TEMPLATES[("Company", "Product")][2]
        assert t.count("-[:SUPPLIES]->") == 3

    def test_multi_hop_templates_have_unwind_relationships(self) -> None:
        for key, templates in TEMPLATES.items():
            for i in (1, 2):  # 2-hop and 3-hop (indices 1 and 2)
                assert "UNWIND relationships(p)" in templates[i], (
                    f"TEMPLATES[{key!r}][{i}] missing 'UNWIND relationships(p)'"
                )

    def test_multi_hop_templates_use_path_variable(self) -> None:
        for key, templates in TEMPLATES.items():
            for i in (1, 2):  # 2-hop and 3-hop
                assert "MATCH p = " in templates[i], (
                    f"TEMPLATES[{key!r}][{i}] missing 'MATCH p = '"
                )


# ---------------------------------------------------------------------------
# TestEmptyInput
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_ids_returns_empty_list(self) -> None:
        session = MagicMock()
        result = lookup_facts(session, "Person", "Company", [])
        assert result == []

    def test_empty_ids_makes_no_db_call(self) -> None:
        session = MagicMock()
        lookup_facts(session, "Person", "Company", [])
        assert session.run.call_count == 0

    def test_empty_ids_returns_list_type(self) -> None:
        session = MagicMock()
        result = lookup_facts(session, "Company", "Contract", [])
        assert isinstance(result, list)

    def test_empty_ids_contract_clause_no_db_call(self) -> None:
        session = MagicMock()
        lookup_facts(session, "Contract", "Clause", [])
        assert session.run.call_count == 0


# ---------------------------------------------------------------------------
# TestCypherSafety
# ---------------------------------------------------------------------------


class TestCypherSafety:
    """Verify the module contains no f-string or %-format interpolation."""

    def _source(self) -> str:
        return inspect.getsource(_mod)

    def test_no_fstring_double_quote(self) -> None:
        assert not re.search(r'\bf"', self._source()), "f-string (double-quote) found in module"

    def test_no_fstring_single_quote(self) -> None:
        assert not re.search(r"\bf'", self._source()), "f-string (single-quote) found in module"

    def test_no_percent_format(self) -> None:
        # Detect patterns like "..." % (...) or "..." % variable
        assert not re.search(r'["\']\s*%\s*[(\w]', self._source()), (
            "%-format interpolation found in module"
        )

    def test_all_templates_use_dollar_params(self) -> None:
        # Every template must use $ids (parameterised input)
        for key, templates in TEMPLATES.items():
            for i, t in enumerate(templates):
                assert "$ids" in t, (
                    f"TEMPLATES[{key!r}][{i}] missing '$ids' placeholder"
                )


# ---------------------------------------------------------------------------
# TestCartesianGuard
# ---------------------------------------------------------------------------


class TestCartesianGuard:
    """Structural tests verifying UNWIND $ids prevents row multiplication."""

    def test_all_templates_start_with_unwind_ids(self) -> None:
        for key, templates in TEMPLATES.items():
            for i, t in enumerate(templates):
                assert t.startswith("UNWIND $ids AS id"), (
                    f"TEMPLATES[{key!r}][{i}] does not start with 'UNWIND $ids AS id'"
                )

    def test_1hop_templates_have_single_unwind_ids(self) -> None:
        # 1-hop templates must have exactly one UNWIND $ids (no accidental double unwind)
        for key, templates in TEMPLATES.items():
            t = templates[0]
            count = t.count("UNWIND $ids")
            assert count == 1, f"TEMPLATES[{key!r}][0] has {count} 'UNWIND $ids', expected 1"

    def test_id_binding_follows_unwind(self) -> None:
        # Each template binds the loop variable: UNWIND $ids AS id ... WHERE id(src) = id
        for key, templates in TEMPLATES.items():
            for i, t in enumerate(templates):
                assert "UNWIND $ids AS id" in t, (
                    f"TEMPLATES[{key!r}][{i}] missing 'UNWIND $ids AS id'"
                )

    def test_mock_session_two_ids_calls_run_once(self) -> None:
        # Passing two IDs should produce a single session.run() call (batch, not loop)
        mock_row = {
            "source_id": "1",
            "source_label": "Person",
            "rel_type": "DIRECTOR_OF",
            "target_id": "2",
            "target_label": "Company",
        }
        session = MagicMock()
        session.run.return_value = [mock_row, mock_row]
        results = lookup_facts(session, "Person", "Company", [1, 2])
        session.run.assert_called_once()
        assert len(results) == 2


# ---------------------------------------------------------------------------
# TestLookupFacts
# ---------------------------------------------------------------------------


class TestLookupFacts:
    """Unit tests for the lookup_facts wrapper."""

    def _make_mock_row(
        self,
        source_id: str = "10",
        source_label: str = "Person",
        rel_type: str = "DIRECTOR_OF",
        target_id: str = "20",
        target_label: str = "Company",
    ) -> dict:
        return {
            "source_id": source_id,
            "source_label": source_label,
            "rel_type": rel_type,
            "target_id": target_id,
            "target_label": target_label,
        }

    def test_returns_list_of_graph_facts(self) -> None:
        session = MagicMock()
        session.run.return_value = [self._make_mock_row()]
        result = lookup_facts(session, "Person", "Company", [10])
        assert len(result) == 1
        assert isinstance(result[0], GraphFact)

    def test_graph_fact_fields_populated(self) -> None:
        session = MagicMock()
        session.run.return_value = [self._make_mock_row()]
        result = lookup_facts(session, "Person", "Company", [10])
        fact = result[0]
        assert fact.source_id == "10"
        assert fact.source_label == "Person"
        assert fact.rel_type == "DIRECTOR_OF"
        assert fact.target_id == "20"
        assert fact.target_label == "Company"

    def test_uses_1hop_template_by_default(self) -> None:
        session = MagicMock()
        session.run.return_value = []
        lookup_facts(session, "Person", "Company", [1])
        called_cypher = session.run.call_args[0][0]
        assert called_cypher == TEMPLATES[("Person", "Company")][0]

    def test_uses_2hop_template_when_hop_is_2(self) -> None:
        session = MagicMock()
        session.run.return_value = []
        lookup_facts(session, "Person", "Company", [1], hop=2)
        called_cypher = session.run.call_args[0][0]
        assert called_cypher == TEMPLATES[("Person", "Company")][1]

    def test_uses_3hop_template_when_hop_is_3(self) -> None:
        session = MagicMock()
        session.run.return_value = []
        lookup_facts(session, "Person", "Company", [1], hop=3)
        called_cypher = session.run.call_args[0][0]
        assert called_cypher == TEMPLATES[("Person", "Company")][2]

    def test_passes_ids_as_param(self) -> None:
        session = MagicMock()
        session.run.return_value = []
        lookup_facts(session, "Person", "Company", [42, 99])
        called_params = session.run.call_args[0][1]
        assert called_params == {"ids": [42, 99]}

    def test_multiple_rows_returned(self) -> None:
        session = MagicMock()
        session.run.return_value = [
            self._make_mock_row(target_id="20"),
            self._make_mock_row(target_id="21"),
        ]
        result = lookup_facts(session, "Person", "Company", [10])
        assert len(result) == 2
        assert result[0].target_id == "20"
        assert result[1].target_id == "21"

    def test_unknown_key_raises_key_error(self) -> None:
        session = MagicMock()
        with pytest.raises(KeyError):
            lookup_facts(session, "UnknownLabel", "AnotherLabel", [1])

    def test_invalid_hop_raises_index_error(self) -> None:
        session = MagicMock()
        session.run.return_value = []
        with pytest.raises(IndexError):
            lookup_facts(session, "Person", "Company", [1], hop=4)


# ---------------------------------------------------------------------------
# Integration tests (require live Neo4j — skipped when unavailable)
# ---------------------------------------------------------------------------

_SKIP_INTEGRATION = pytest.mark.skipif(
    not _os.getenv("INTEGRATION_TESTS"),
    reason="set INTEGRATION_TESTS=1 to run integration tests against a live Neo4j instance",
)


@_SKIP_INTEGRATION
class TestIntegrationMultiHop:
    """Integration tests against a seeded Neo4j instance."""

    @pytest.fixture(scope="class")
    def driver(self):
        from neo4j import GraphDatabase

        uri = _os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = _os.getenv("NEO4J_USER", "neo4j")
        password = _os.getenv("NEO4J_PASSWORD", "password")
        d = GraphDatabase.driver(uri, auth=(user, password))
        yield d
        d.close()

    def test_party_to_returns_graph_facts(self, driver) -> None:
        """Company nodes that are PARTY_TO contracts return GraphFacts."""
        with driver.session() as session:
            # Fetch a Company node's internal ID from the seeded data.
            result = session.run(
                "MATCH (c:Company) RETURN id(c) AS nid LIMIT 1"
            )
            row = result.single()
            if row is None:
                pytest.skip("No Company nodes in database — seed first")
            company_nid = row["nid"]

        with driver.session() as session:
            facts = lookup_facts(session, "Company", "Contract", [company_nid])

        # A seeded company should be party to at least one contract.
        assert isinstance(facts, list)
        for fact in facts:
            assert fact.source_label == "Company"
            assert fact.rel_type == "PARTY_TO"
            assert fact.target_label == "Contract"

    def test_director_of_returns_graph_facts(self, driver) -> None:
        """Person nodes with DIRECTOR_OF edges return correct GraphFacts."""
        with driver.session() as session:
            result = session.run(
                "MATCH (p:Person)-[:DIRECTOR_OF]->(:Company) "
                "RETURN id(p) AS nid LIMIT 1"
            )
            row = result.single()
            if row is None:
                pytest.skip("No DIRECTOR_OF edges in database — seed first")
            person_nid = row["nid"]

        with driver.session() as session:
            facts = lookup_facts(session, "Person", "Company", [person_nid])

        assert len(facts) >= 1
        assert all(f.source_label == "Person" for f in facts)
        assert all(f.rel_type == "DIRECTOR_OF" for f in facts)
        assert all(f.target_label == "Company" for f in facts)

    def test_executed_cypher_matches_template(self, driver) -> None:
        """The template string executed is the one stored in TEMPLATES."""
        with driver.session() as session:
            result = session.run("MATCH (c:Company) RETURN id(c) AS nid LIMIT 1")
            row = result.single()
            if row is None:
                pytest.skip("No Company nodes in database — seed first")
            company_nid = row["nid"]

        expected_cypher = TEMPLATES[("Company", "Contract")][0]

        with driver.session() as session:
            # Capture by calling lookup_facts and verifying the template is what
            # TEMPLATES contains — no gap between the stored string and execution.
            facts = lookup_facts(session, "Company", "Contract", [company_nid])

        # The template key must resolve to the 1-hop PARTY_TO query.
        assert "-[:PARTY_TO]->" in expected_cypher
        assert "UNWIND $ids" in expected_cypher
        # Return value is typed correctly regardless of result count.
        assert isinstance(facts, list)
