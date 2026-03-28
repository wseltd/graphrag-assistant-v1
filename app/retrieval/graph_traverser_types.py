"""Graph traverser types.

Pure data definitions — no logic, no internal imports.

GraphFact
---------
A single edge fact extracted from a Neo4j traversal result.  Each instance
represents one relationship between two nodes as returned by a Cypher query.

Type aliases
------------
CypherTemplate  — a raw Cypher string stored as a module-level constant.
                  All templates must use UNWIND $ids and named $param
                  placeholders; never interpolate user input.
TemplateKey     — a (source_label, relationship_type) pair used to look up
                  the appropriate CypherTemplate in a dispatch table.
"""
from __future__ import annotations

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

CypherTemplate = str
"""A parameterised Cypher query string.

Convention: every template must contain ``UNWIND $ids`` and use ``$param``
placeholders for all variable inputs.  String interpolation of any external
value into a CypherTemplate is forbidden.
"""

TemplateKey = tuple[str, str]
"""(source_label, relationship_type) dispatch key for CypherTemplate lookup."""


# ---------------------------------------------------------------------------
# GraphFact
# ---------------------------------------------------------------------------


class GraphFact(BaseModel):
    """One directed relationship fact returned by a graph traversal.

    All fields are immutable strings.  GraphFact instances are value objects:
    two facts with identical field values compare equal regardless of identity.

    Attributes:
        source_id:     Unique node identifier of the source node.
        source_label:  Neo4j label of the source node (e.g. ``"Person"``).
        rel_type:      Relationship type (e.g. ``"DIRECTOR_OF"``).
        target_id:     Unique node identifier of the target node.
        target_label:  Neo4j label of the target node (e.g. ``"Company"``).
    """

    source_id: str
    source_label: str
    rel_type: str
    target_id: str
    target_label: str

    model_config = {"frozen": True}

    def __repr__(self) -> str:
        parts = ", ".join([
            "source_id=" + repr(self.source_id),
            "source_label=" + repr(self.source_label),
            "rel_type=" + repr(self.rel_type),
            "target_id=" + repr(self.target_id),
            "target_label=" + repr(self.target_label),
        ])
        return "GraphFact(" + parts + ")"
