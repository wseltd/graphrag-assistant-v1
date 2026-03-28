"""Write benchmark_queries.jsonl and benchmark_answers.jsonl from static definitions.

Run:
    python -m scripts.seed.seed_benchmark

Output files:
    data/processed/benchmark_queries.jsonl  — 20 query records
    data/expected/benchmark_answers.jsonl   — 20 expected-answer records

Both files are deterministic and byte-identical on repeated runs.

required_chunk_ids strings use the format "<doc_id>:<chunk_id>" (e.g. "K001:9")
where chunk_id is the integer offset matching chunks.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = [
    "QUERIES",
    "EXPECTED_ANSWERS",
    "ANSWERS",
    "write_jsonl",
    "write_queries",
    "write_answers",
    "main",
]

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
_PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
_EXPECTED_DIR = Path(__file__).parent.parent.parent / "data" / "expected"
_QUERIES_FILE = _PROCESSED_DIR / "benchmark_queries.jsonl"
_ANSWERS_FILE = _EXPECTED_DIR / "benchmark_answers.jsonl"

# ---------------------------------------------------------------------------
# Benchmark queries
# Distribution: 10 multi_hop_graph, 5 graph_plus_text, 5 distractor
# All entity IDs are from T010 seed CSVs (companies, people, addresses, products).
# All chunk IDs reference {doc_id}:{chunk_id} pairs present in chunks.jsonl.
# ---------------------------------------------------------------------------
QUERIES: list[dict] = [
    # ------------------------------------------------------------------ #
    # BQ001–BQ010  multi_hop_graph
    # ------------------------------------------------------------------ #
    {
        "query_id": "BQ001",
        "question": (
            "Who is the nominated contract manager on the buyer side of contract K001, "
            "and what is their job title at the buyer company?"
        ),
        "category": "multi_hop_graph",
        "required_entity_ids": ["C001", "P001"],
        "required_chunk_ids": ["K001:9"],
    },
    {
        "query_id": "BQ002",
        "question": (
            "Which company is contracted as the cloud migration consultant to "
            "Meridian Holdings Ltd, and in which UK city is it registered?"
        ),
        "category": "multi_hop_graph",
        "required_entity_ids": ["C001", "C003", "A003"],
        "required_chunk_ids": [],
    },
    {
        "query_id": "BQ003",
        "question": (
            "How many distinct products does Albrecht und Partner GmbH supply "
            "in the product catalogue, and what are their product IDs?"
        ),
        "category": "multi_hop_graph",
        "required_entity_ids": ["C004", "PR002", "PR006", "PR010", "PR014"],
        "required_chunk_ids": [],
    },
    {
        "query_id": "BQ004",
        "question": (
            "Which contract connects Thornbridge Consulting Ltd to Vantage Systems Ltd, "
            "and who is Thornbridge's nominated contract manager in that agreement?"
        ),
        "category": "multi_hop_graph",
        "required_entity_ids": ["C009", "C006", "P007"],
        "required_chunk_ids": ["K005:12"],
    },
    {
        "query_id": "BQ005",
        "question": (
            "Which two companies each have exactly two individuals listed in the "
            "people dataset as employees or directors?"
        ),
        "category": "multi_hop_graph",
        "required_entity_ids": ["C004", "P003", "P014", "C008", "P008", "P013"],
        "required_chunk_ids": [],
    },
    {
        "query_id": "BQ006",
        "question": (
            "Who is the Finance Director of the company that engaged Pinnacle "
            "Contracts PLC for disaster recovery planning under contract K006?"
        ),
        "category": "multi_hop_graph",
        "required_entity_ids": ["C012", "C007", "P015"],
        "required_chunk_ids": ["K006:11", "K006:12"],
    },
    {
        "query_id": "BQ007",
        "question": (
            "Which supplier company provides both a software licence platform and a "
            "maintenance assurance programme to the same client under a single contract?"
        ),
        "category": "multi_hop_graph",
        "required_entity_ids": ["C006", "C009", "PR008", "PR015"],
        "required_chunk_ids": ["K005:11"],
    },
    {
        "query_id": "BQ008",
        "question": (
            "Which individual appears as the buyer-side contract manager across the "
            "greatest number of contracts in the dataset, and how many contracts does "
            "that cover?"
        ),
        "category": "multi_hop_graph",
        "required_entity_ids": ["P001", "C001"],
        "required_chunk_ids": [],
    },
    {
        "query_id": "BQ009",
        "question": (
            "What professional service product does the company that is the consultant "
            "in contract K006 supply in the product catalogue?"
        ),
        "category": "multi_hop_graph",
        "required_entity_ids": ["C007", "PR013"],
        "required_chunk_ids": [],
    },
    {
        "query_id": "BQ010",
        "question": (
            "Which supplier in the product catalogue exclusively supplies Hardware "
            "products, with no Software Licences, Professional Services, or "
            "Maintenance & Support products in their portfolio?"
        ),
        "category": "multi_hop_graph",
        "required_entity_ids": ["C004", "PR002", "PR006", "PR010", "PR014"],
        "required_chunk_ids": [],
    },
    # ------------------------------------------------------------------ #
    # BQ011–BQ015  graph_plus_text
    # ------------------------------------------------------------------ #
    {
        "query_id": "BQ011",
        "question": (
            "What is the liability cap that Nexus Procurement PLC has agreed to "
            "under contract K001 with Meridian Holdings Ltd?"
        ),
        "category": "graph_plus_text",
        "required_entity_ids": ["C001", "C002"],
        "required_chunk_ids": ["K001:21", "K001:22"],
    },
    {
        "query_id": "BQ012",
        "question": (
            "Under what three specific circumstances can Meridian Holdings Ltd "
            "terminate contract K002 with Hartwell Solutions Ltd with immediate effect?"
        ),
        "category": "graph_plus_text",
        "required_entity_ids": ["C001", "C003"],
        "required_chunk_ids": ["K002:24", "K002:25"],
    },
    {
        "query_id": "BQ013",
        "question": (
            "What service credit can Meridian Holdings Ltd claim if CentroLogic "
            "Industrie SA fails to meet Priority 1 incident restoration targets for "
            "two consecutive months under contract K004?"
        ),
        "category": "graph_plus_text",
        "required_entity_ids": ["C001", "C005"],
        "required_chunk_ids": ["K004:15", "K004:16"],
    },
    {
        "query_id": "BQ014",
        "question": (
            "Who owns the deliverables produced solely for Meridian Holdings Ltd "
            "under contract K002, and what happens to Hartwell Solutions Ltd's "
            "pre-existing intellectual property?"
        ),
        "category": "graph_plus_text",
        "required_entity_ids": ["C001", "C003"],
        "required_chunk_ids": ["K002:15", "K002:16"],
    },
    {
        "query_id": "BQ015",
        "question": (
            "What are the two instalment amounts and their respective payment "
            "triggers for the disaster recovery engagement under contract K006?"
        ),
        "category": "graph_plus_text",
        "required_entity_ids": ["C012", "C007"],
        "required_chunk_ids": ["K006:13", "K006:14"],
    },
    # ------------------------------------------------------------------ #
    # BQ016–BQ020  distractor
    # ------------------------------------------------------------------ #
    {
        "query_id": "BQ016",
        "question": (
            "Does Eurocom Trading SA hold any active procurement or supply contract "
            "documented in the dataset?"
        ),
        "category": "distractor",
        "required_entity_ids": ["C010"],
        "required_chunk_ids": [],
    },
    {
        "query_id": "BQ017",
        "question": (
            "Which contracts involve Lucas Vandermeer's employer, Redstone "
            "Technologies LLC, as a buyer or supplier?"
        ),
        "category": "distractor",
        "required_entity_ids": ["P008", "C008"],
        "required_chunk_ids": [],
    },
    {
        "query_id": "BQ018",
        "question": (
            "What contract governs the procurement of the Data Analytics Platform "
            "(PR005) by any buyer in the dataset?"
        ),
        "category": "distractor",
        "required_entity_ids": ["PR005", "C002"],
        "required_chunk_ids": [],
    },
    {
        "query_id": "BQ019",
        "question": (
            "Which company registered in Brussels has a direct supply or consultancy "
            "relationship documented in any contract in the dataset?"
        ),
        "category": "distractor",
        "required_entity_ids": ["C010", "A010"],
        "required_chunk_ids": [],
    },
    {
        "query_id": "BQ020",
        "question": (
            "Is there any contract between Stackfield Digital GmbH and any "
            "procurement entity listed in the dataset?"
        ),
        "category": "distractor",
        "required_entity_ids": ["C011"],
        "required_chunk_ids": [],
    },
]

# ---------------------------------------------------------------------------
# Expected answers
# Each entry matches the query at the same list position (query_ids must agree).
# graph_evidence entries reference rows in directorships.csv (DIRECTOR_OF),
# supplies.csv (SUPPLIES), or contracts.csv (PARTY_TO).
# text_citations chunk_id values are integers matching chunks.jsonl.
# ---------------------------------------------------------------------------
ANSWERS: list[dict] = [
    # BQ001
    {
        "query_id": "BQ001",
        "answer": (
            "James Blackwood (P001), Chief Procurement Officer at Meridian Holdings Ltd "
            "(C001), is the nominated contract manager on the buyer side of contract K001."
        ),
        "graph_evidence": [
            {"source_id": "P001", "target_id": "C001", "label": "DIRECTOR_OF"},
            {"source_id": "C001", "target_id": "K001", "label": "PARTY_TO"},
        ],
        "text_citations": [
            {
                "doc_id": "K001",
                "chunk_id": 9,
                "quote": (
                    "James Blackwood holds authority to approve delivery milestones, "
                    "authorise scope changes in writing, and formally accept the delivered "
                    "system on behalf of the Buyer."
                ),
            },
        ],
        "mode": "graph_rag",
    },
    # BQ002
    {
        "query_id": "BQ002",
        "answer": (
            "Hartwell Solutions Ltd (C003), registered in Birmingham, is the cloud "
            "migration consultant to Meridian Holdings Ltd (C001) under contract K002."
        ),
        "graph_evidence": [
            {"source_id": "C001", "target_id": "K002", "label": "PARTY_TO"},
            {"source_id": "C003", "target_id": "K002", "label": "PARTY_TO"},
        ],
        "text_citations": [],
        "mode": "graph_rag",
    },
    # BQ003
    {
        "query_id": "BQ003",
        "answer": (
            "Albrecht und Partner GmbH (C004) supplies four distinct products: "
            "PR002 (Network Security Appliance), PR006 (Server Rack Unit 42U), "
            "PR010 (Industrial Grade Router), and PR014 (UPS Power Module 10kVA)."
        ),
        "graph_evidence": [
            {"source_id": "C004", "target_id": "PR002", "label": "SUPPLIES"},
            {"source_id": "C004", "target_id": "PR006", "label": "SUPPLIES"},
            {"source_id": "C004", "target_id": "PR010", "label": "SUPPLIES"},
            {"source_id": "C004", "target_id": "PR014", "label": "SUPPLIES"},
        ],
        "text_citations": [],
        "mode": "graph_rag",
    },
    # BQ004
    {
        "query_id": "BQ004",
        "answer": (
            "Contract K005 connects Thornbridge Consulting Ltd (C009, client) with "
            "Vantage Systems Ltd (C006, supplier). Emma Thornton (P007), Principal "
            "Consultant at Thornbridge Consulting Ltd, is the nominated contract manager."
        ),
        "graph_evidence": [
            {"source_id": "P007", "target_id": "C009", "label": "DIRECTOR_OF"},
            {"source_id": "C009", "target_id": "K005", "label": "PARTY_TO"},
            {"source_id": "C006", "target_id": "K005", "label": "PARTY_TO"},
        ],
        "text_citations": [
            {
                "doc_id": "K005",
                "chunk_id": 12,
                "quote": (
                    "Emma Thornton holds authority to approve configuration changes, "
                    "certify deliverables, and authorise assurance renewals on behalf of "
                    "the Client."
                ),
            },
        ],
        "mode": "graph_rag",
    },
    # BQ005
    {
        "query_id": "BQ005",
        "answer": (
            "Albrecht und Partner GmbH (C004) has two individuals listed: "
            "Heinrich Müller (P003, Managing Director) and Katarina Bauer (P014, "
            "Legal Counsel). Redstone Technologies LLC (C008) also has two: "
            "Lucas Vandermeer (P008, Head of Technology) and Thomas Redstone "
            "(P013, Infrastructure Lead)."
        ),
        "graph_evidence": [
            {"source_id": "P003", "target_id": "C004", "label": "DIRECTOR_OF"},
            {"source_id": "P014", "target_id": "C004", "label": "DIRECTOR_OF"},
            {"source_id": "P008", "target_id": "C008", "label": "DIRECTOR_OF"},
            {"source_id": "P013", "target_id": "C008", "label": "DIRECTOR_OF"},
        ],
        "text_citations": [],
        "mode": "graph_rag",
    },
    # BQ006
    {
        "query_id": "BQ006",
        "answer": (
            "William Clearwater (P015), Finance Director of Clearwater Infrastructure "
            "PLC (C012), is the nominated contract manager for K006. He holds authority "
            "to approve phase deliverables and formally certify completion on behalf of "
            "Clearwater Infrastructure PLC."
        ),
        "graph_evidence": [
            {"source_id": "P015", "target_id": "C012", "label": "DIRECTOR_OF"},
            {"source_id": "C012", "target_id": "K006", "label": "PARTY_TO"},
            {"source_id": "C007", "target_id": "K006", "label": "PARTY_TO"},
        ],
        "text_citations": [
            {
                "doc_id": "K006",
                "chunk_id": 11,
                "quote": (
                    "The Client's nominated contract manager for this engagement is "
                    "William Clearwater, Finance Director of Clearwater Infrastructure PLC."
                ),
            },
            {
                "doc_id": "K006",
                "chunk_id": 12,
                "quote": (
                    "William Clearwater holds authority to approve phase deliverables, "
                    "authorise any changes to the agreed scope, and formally certify "
                    "completion of the engagement on behalf of the Client."
                ),
            },
        ],
        "mode": "graph_rag",
    },
    # BQ007
    {
        "query_id": "BQ007",
        "answer": (
            "Vantage Systems Ltd (C006) supplies both the Infrastructure Monitoring "
            "Suite (PR008, a Software Licences product) and the Software Licence "
            "Assurance Programme (PR015, a Maintenance & Support product) to "
            "Thornbridge Consulting Ltd (C009) under contract K005."
        ),
        "graph_evidence": [
            {"source_id": "C006", "target_id": "PR008", "label": "SUPPLIES"},
            {"source_id": "C006", "target_id": "PR015", "label": "SUPPLIES"},
            {"source_id": "C006", "target_id": "K005", "label": "PARTY_TO"},
            {"source_id": "C009", "target_id": "K005", "label": "PARTY_TO"},
        ],
        "text_citations": [
            {
                "doc_id": "K005",
                "chunk_id": 11,
                "quote": (
                    "The programme entitles the Client to all major and minor product "
                    "version releases, access to the Supplier's technical support portal, "
                    "and priority escalation for critical platform defects."
                ),
            },
        ],
        "mode": "graph_rag",
    },
    # BQ008
    {
        "query_id": "BQ008",
        "answer": (
            "James Blackwood (P001), Chief Procurement Officer at Meridian Holdings Ltd "
            "(C001), acts as buyer-side contract manager in four contracts: K001 "
            "(ERP licence), K002 (cloud migration), K003 (hardware supply), and K004 "
            "(managed IT services)."
        ),
        "graph_evidence": [
            {"source_id": "P001", "target_id": "C001", "label": "DIRECTOR_OF"},
            {"source_id": "C001", "target_id": "K001", "label": "PARTY_TO"},
            {"source_id": "C001", "target_id": "K002", "label": "PARTY_TO"},
            {"source_id": "C001", "target_id": "K003", "label": "PARTY_TO"},
            {"source_id": "C001", "target_id": "K004", "label": "PARTY_TO"},
        ],
        "text_citations": [],
        "mode": "graph_rag",
    },
    # BQ009
    {
        "query_id": "BQ009",
        "answer": (
            "Pinnacle Contracts PLC (C007), the consultant in contract K006, supplies "
            "Disaster Recovery Planning (PR013) in the product catalogue — a "
            "Professional Services engagement priced at £28,000 per engagement."
        ),
        "graph_evidence": [
            {"source_id": "C007", "target_id": "K006", "label": "PARTY_TO"},
            {"source_id": "C007", "target_id": "PR013", "label": "SUPPLIES"},
        ],
        "text_citations": [],
        "mode": "graph_rag",
    },
    # BQ010
    {
        "query_id": "BQ010",
        "answer": (
            "Albrecht und Partner GmbH (C004) is the only supplier whose entire "
            "product catalogue consists of Hardware items: Network Security Appliance "
            "(PR002), Server Rack Unit 42U (PR006), Industrial Grade Router (PR010), "
            "and UPS Power Module 10kVA (PR014)."
        ),
        "graph_evidence": [
            {"source_id": "C004", "target_id": "PR002", "label": "SUPPLIES"},
            {"source_id": "C004", "target_id": "PR006", "label": "SUPPLIES"},
            {"source_id": "C004", "target_id": "PR010", "label": "SUPPLIES"},
            {"source_id": "C004", "target_id": "PR014", "label": "SUPPLIES"},
        ],
        "text_citations": [],
        "mode": "graph_rag",
    },
    # BQ011
    {
        "query_id": "BQ011",
        "answer": (
            "Under contract K001, Nexus Procurement PLC's total aggregate liability to "
            "Meridian Holdings Ltd is capped at one hundred per cent (100%) of the "
            "total fees paid in the twelve months immediately preceding the event giving "
            "rise to the claim."
        ),
        "graph_evidence": [
            {"source_id": "C001", "target_id": "K001", "label": "PARTY_TO"},
            {"source_id": "C002", "target_id": "K001", "label": "PARTY_TO"},
        ],
        "text_citations": [
            {
                "doc_id": "K001",
                "chunk_id": 21,
                "quote": (
                    "The Supplier's total aggregate liability to the Buyer under or in "
                    "connection with this Agreement, whether arising in contract, tort "
                    "(including negligence), breach of statutory duty, or otherwise, "
                    "shall not exceed one hundred per cent (100%) of the total fees paid "
                    "by Meridian Holdings Ltd to Nexus Procurement PLC under this "
                    "Agreement in the twelve (12) months immediately preceding the event "
                    "giving rise to the claim."
                ),
            },
            {
                "doc_id": "K001",
                "chunk_id": 22,
                "quote": (
                    "not exceed one hundred per cent (100%) of the total fees paid by "
                    "Meridian Holdings Ltd to Nexus Procurement PLC under this Agreement "
                    "in the twelve (12) months immediately preceding the event giving "
                    "rise to the claim."
                ),
            },
        ],
        "mode": "graph_rag",
    },
    # BQ012
    {
        "query_id": "BQ012",
        "answer": (
            "Meridian Holdings Ltd may terminate contract K002 with Hartwell Solutions "
            "Ltd with immediate effect if: (1) Hartwell Solutions Ltd enters insolvency "
            "proceedings; (2) ceases to carry on business; or (3) is unable to provide "
            "adequately qualified personnel to fulfil its obligations."
        ),
        "graph_evidence": [
            {"source_id": "C001", "target_id": "K002", "label": "PARTY_TO"},
            {"source_id": "C003", "target_id": "K002", "label": "PARTY_TO"},
        ],
        "text_citations": [
            {
                "doc_id": "K002",
                "chunk_id": 24,
                "quote": (
                    "Meridian Holdings Ltd may terminate with immediate effect if Hartwell "
                    "Solutions Ltd enters insolvency proceedings, ceases to carry on "
                    "business, or is unable to provide adequately qualified personnel to "
                    "fulfil its obligations."
                ),
            },
            {
                "doc_id": "K002",
                "chunk_id": 25,
                "quote": (
                    "Meridian Holdings Ltd may terminate with immediate effect if Hartwell "
                    "Solutions Ltd enters insolvency proceedings, ceases to carry on "
                    "business, or is unable to provide adequately qualified personnel to "
                    "fulfil its obligations."
                ),
            },
        ],
        "mode": "graph_rag",
    },
    # BQ013
    {
        "query_id": "BQ013",
        "answer": (
            "If CentroLogic Industrie SA fails to meet Priority 1 restoration targets in "
            "any two consecutive months, Meridian Holdings Ltd may claim a service "
            "credit equal to ten per cent (10%) of the monthly managed services fee "
            "for each affected month."
        ),
        "graph_evidence": [
            {"source_id": "C001", "target_id": "K004", "label": "PARTY_TO"},
            {"source_id": "C005", "target_id": "K004", "label": "PARTY_TO"},
        ],
        "text_citations": [
            {
                "doc_id": "K004",
                "chunk_id": 15,
                "quote": (
                    "Failure to meet Priority 1 restoration targets in any two consecutive "
                    "months shall entitle the Client to claim a service credit equal to "
                    "ten per cent (10%) of the monthly managed services fee for each "
                    "affected month."
                ),
            },
            {
                "doc_id": "K004",
                "chunk_id": 16,
                "quote": (
                    "Failure to meet Priority 1 restoration targets in any two consecutive "
                    "months shall entitle the Client to claim a service credit equal to "
                    "ten per cent (10%) of the monthly managed services fee for each "
                    "affected month."
                ),
            },
        ],
        "mode": "graph_rag",
    },
    # BQ014
    {
        "query_id": "BQ014",
        "answer": (
            "All deliverables produced solely for Meridian Holdings Ltd under K002 "
            "(including the Infrastructure Assessment Report and Migration Runbook) vest "
            "in Meridian Holdings Ltd upon full payment. Hartwell Solutions Ltd retains "
            "ownership of its pre-existing IP but grants Meridian a non-exclusive, "
            "royalty-free licence to use those materials to maintain the migrated "
            "infrastructure."
        ),
        "graph_evidence": [
            {"source_id": "C001", "target_id": "K002", "label": "PARTY_TO"},
            {"source_id": "C003", "target_id": "K002", "label": "PARTY_TO"},
        ],
        "text_citations": [
            {
                "doc_id": "K002",
                "chunk_id": 15,
                "quote": (
                    "All deliverables produced solely for the Client under this Agreement, "
                    "including the Infrastructure Assessment Report, the Migration Runbook, "
                    "and any bespoke configuration scripts or documentation, shall vest in "
                    "Meridian Holdings Ltd upon full payment of the associated fees."
                ),
            },
            {
                "doc_id": "K002",
                "chunk_id": 16,
                "quote": (
                    "Pre-existing intellectual property belonging to Hartwell Solutions "
                    "Ltd, including proprietary methodology frameworks, toolsets, and "
                    "assessment templates used in the delivery of the services, shall "
                    "remain the exclusive property of the Consultant."
                ),
            },
        ],
        "mode": "graph_rag",
    },
    # BQ015
    {
        "query_id": "BQ015",
        "answer": (
            "The K006 engagement is paid in two equal instalments of £14,000 each. "
            "The first instalment is due within thirty (30) calendar days of the "
            "Effective Date (1 September 2025). The second instalment is due within "
            "thirty (30) calendar days of Clearwater Infrastructure PLC's written "
            "acceptance of the Phase Two deliverables."
        ),
        "graph_evidence": [
            {"source_id": "C012", "target_id": "K006", "label": "PARTY_TO"},
            {"source_id": "C007", "target_id": "K006", "label": "PARTY_TO"},
        ],
        "text_citations": [
            {
                "doc_id": "K006",
                "chunk_id": 13,
                "quote": (
                    "The first instalment of fourteen thousand pounds sterling "
                    "(£14,000.00) is due within thirty (30) calendar days of the "
                    "Effective Date."
                ),
            },
            {
                "doc_id": "K006",
                "chunk_id": 14,
                "quote": (
                    "The second instalment of fourteen thousand pounds sterling "
                    "(£14,000.00) is due within thirty (30) calendar days of the "
                    "Client's written acceptance of the Phase Two deliverables."
                ),
            },
        ],
        "mode": "graph_rag",
    },
    # BQ016
    {
        "query_id": "BQ016",
        "answer": (
            "No. Eurocom Trading SA (C010) appears in the companies dataset but is "
            "not a party to any of the six contracts (K001–K006). No procurement or "
            "supply contract involving Eurocom Trading SA can be retrieved from the "
            "dataset."
        ),
        "graph_evidence": [],
        "text_citations": [],
        "mode": "graph_rag",
    },
    # BQ017
    {
        "query_id": "BQ017",
        "answer": (
            "No contracts involving Redstone Technologies LLC (C008) exist in the "
            "dataset. Lucas Vandermeer (P008) is listed as an employee of C008, but "
            "C008 is not a party to any of the six contracts (K001–K006)."
        ),
        "graph_evidence": [],
        "text_citations": [],
        "mode": "graph_rag",
    },
    # BQ018
    {
        "query_id": "BQ018",
        "answer": (
            "No contract governing the procurement of the Data Analytics Platform "
            "(PR005) exists in the dataset. PR005 is listed in the product catalogue "
            "as supplied by Nexus Procurement PLC (C002), but it is not referenced in "
            "any of the six contracts (K001–K006)."
        ),
        "graph_evidence": [],
        "text_citations": [],
        "mode": "graph_rag",
    },
    # BQ019
    {
        "query_id": "BQ019",
        "answer": (
            "No company registered in Brussels holds any contract relationship in the "
            "dataset. Eurocom Trading SA (C010), registered at Avenue Louise 367, "
            "Brussels, is not a party to any of the six contracts (K001–K006)."
        ),
        "graph_evidence": [],
        "text_citations": [],
        "mode": "graph_rag",
    },
    # BQ020
    {
        "query_id": "BQ020",
        "answer": (
            "No. Stackfield Digital GmbH (C011) appears in the companies dataset but "
            "has not entered into any of the six contracts (K001–K006) documented in "
            "the dataset."
        ),
        "graph_evidence": [],
        "text_citations": [],
        "mode": "graph_rag",
    },
]

# Public alias used by downstream consumers that import the canonical name.
# ANSWERS is kept for backward compatibility with tests written against T014.a.
EXPECTED_ANSWERS: list[dict] = ANSWERS


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_queries(out_file: Path, queries: list[dict]) -> None:
    """Write *queries* as JSONL to *out_file* (atomic, sorted keys)."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        for record in queries:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_answers(out_file: Path, answers: list[dict]) -> None:
    """Write *answers* as JSONL to *out_file* (atomic, sorted keys)."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        for record in answers:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Generic JSONL writer
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent


def write_jsonl(records: list[dict], path: str) -> None:
    """Write *records* as JSONL to *path* (sorted keys, UTF-8).

    Relative paths are resolved against the repository root so the script is
    runnable from any working directory.
    """
    out = Path(path) if Path(path).is_absolute() else _REPO_ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Write benchmark query and answer files to their canonical locations."""
    write_jsonl(QUERIES, "data/processed/benchmark_queries.jsonl")
    write_jsonl(EXPECTED_ANSWERS, "data/expected/benchmark_answers.jsonl")
    queries_out = _REPO_ROOT / "data/processed/benchmark_queries.jsonl"
    answers_out = _REPO_ROOT / "data/expected/benchmark_answers.jsonl"
    print(f"Wrote {len(QUERIES)} queries → {queries_out}")
    print(f"Wrote {len(EXPECTED_ANSWERS)} answers → {answers_out}")


if __name__ == "__main__":
    main()
