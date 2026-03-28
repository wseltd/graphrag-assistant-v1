# GraphRAG Assistant

GraphRAG Assistant answers relationship-heavy enterprise queries over procurement and
contract data by combining graph traversal with constrained vector retrieval. Pure
embedding similarity search treats every text chunk as independent — it cannot reliably
answer questions like "who directs the suppliers on contract C-2024-07?" because no
single chunk contains all required facts across companies, directors, and clauses. This
system resolves named entities from the question (companies, persons, contracts) against
a Neo4j property graph, traverses typed edges (DIRECTOR_OF, SUPPLIES, PARTY_TO,
HAS_CLAUSE) to collect graph evidence, then restricts vector search to chunks reachable
from those entities. The result: multi-hop answers supported by both graph facts and
verbatim text citations, with a plain-RAG baseline for head-to-head comparison.

## Architecture

```
+---------------------------+  +----------------------------+
|        Plain RAG          |  |          GraphRAG          |
+---------------------------+  +----------------------------+
|  question                 |  |  question                  |
|    |                      |  |    |                       |
|  [embed]                  |  |  [entity extract]          |
|    |                      |  |    |                       |
|  [vector search]          |  |  [Cypher traversal]        |
|    |                      |  |    |                       |
|  [generate]               |  |  [constrained chunks]      |
|    |                      |  |    |                       |
|  answer                   |  |  [generate + evidence]     |
+---------------------------+  |    |                       |
                               |  answer                    |
                               +----------------------------+

Domain entities: Company, Person, Address, Product, Contract, Clause, Chunk
Edges: DIRECTOR_OF, REGISTERED_AT, SUPPLIES, PARTY_TO,
       HAS_CLAUSE, FROM_CONTRACT, ABOUT_COMPANY, RELATED_TO
```

## Quick Start

Prerequisites: Docker with the Compose plugin.

```bash
docker compose up -d
curl http://localhost:8000/health
curl -X POST http://localhost:8000/seed
curl -X POST http://localhost:8000/query/graph-rag \
     -H "Content-Type: application/json" \
     -d '{"question": "Who directs the companies supplying software to MeridianCorp?"}'
curl -X POST http://localhost:8000/benchmark/run
curl http://localhost:8000/benchmark/results/<run_id>
```

Replace `<run_id>` with the value returned by `POST /benchmark/run`.

## Example Queries

**Query 1 — multi-hop graph question**

```
POST /query/graph-rag
{"question": "Who directs the companies supplying software to MeridianCorp?"}
```

```json
{
  "answer": "Jane Doe is a director of TechSupply Ltd, which supplies software licences to MeridianCorp under contract C-2024-03.",
  "graph_evidence": [
    {"source_id": "person-002", "target_id": "company-003", "label": "DIRECTOR_OF"},
    {"source_id": "company-003", "target_id": "product-007", "label": "SUPPLIES"}
  ],
  "text_citations": [
    {
      "doc_id": "contract-003",
      "chunk_id": "chunk-d4e5f6",
      "quote": "TechSupply Ltd shall deliver software licences as specified in Schedule A."
    }
  ],
  "retrieval_debug": {
    "graph_query": "MATCH (c:Company)-[:SUPPLIES]->(p:Product) WHERE c.id IN $ids ...",
    "entity_matches": [
      {"name": "MeridianCorp", "node_id": "company-001", "score": 1.0}
    ],
    "retrieved_node_ids": ["company-001", "company-003", "person-002"],
    "chunk_ids": ["chunk-d4e5f6"],
    "timings": {
      "embed_ms": 11.2, "graph_ms": 9.4,
      "retrieve_ms": 28.7, "generate_ms": 102.5
    }
  },
  "mode": "graph_rag"
}
```

**Query 2 — graph-plus-text clause question**

```
POST /query/graph-rag
{"question": "Which contracts with CloudBase Ltd include a force-majeure clause?"}
```

```json
{
  "answer": "Contract C-2024-01 with CloudBase Ltd includes a force-majeure clause (Clause 12.3) covering events beyond reasonable control.",
  "graph_evidence": [
    {"source_id": "contract-001", "target_id": "clause-012", "label": "HAS_CLAUSE"},
    {"source_id": "company-005", "target_id": "contract-001", "label": "PARTY_TO"}
  ],
  "text_citations": [
    {
      "doc_id": "contract-001",
      "chunk_id": "chunk-a1b2c3",
      "quote": "Neither party shall be liable for failure to perform due to events beyond reasonable control."
    }
  ],
  "retrieval_debug": {
    "graph_query": "MATCH (c:Contract)-[:HAS_CLAUSE]->(cl:Clause) WHERE c.id IN $ids ...",
    "entity_matches": [
      {"name": "CloudBase Ltd", "node_id": "company-005", "score": 0.92}
    ],
    "retrieved_node_ids": ["company-005", "contract-001", "clause-012"],
    "chunk_ids": ["chunk-a1b2c3"],
    "timings": {
      "embed_ms": 10.8, "graph_ms": 7.3,
      "retrieve_ms": 25.1, "generate_ms": 88.6
    }
  },
  "mode": "graph_rag"
}
```

## Benchmark Design

Twenty fixed queries in `data/processed/benchmark_queries.jsonl`:

- **10 multi-hop graph queries** — require two or more relationship traversals;
  embedding-only retrieval is expected to miss these.
- **5 graph-plus-text clause queries** — locate contract entities via graph, then
  retrieve and cite clause text.
- **5 distractor questions** — answerable from text alone; validates that GraphRAG
  does not over-claim graph evidence.

Four metrics are computed per run:

- **Answer accuracy** — keyword overlap against expected answers in
  `data/expected/benchmark_answers.jsonl`
- **Citation coverage** — fraction of expected chunk IDs present in the response
- **Latency** — wall-clock time per query in milliseconds
- **GraphRAG vs plain-RAG hit rate** — head-to-head win rate across identical queries

Run via `POST /benchmark/run`; retrieve results via `GET /benchmark/results/{run_id}`.

## Limitations

- The generation layer is a template stub and does not call a real language model.
  Replace `GenerationProvider` for production-quality answers.
- No authentication is configured on any endpoint; this is a demo system only.
- Neo4j runs as a single instance with no replication, clustering, or backup.
- Ingestion is batch-only; there is no incremental update path. Re-run `POST /seed`
  to reload data after changes.
- Entity extraction is keyword-based; compound names and abbreviations reduce recall.

## Trade-Offs

- **Graph-first, not graph-only:** when no entity is resolved from the question, the
  system falls back to unconstrained vector search. Recall is preserved; precision is
  lower without graph anchoring.
- **Synchronous calls:** all embedding, graph, and generation calls are synchronous.
  Adequate for single-user demo workloads; concurrent load requires an async refactor.
- **Local embeddings:** `sentence-transformers` operates fully offline but produces
  lower-quality representations than hosted APIs. Swap via `EmbeddingProvider` without
  changing retrieval logic.
- **Fixed graph schema:** the domain model is compiled into Cypher templates. Extending
  the schema requires adding new templates, not a configuration change.

## Non-Goals

- Multi-tenancy or per-user data isolation
- Real-time or streaming ingestion
- Cloud deployment or managed Neo4j Aura support
- Frontend UI or interactive query console
- Enterprise authentication or authorisation
