# GraphRAG Assistant

GraphRAG Assistant answers relationship-heavy enterprise queries over procurement and
contract data by combining graph traversal with constrained vector retrieval. Pure
embedding similarity search treats every text chunk as independent — it cannot reliably
answer questions like "who directs the suppliers on contract K001?" because no single
chunk contains all required facts across companies, directors, and clauses. This system
resolves named entities from the question (companies, persons, contracts) against a
Neo4j property graph, traverses typed edges (DIRECTOR_OF, SUPPLIES, PARTY_TO,
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
|  [embed]                  |  |  [entity resolver]         |
|    |                      |  |    ↓                       |
|  [vector search]          |  |  [graph traversal]         |
|    |                      |  |    ↓                       |
|  [generate]               |  |  [constrained retrieval]   |
|    |                      |  |    ↓                       |
|  answer + citations       |  |  [answer builder]          |
+---------------------------+  |    ↓                       |
                               |  answer + graph + citations|
                               +----------------------------+
```

**Domain graph:**

```
Nodes:    Company, Person, Address, Product, Contract, Clause, Chunk
Edges:    (Person)  -[:DIRECTOR_OF]->  (Company)
          (Company) -[:PARTY_TO]->     (Contract)
          (Company) -[:SUPPLIES]->     (Product)
          (Company) -[:REGISTERED_AT]-> (Address)
          (Contract)-[:HAS_CLAUSE]->   (Clause)
          (Chunk)   -[:FROM_CONTRACT]-> (Contract)
          (Chunk)   -[:ABOUT_COMPANY]-> (Company)
          (Chunk)   -[:RELATED_TO]->   (Person)
```

**Traversal paths guaranteed deterministically:**

- Company → PARTY_TO → Contract → HAS_CLAUSE → Clause (2-hop outbound)
- Company ← DIRECTOR_OF ← Person (inbound director lookup)
- Company → PARTY_TO → Contract ← PARTY_TO ← CoParty (co-party supplier discovery)
- CoParty ← DIRECTOR_OF ← Director (directors of supplier companies)

## Quick Start

Prerequisites: Docker with the Compose plugin.

```bash
docker compose up -d
curl http://localhost:8000/health

# Seed the graph (loads companies, contracts, clauses, chunks, relationships)
curl -s -X POST http://localhost:8000/seed | python3 -m json.tool

# Ask a multi-hop graph question (API key required)
curl -s -X POST http://localhost:8000/query/graph-rag \
     -H "Content-Type: application/json" \
     -H "X-Api-Key: dev-key-change-in-prod" \
     -d '{"question": "Who directs the suppliers to Meridian Holdings?"}' \
     | python3 -m json.tool

# Run the benchmark
curl -s -X POST http://localhost:8000/benchmark/run \
     -H "X-Api-Key: dev-key-change-in-prod" \
     | python3 -m json.tool
```

## Example Output

**Query: Who directs the suppliers to Meridian Holdings?**

The entity resolver finds `Meridian Holdings` → C001. The traversal expands outbound
PARTY_TO edges to contracts K001–K004, discovers the co-party companies (Nexus
Procurement, Hartwell Solutions, Albrecht und Partner, Sigma Systems), then follows
inbound DIRECTOR_OF edges from each co-party to their directors.

```json
{
  "answer": "Query: Who directs the suppliers to Meridian Holdings?\nGraph context:\n  Meridian Holdings --[PARTY_TO]--> K001\n  Meridian Holdings --[PARTY_TO]--> K002\n  Nexus Procurement --[PARTY_TO]--> K001\n  Hartwell Solutions --[PARTY_TO]--> K002\n  Victoria Walsh --[DIRECTOR_OF]--> Nexus Procurement\n  ...\nSupporting text:\n  [K001:0] Perpetual non-exclusive non-transferable licence...",
  "graph_evidence": [
    {"source_id": "Meridian Holdings", "target_id": "K001", "label": "PARTY_TO"},
    {"source_id": "Victoria Walsh",    "target_id": "Nexus Procurement", "label": "DIRECTOR_OF"}
  ],
  "text_citations": [
    {"doc_id": "K001", "chunk_id": "K001:0", "quote": "Perpetual non-exclusive non-transferable licence..."}
  ],
  "retrieval_debug": {
    "entity_matches": ["C001"],
    "retrieved_node_ids": ["C001"],
    "chunk_ids": ["K001:0", "K001:1"],
    "timings": {"graph_ms": 12.4, "retrieve_ms": 31.2, "generate_ms": 0.8}
  },
  "mode": "graph_rag"
}
```

**Query: What is the liability cap under contract K001?**

The entity resolver matches `K001` directly to the Contract node (no fuzzy matching
needed). Traversal follows HAS_CLAUSE edges to clause text. The constrained retrieval
returns only chunks linked to K001.

## Authentication

All `/query/*` and `/benchmark/*` endpoints require an `X-Api-Key` header:

```bash
-H "X-Api-Key: dev-key-change-in-prod"
```

Set `API_KEY` in the environment (comma-separated for multiple keys) to override the
default. `/health` and `/seed` are unauthenticated.

## Entity Resolution

Two-stage resolver:

1. **Candidate extraction** — capitalised-run tokenisation scans the query for noun
   phrases (e.g. "Meridian Holdings Ltd", "K001", "Victoria Walsh").
2. **Graph lookup** — each candidate runs a bidirectional CONTAINS query against
   Company, Person, and Product nodes so that "Meridian Holdings" matches the stored
   name "Meridian Holdings" even when the query appends "Ltd". Contract nodes are
   matched on both `title` and `contract_id`.

## Benchmark Design

Fixed queries in `data/processed/benchmark_queries.jsonl`:

- **Multi-hop graph queries** — require two or more relationship traversals; vector-only
  retrieval is expected to miss these.
- **Graph-plus-text clause queries** — locate contract entities via graph, then retrieve
  and cite clause text.
- **Distractor questions** — answerable from text alone; validates that GraphRAG does
  not over-claim graph evidence.

Four metrics per run: answer accuracy (keyword overlap), citation coverage, latency,
and GraphRAG vs plain-RAG head-to-head win rate.

Run via `POST /benchmark/run`; retrieve results via `GET /benchmark/results/{run_id}`.

## Limitations

- The answer builder is a deterministic template, not a language model. It outputs
  graph facts and supporting text excerpts verbatim. Replace `GenerationProvider` for
  fluent prose answers.
- Entity extraction is keyword-based; highly abbreviated or hyphenated names reduce
  recall.
- Neo4j runs as a single instance with no replication, clustering, or backup.
- Ingestion is batch-only; there is no incremental update path. Re-run `POST /seed`
  to reload data after changes.
- Synchronous request handling is adequate for single-user demo workloads; concurrent
  load requires an async refactor.

## Trade-Offs

- **Graph-first, not graph-only:** when no entity is resolved from the question, the
  system falls back to unconstrained vector search. Recall is preserved; precision is
  lower without graph anchoring.
- **Local embeddings:** `sentence-transformers` operates fully offline. Swap via
  `EmbeddingProvider` without changing retrieval logic.
- **Fixed graph schema:** the domain model is compiled into Cypher templates. Extending
  the schema requires adding new templates, not a configuration change.
- **Deterministic answers:** the answer builder produces consistent, auditable output
  for identical inputs. It does not hallucinate but also does not paraphrase.

## Non-Goals

- Multi-tenancy or per-user data isolation
- Real-time or streaming ingestion
- Cloud deployment or managed Neo4j Aura support
- Frontend UI or interactive query console
