# ResearchCompass

A multi-agent AI research assistant for literature analysis. Upload research papers and
ResearchCompass extracts their text, chunks and embeds it, indexes it in a vector store, and
turns six specialised agents loose on it — answering questions with citations, summarising
across papers, extracting datasets/methods/metrics, building a knowledge graph, identifying
research gaps, and exporting a literature review as Markdown or PDF.

```
                    ┌──────────────────────── React + Vite ────────────────────────┐
                    │ Upload · Library · Ask · Summaries · Extraction · Gaps ·     │
                    │ Knowledge Graph · Reports                                    │
                    └───────────────────────────┬─────────────────────────────────┘
                                                │ REST
                    ┌───────────────────────────┴─────────────────────────────────┐
                    │                    FastAPI backend                          │
                    │                                                             │
   PDF ─▶ PyMuPDF ─▶ chunking ─▶ MiniLM embeddings ─▶ ChromaDB                    │
                    │                                    │                        │
                    │                    ┌───────────────┴──────────────┐         │
                    │                    │   LangGraph orchestration     │        │
                    │                    │  retrieval · summarization ·  │        │
                    │                    │  extraction · QA · gaps ·     │─▶ Gemini│
                    │                    │  knowledge graph              │        │
                    │                    └───────────────┬──────────────┘         │
                    │                                    │                        │
                    │              SQLite (metadata)     └──▶ Neo4j (graph)       │
                    └─────────────────────────────────────────────────────────────┘
```

---

## Quick start

### Docker (everything, including Neo4j)

```bash
cp .env.example .env
```

Put your Gemini key in `.env` (`GOOGLE_API_KEY=...`), then:

```bash
docker compose up --build
```

- Frontend — http://localhost:8080
- API docs — http://localhost:8000/docs
- Neo4j Browser — http://localhost:7474 (`neo4j` / `researchcompass`)

The first build downloads the embedding model into a named volume, so it takes a few minutes;
subsequent starts are fast.

### Local development

**Backend**

```bash
cd backend
python -m venv .venv
```

```bash
.venv\Scripts\pip install -r requirements-dev.txt
```

Copy `backend/.env.example` to `backend/.env` and add your key, then:

```bash
.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

Vite serves on http://localhost:5173 and proxies `/api` to port 8000.

**Neo4j only** (optional — there is a working fallback, see below):

```bash
docker compose up -d neo4j
```

---

## How it works

### Step 1 — Upload
`POST /api/papers/upload` accepts up to 20 PDFs. Each file is validated (magic bytes, size
limit) and de-duplicated by SHA-256, so re-uploading the same paper is a no-op.

### Step 2 — Text and metadata extraction
`app/services/pdf_extract.py` uses **PyMuPDF** to pull the text layer page by page. Metadata is
recovered from three sources in order of trust: the PDF's own Info dictionary, layout
heuristics (the largest-font block on page 1 is almost always the title), and regex sweeps for
DOI, arXiv id, year, abstract and keywords. The full text is also sliced into canonical
sections (`introduction`, `method`, `results`, `limitations`, …), which later lets the system
fit a long paper into a prompt by dropping the reference list rather than truncating blindly.

Scanned PDFs with no text layer are rejected with a clear message — they need OCR first.

### Step 3 — Chunking
**LangChain's `RecursiveCharacterTextSplitter`** with configurable size/overlap. Each chunk
carries its page range and section label, so every later citation can point at a real location
in the source PDF. A chunk that *opens* with a heading is attributed to that section, not the
preceding one.

### Step 4 — Embeddings
**sentence-transformers `all-MiniLM-L6-v2`** by default. If the model cannot be loaded (no
network, no torch, air-gapped CI), the system falls back to a deterministic hashed-bigram
embedder so the entire pipeline still runs; `/api/health` reports `embeddings.degraded: true`
and the UI shows a banner. Because two backends can share a dimension while living in different
vector spaces, the Chroma collection is stamped with the backend that wrote it and a mismatch
is loudly flagged rather than silently returning nonsense.

### Step 5 — Vector store
**ChromaDB** persistent client with cosine distance. Embeddings are computed by our own
embedder and handed to Chroma explicitly, so the backend stays swappable.

### Step 6 — Multi-agent processing
**LangGraph** orchestrates the agents as a stateful graph over one shared blackboard:

```
router ─┬─ retrieve ─ qa ───────────────────── END
        └─ load ─┬─ summarize ──────────────── END
                 ├─ multi_summarize ─┐         END
                 ├─ extract ─────────┤         END
                 ├─ gap ─────────────┤         END
                 └─ build_graph ─────┘         END
```

| Agent | Node | What it does |
|---|---|---|
| Retrieval | `retrieve` | Multi-query expansion + reciprocal-rank fusion over ChromaDB |
| Summarization | `summarize` / `multi_summarize` | Single-paper structure, or cross-paper themes, agreements and contradictions |
| Information Extraction | `extract` | Datasets, methods, metrics, tasks, tools, limitations — one call per paper, then aggregated corpus-wide |
| Question Answering | `qa` | Grounded RAG; `[S#]` markers are resolved back to real chunks with page numbers |
| Research Gap | `gap` | Limitations, missing comparisons, under-explored intersections, each with a concrete proposed study |
| Knowledge Graph | `build_graph` | Typed entities and relations; hallucinated edge endpoints are dropped |

The `review` intent chains `load → multi_summarize → extract → gap → build_graph` in one
stateful run — this is what report generation uses.

Every run is written to an `agent_runs` audit table with its full execution trace, and the UI
can expand that trace on any result.

**Failure isolation.** A failing agent never takes down the run: the error is recorded in
`errors`/`trace`, the workflow continues, and the response comes back with `status: "partial"`.
So a quota error during summarisation still leaves you the retrieved evidence and the graph.

### Step 7 — Knowledge graph
Extracted entities and relations are merged corpus-wide (canonical name keys collapse
`BERT` / `bert` / `The BERT` into one node) and persisted to **Neo4j**. If Neo4j is unreachable,
a JSON-file-backed in-memory store takes over transparently so the visualisation still works;
the active backend is reported by `/api/health` and shown in the UI. The frontend renders it
with **react-force-graph-2d**, with type filters and a minimum-degree slider.

### Step 8 — Reports
A literature review with a corpus table, synthesis, per-paper comparison table, corpus-wide
dataset/method/metric tables, research gaps, a knowledge-graph summary and references —
rendered as Markdown and exported to PDF with ReportLab. If any agent failed, the report says
so at the top and lists why, instead of silently showing empty sections.

---

## API

Interactive docs at `/docs`. Highlights:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/papers/upload` | Upload and index PDFs |
| `GET` | `/api/papers` | List papers (filter by `status`, `q`) |
| `GET` | `/api/papers/{id}/chunks` | Inspect a paper's chunks |
| `POST` | `/api/papers/{id}/reindex` | Re-run ingestion |
| `DELETE` | `/api/papers/{id}` | Delete a paper and every derived artefact |
| `POST` | `/api/papers/search` | Semantic search (no LLM) |
| `POST` | `/api/agents/ask` | Question Answering agent |
| `POST` | `/api/agents/summarize` | Summarization agent (`scope`: `single` \| `multi`) |
| `POST` | `/api/agents/extract` | Information Extraction agent |
| `POST` | `/api/agents/gaps` | Research Gap agent |
| `POST` | `/api/agents/graph/build` | Knowledge Graph agent |
| `POST` | `/api/agents/review` | Full pipeline in one run |
| `GET` | `/api/agents/runs` | Audit trail |
| `GET` | `/api/graph` | Graph for visualisation (filter by paper, type, degree) |
| `POST` | `/api/reports` | Generate a literature review |
| `GET` | `/api/reports/{id}/pdf` | Download PDF |
| `GET` | `/api/health` | Which backends are actually active |

An empty `paper_ids` array means "the whole indexed corpus".

---

## Configuration

Backend settings come from `backend/.env` (see `backend/.env.example`).

| Variable | Default | Notes |
|---|---|---|
| `GOOGLE_API_KEY` | — | Required for the agents. Ingestion and search work without it. |
| `GEMINI_MODEL` | `gemini-flash-latest` | Pin a dated id for reproducibility |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | |
| `EMBEDDING_OFFLINE_FALLBACK` | `false` | Force the hashing embedder |
| `NEO4J_ENABLED` | `true` | `false` uses the in-memory graph store |
| `NEO4J_URI` | `bolt://localhost:7687` | |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `200` | |
| `RETRIEVAL_TOP_K` | `8` | |
| `MAX_UPLOAD_MB` | `50` | |

### A note on Gemini free-tier quota

The free tier caps **requests per day per model** (as low as 20 on some projects). A full
report runs one call per agent plus one per paper for extraction and graph building, so it can
exhaust a small daily allowance in a single click. ResearchCompass distinguishes per-minute
rate limits (retried with exponential backoff) from hard daily caps (surfaced immediately —
retrying cannot clear them), and condenses the provider's multi-kilobyte error into one
actionable line. If you hit the cap, enable billing or wait for the reset.

---

## Testing

```bash
cd backend
.venv\Scripts\python -m pytest
```

97 tests, no network access required — the suite runs against an isolated temp directory with
the offline embedder, Neo4j disabled, and a mocked Gemini client.

Coverage:

- **`test_ingestion_pipeline.py`** — PDF text/metadata/section extraction, chunk sizing and
  page-and-section attribution, embedder determinism and relative similarity, vector store
  add/search/scope/delete
- **`test_agents.py`** — graph wiring, per-intent routing, citation-marker resolution,
  per-paper extraction and aggregation, gap severity ordering, hallucinated-edge rejection,
  the chained `review` pipeline, failure isolation, run persistence
- **`test_api.py`** — the full HTTP surface, including duplicate detection, invalid PDFs,
  validation errors, graph pruning on paper delete, and report Markdown/PDF output
- **`test_graph_and_reports.py`** — graph persistence, idempotent merging, orphan pruning,
  restart durability; Markdown table/section rendering and PDF export
- **`test_embedding_guard.py`** — the vector-space mismatch guard
- **`test_llm_errors.py`** — retryable vs terminal provider errors, error condensation
- **`test_output_cleanup.py`** — citation markers stripped from entity names, incomplete
  reports explaining themselves

Tests build real PDFs with PyMuPDF rather than using fixtures, so extraction is exercised
against genuine PDF structure.

### Manual verification

1. Upload 3–5 related papers through the frontend.
2. Ask a question and check the cited page numbers against the source PDF.
3. Run the Research Gap agent and confirm each gap traces to a real limitation.
4. Open the knowledge graph and click through nodes.
5. Generate a report and export the PDF.

---

## Project layout

```
backend/
  app/
    agents/        prompts, per-agent nodes, LangGraph workflow, output cleanup
    api/           routers: papers, agents, graph, reports, system
    services/      pdf_extract, chunking, embeddings, vector_store, llm,
                   graph_store, ingestion, report
    config.py      pydantic-settings
    models.py      SQLModel tables
    schemas.py     API request/response models
    main.py        app wiring, CORS, error handlers
  tests/
frontend/
  src/
    api/           fetch client
    components/    Layout, PaperPicker, AgentTrace, shared UI
    context/       corpus + selection state
    pages/         Dashboard, Upload, Library, Ask, Summaries,
                   Extraction, Gaps, GraphView, Reports
docker-compose.yml
```

---

## Known limits

- **Scanned PDFs** are rejected — there is no OCR stage.
- **Metadata heuristics** handle common two-column academic layouts well; unusual title pages
  may yield a wrong title or author list. Every field is editable via reindex, and the LLM
  extraction agent can recover what the heuristics miss.
- **Ingestion is synchronous.** A 20-file upload holds the request open. A job queue would be
  the right answer for larger corpora.
- **The knowledge graph merges on normalised names**, so genuinely distinct entities sharing a
  name will collapse into one node.
