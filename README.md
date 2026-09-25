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

Indexing runs **in the background**: the endpoint returns a job immediately and the client
follows per-file progress over server-sent events (`/api/jobs/{id}/stream`), so a large
upload no longer holds the request open with no feedback. Pass `?wait=true` for the
synchronous result, which is convenient for scripting and tests.

### Step 2 — Text and metadata extraction
`app/services/pdf_extract.py` uses **PyMuPDF** to pull the text layer page by page. Metadata is
recovered from three sources in order of trust: the PDF's own Info dictionary, layout
heuristics (the largest-font block on page 1 is almost always the title), and regex sweeps for
DOI, arXiv id, year, abstract and keywords. The full text is also sliced into canonical
sections (`introduction`, `method`, `results`, `limitations`, …), which later lets the system
fit a long paper into a prompt by dropping the reference list rather than truncating blindly.

**OCR for scanned pages.** A page with almost no text but substantial image coverage is
rendered at 200 DPI and recognised with **RapidOCR** (ONNX, ships its own models — no system
binary, unlike Tesseract). Gemini vision is available as an alternative but costs one request
per page, so it is never selected silently.

The decision is made **per page, not per document**: papers are frequently born-digital with a
photocopied appendix, and recognising a page whose text layer is already perfect only degrades
it. A page with no text *and* no images is blank, not scanned, and is skipped.

Reading order is reconstructed rather than assumed — OCR returns boxes in detection order,
which on a two-column paper interleaves the columns into nonsense. Lines are grouped into
columns by the widest vertical gutter, then read top-to-bottom within each column.

**A tuning note worth knowing.** At the engine's default `unclip_ratio` of 1.6 — tuned for
signage, not documents — one line in four of tightly-spaced 9pt body text was **never detected
at all**. The page rendered correctly and the ink was present; the line simply vanished with no
error. `OCR_UNCLIP_RATIO` defaults to 2.0 here, which recovers every line while still keeping
adjacent columns separate. A DPI sweep showed 200 is already the accuracy optimum, so raising
it is not the fix (400 DPI was measurably *worse*). There is a regression test for this.

Results are cached on the document's content hash, so re-indexing skips recognition entirely
(measured: 4.8 s → 0.8 s). Papers, chunks and search results all carry a `source` of `native`
or `ocr`, and the UI flags OCR-derived text because recognition can introduce errors.

Anything unreadable fails with a message that says *why* — OCR disabled, no engine installed,
or a scan too low-resolution to recognise — rather than a bare rejection.

### Step 2b — Figures, charts and tables
A results chart carries the paper's actual findings, and text extraction throws all of it
away: a line plot becomes nothing, and a table becomes a run of numbers with no headings.

Academic charts are usually **vector drawings rather than embedded rasters**, so
`page.get_images()` finds almost nothing. What is reliably present is the caption, so regions
are anchored on captions instead: a figure caption sits *below* its artwork, a table caption
*above* its content. Tables are matched against PyMuPDF's table finder and never against
graphics — unioning graphics below a table caption grabs the next figure on the page.

A caption is distinguished from a cross-reference by the word after the number: *"Fig. 4
depicts…"* is a sentence in the body text, while *"Fig. 4 | ROC curves…"* names the figure.

The **Figure Agent** then reads each image with the vision model and returns the chart type,
axis labels and units, the series and their trends, values readable off the plot, and what
the figure demonstrates. Those descriptions are embedded, so *"which paper shows accuracy
dropping after epoch 50?"* becomes answerable — a question no text chunk can satisfy.
Analysed figures also join the knowledge graph, linked to the datasets and methods they
depict, and a QA answer can cite a figure and show it.

**Cost is explicit.** Extraction happens at ingestion and is free; reading a figure is one
vision request each, so it is never automatic. The UI states the exact number of requests
before you spend them, and tables whose contents were parsed directly from the PDF are marked
*skipped* — a picture of them would add nothing.

Verified against five real papers: 40 figures and tables extracted, checked by eye.

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

The `review` intent **fans out to all four agents at once** — they read the same loaded
documents and write disjoint result keys, so running them one after another only added
latency. This is what report generation uses.

Every run is written to an `agent_runs` audit table with its full execution trace, and the UI
can expand that trace on any result.

**Failure isolation.** A failing agent never takes down the run: the error is recorded in
`errors`/`trace`, the workflow continues, and the response comes back with `status: "partial"`.
So a quota error during summarisation still leaves you the retrieved evidence and the graph.

**Concurrency.** Nodes return state *deltas*, and the `trace` / `errors` / `llm_calls`
channels carry reducers, so concurrent branches merge their bookkeeping instead of
overwriting each other. `extract` and `build_graph` additionally fan out one call per paper.
A single global semaphore (`LLM_MAX_CONCURRENCY`) bounds every in-flight request, because
per-call-site limits would multiply across nested fan-outs and trip the per-minute quota.

**Response cache.** Identical requests — same model, prompt, system instruction, temperature,
token cap and schema — are served from a content-addressed SQLite cache. Measured live: a
repeat call returned in **30 ms instead of 19.4 s and consumed no quota**. On the free tier,
where the cap is requests *per day*, this is what makes re-running an analysis possible at all.
Inspect it at `/api/cache`; clear it with `DELETE /api/cache`.

### Step 7 — Knowledge graph
Extracted entities and relations are merged corpus-wide (canonical name keys collapse
`BERT` / `bert` / `The BERT` into one node) and persisted to **Neo4j**. If Neo4j is unreachable,
a JSON-file-backed in-memory store takes over transparently so the visualisation still works;
the active backend is reported by `/api/health` and shown in the UI. The frontend renders it
with **react-force-graph-2d**, with type filters and a minimum-degree slider.

### Step 7b — Reading and comparing
Five features that turn the corpus from something you query into something you work in.

**In-app PDF reader with citation highlighting.** Click any citation in an answer and the paper
opens at that page with the cited sentence highlighted. The rectangles are computed server-side
with PyMuPDF rather than against the browser's text layer: extraction normalises ligatures and
hyphenation, so the stored chunk text does not match the page byte-for-byte and needs the same
engine that produced it. Search uses short phrases and falls back progressively; a miss still
navigates to the right page and says so rather than failing silently.

**Explain this passage.** Select any text in the reader and have it explained at one of three
levels. Background the agent adds is returned separately from what the passage itself says, so
the two are never confused.

**Comparison matrix.** Define your own columns — sample size, hardware, whether code was
released — and they are filled across every paper, one call per paper. Each cell carries the
verbatim sentence supporting it, and a column the paper does not address comes back explicitly
"not reported": a fabricated sample size is worse than a blank cell. Exports to CSV.

**Discover.** Related work from **OpenAlex** (free, no API key), including a corpus-wide view
that ranks the papers several of yours point at but which you do not have. Candidates are
suggestions to review — nothing is added to the corpus automatically.

**Citation export.** BibTeX, APA, IEEE, MLA and RIS from metadata already extracted, so it
costs nothing and works without a key.

**Conversation memory.** Follow-up questions keep the earlier turns, so "why?" resolves against
what was just asked. The replayed window is bounded because each turn costs prompt tokens.

### Step 6b — GraphRAG
Vector search finds passages that *resemble* the question. Some questions have no such
passage: *"which methods were evaluated on the same dataset as wav2vec?"* is answered by the
relationships between chunks, not by any one of them, and no amount of similarity search
surfaces it.

Those are answered by walking the graph instead. The question is linked to entities, the
neighbourhood around them is collected, and the edges are serialised as facts the model reads
and cites as `[G1]`, `[G2]` — each carrying the papers that assert it, so a graph-derived
answer is as checkable as a text-derived one.

Three modes: **hybrid** (default) fuses passages with relationships, **passages** is classic
RAG, **relationships** uses the graph alone.

**Entity linking is the whole game.** A wrongly linked entity produces confidently wrong
context, so matching is deliberately conservative: exact surface forms first — including
acronym expansions, so "CNN" reaches "Convolutional Neural Network (CNN)" — then embedding
similarity only above a confidence floor. When nothing matches, graph retrieval returns
nothing and the answer rests on the text alone. Inventing a subgraph would be worse than
having none.

Facts are de-duplicated through the same entity resolution as discovery below: without it the
identical relationship appeared three times, once per spelling of the concept. Structural
edges (`MENTIONS`, `AUTHORED_BY`) are excluded — they are bookkeeping, not findings.

### Step 7c — Literature-Based Discovery
Connections the corpus implies but never states.

If one paper links A→B and a different paper links B→C, while nothing links A→C and no paper
mentions both ends, then A→C is a candidate discovery. Swanson found the fish-oil/Raynaud's
connection this way in 1986, from two literatures that did not cite each other.

Two rules decide whether a candidate is a discovery rather than an artefact:

- **Disjointness.** A and C must share no paper. If they co-occur even once, the connection is
  already known.
- **Hub suppression.** A term connected to everything — "model", "accuracy" — links everything
  to everything. Each intermediate is weighted by inverse degree, so a term shared by two
  entities counts far more than one shared by forty. Without this the ranking is meaningless.

**Entity resolution matters more than the traversal.** Papers introduce a term in full and then
use its acronym, so "Convolutional Neural Network (CNN)" and "CNN" arrive as two unconnected
nodes, nothing bridges the two papers, and every chain stays inside one paper — the method
returns nothing. Aliases are merged before traversal, which is what creates the cross-paper
bridges the technique depends on.

Two modes are offered because a small corpus has no disjoint literatures at all:
`strict` applies Swanson's criterion, `unstated` only requires that no paper states the link
directly. The **Hypothesis Agent** then judges the survivors, states each as a falsifiable
claim, and proposes a test — and is expected to reject most of them.

> **Scale matters.** The technique was designed for literatures of thousands of papers. On five
> papers the mechanism works and is tested, but the yield is small; strict mode may return
> nothing at all, and the diagnostics say exactly why.

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
| `POST` | `/api/papers/upload` | Upload PDFs; returns a job (`?wait=true` for the sync result) |
| `GET` | `/api/jobs/{id}` | Job status and per-file progress |
| `GET` | `/api/jobs/{id}/stream` | Live progress (server-sent events) |
| `GET` | `/api/figures` | List extracted figures, charts and tables |
| `GET` | `/api/figures/estimate` | How many vision requests an analysis would cost |
| `POST` | `/api/figures/analyse` | Read figures with the vision model |
| `GET` | `/api/figures/{id}/image` | The rendered figure image |
| `GET` | `/api/reader/highlight` | Locate a cited passage in its PDF |
| `POST` | `/api/reader/explain` | Explain a selected passage |
| `GET` | `/api/reader/citations` | Export a bibliography (BibTeX/APA/IEEE/MLA/RIS) |
| `POST` | `/api/matrix` | Fill custom comparison columns across papers |
| `GET` | `/api/matrix/{id}/csv` | Download a comparison as CSV |
| `GET` | `/api/discover/gaps` | Literature your corpus is missing |
| `POST` | `/api/lbd` | Implied connections (Swanson ABC model) |
| `POST` | `/api/lbd/closed` | Why are two concepts linked? |
| `GET` | `/api/lbd/hypotheses` | Saved hypotheses |
| `GET` | `/api/discover/search` | Search OpenAlex |
| `GET` | `/api/agents/conversations` | Multi-turn question threads |
| `GET` | `/api/cache` | LLM cache statistics |
| `DELETE` | `/api/cache` | Clear the LLM cache |
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
| `OCR_ENABLED` | `true` | Recover text from scanned pages |
| `OCR_ENGINE` | `auto` | `auto` \| `rapidocr` \| `gemini` \| `none` |
| `OCR_DPI` | `200` | Raise to 300 for poor scans |
| `OCR_MIN_CHARS_PER_PAGE` | `120` | Below this a page is an OCR candidate |
| `OCR_MIN_IMAGE_COVERAGE` | `0.25` | ...but only if images cover this much |
| `OCR_MAX_PAGES_PER_DOCUMENT` | `40` | Guard against a huge scan |
| `OCR_UNCLIP_RATIO` | `2.0` | Detector polygon expansion; below ~2.0 drops lines |
| `FIGURES_ENABLED` | `true` | Extract figures and tables during ingestion |
| `FIGURE_DPI` | `150` | Rendering resolution for figure images |
| `FIGURE_MAX_PER_DOCUMENT` | `60` | Extraction cap per paper |
| `FIGURE_MAX_ANALYSIS_BATCH` | `40` | Ceiling on one analysis request |
| `DISCOVERY_ENABLED` | `true` | Related-paper lookup via OpenAlex |
| `OPENALEX_CONTACT_EMAIL` | — | Optional; puts requests in OpenAlex's faster pool |
| `CONVERSATION_MEMORY_TURNS` | `6` | Previous turns replayed into a follow-up |
| `LLM_CACHE_ENABLED` | `true` | Serve identical requests from cache |
| `LLM_CACHE_TTL_DAYS` | `30` | `0` disables expiry |
| `LLM_MAX_CONCURRENCY` | `3` | Global ceiling on in-flight LLM calls |
| `LLM_REQUESTS_PER_MINUTE` | `5` | Rate limit; `0` disables. Matches the free tier |
| `LLM_TIMEOUT_SECONDS` | `90` | Per-request deadline |
| `LLM_TOTAL_RETRY_SECONDS` | `180` | Ceiling on one call's whole retry loop |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | |
| `EMBEDDING_BACKEND` | `auto` | `auto` prefers ONNX, falls back to PyTorch |
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

363 tests, no network access required — the suite runs against an isolated temp directory with
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
- **`test_llm_cache.py`** — key sensitivity (any input that changes the answer must miss),
  hit/miss behaviour, and that a repeat call never reaches the provider
- **`test_parallelism.py`** — ordered results from unordered completion, the concurrency cap,
  and that no branch's trace is lost when four of them write state at once
- **`test_jobs.py`** — background ingestion, per-file progress, partial failure, SSE streaming
- **`test_llm_timeout.py`** — the request deadline and the bounded retry window
- **`test_ocr.py`** — real recognition against image-only PDFs, the per-page decision,
  two-column reading order, caching, and graceful degradation when no engine is available
- **`test_figures.py`** — caption vs cross-reference, prose vs tabular content, region
  placement against a PDF containing a real vector chart and a ruled table, the costed
  vision pass, and figures becoming searchable and graph-connected
- **`test_migration.py`** — additive schema changes, including that existing rows receive
  the model's default rather than NULL
- **`test_reader.py`** — citation styles across five formats, LaTeX escaping and key safety,
  and real highlight coordinates located in an actual PDF
- **`test_matrix.py`** — column preparation, evidence on every cell, that a silent paper is
  marked "not reported" rather than guessed, and CSV export
- **`test_discovery_and_memory.py`** — OpenAlex parsing and de-duplication (network stubbed),
  and that a follow-up question actually receives the earlier turns
- **`test_graphrag.py`** — entity linking including acronyms, subgraph expansion, that a
  question with no matching entity falls back to text rather than returning nothing, alias
  de-duplication, and that graph context is scoped to the selected papers
- **`test_lbd.py`** — reproduces Swanson's fish-oil/Raynaud's finding in miniature, and pins
  the rules that make it a discovery: an already-stated link is rejected, shared papers are not
  disjoint literatures, hub terms are discounted below specific ones, and an acronym merges
  with its expansion

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

- **Figure regions are heuristic.** Caption anchoring is accurate on the standard layouts —
  verified by eye on five real papers — but a full-width figure on a two-column page can pull
  a neighbouring column of body text into its crop. The region is still centred on the right
  artwork; it is simply not tightly cropped. Figures whose captions the PDF does not mark in a
  recognisable way are missed entirely.

- **OCR'd text is imperfect.** Recognition is accurate on clean scans but degrades with low
  resolution, skew and handwriting, and there is no de-skew or de-noise stage. Raise `OCR_DPI`
  to 300 for poor scans. OCR-derived text is flagged throughout the UI rather than presented
  as equivalent to a native text layer.
- **Metadata heuristics** handle common two-column academic layouts well; unusual title pages
  may yield a wrong title or author list. Every field is editable via reindex, and the LLM
  extraction agent can recover what the heuristics miss.
- **The job queue is in-process.** Background ingestion runs on a thread pool inside the API
  process, so jobs do not survive a restart (they are marked failed on the next boot rather
  than resuming). A separate worker would be the right answer for a multi-instance deployment.
- **The knowledge graph merges on normalised names**, so genuinely distinct entities sharing a
  name will collapse into one node.
