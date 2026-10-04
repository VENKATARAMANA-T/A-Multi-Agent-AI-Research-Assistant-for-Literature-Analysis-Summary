"""Test configuration.

Environment variables are set *before* any application import so the settings
singleton picks up an isolated data directory, the offline embedder, and a
disabled Neo4j connection. Nothing in the suite touches the network.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="researchcompass-tests-"))

os.environ.update(
    {
        "ENVIRONMENT": "test",
        "DATA_DIR": str(_TEST_ROOT),
        "UPLOAD_DIR": str(_TEST_ROOT / "uploads"),
        "CHROMA_DIR": str(_TEST_ROOT / "chroma"),
        "REPORT_DIR": str(_TEST_ROOT / "reports"),
        "DATABASE_URL": f"sqlite:///{(_TEST_ROOT / 'test.db').as_posix()}",
        "EMBEDDING_OFFLINE_FALLBACK": "true",
        "NEO4J_ENABLED": "false",
        "GOOGLE_API_KEY": "",
        # OCR is off by default so the bulk of the suite stays fast; the tests
        # that exercise it turn it on via the `ocr_enabled` fixture.
        "OCR_ENABLED": "false",
        # No throttling against the fake client — otherwise every mocked call
        # would sleep for the real rate-limit interval.
        "LLM_REQUESTS_PER_MINUTE": "0",
        "JWT_SECRET": "test-secret-not-a-real-key",
        "CHUNK_SIZE": "600",
        "CHUNK_OVERLAP": "80",
        "LOG_LEVEL": "WARNING",
    }
)

import fitz  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import engine, init_db  # noqa: E402
from app.services.graph_store import InMemoryGraphStore, reset_graph_store  # noqa: E402
from app.services.llm import LLMError, set_llm  # noqa: E402
from app.services.vector_store import get_vector_store, reset_vector_store  # noqa: E402

PAPER_ONE = {
    "title": "Sparse Attention Transformers for Long Document Summarization",
    "authors": "Ada Lovelace, Alan Turing, Grace Hopper",
    "abstract": (
        "Transformer models struggle with long documents because self-attention scales "
        "quadratically with sequence length. We introduce SparseSum, a sparse attention "
        "architecture that reduces the cost to linear time while preserving accuracy. "
        "On the arXiv and PubMed summarization benchmarks SparseSum reaches 44.1 ROUGE-L, "
        "outperforming the dense baseline by 2.3 points while using 41 percent less memory."
    ),
    "sections": [
        (
            "1 Introduction",
            "Abstractive summarization of scientific articles requires models that read "
            "thousands of tokens. Dense transformers cannot do this efficiently. We study "
            "whether sparsity recovers the lost accuracy at a fraction of the cost.",
        ),
        (
            "2 Related Work",
            "Longformer and BigBird introduced windowed attention patterns. Our work extends "
            "these ideas with a learned sparsity mask rather than a fixed pattern.",
        ),
        (
            "3 Method",
            "SparseSum computes a routing distribution over key blocks and attends only to the "
            "top-k blocks per query. The routing network is trained jointly with the summarizer "
            "using a straight-through estimator.",
        ),
        (
            "4 Experiments",
            "We evaluate on the arXiv and PubMed datasets using ROUGE-1, ROUGE-2 and ROUGE-L. "
            "All models are trained for 100k steps on eight A100 GPUs using PyTorch.",
        ),
        (
            "5 Results",
            "SparseSum reaches 44.1 ROUGE-L on arXiv and 41.8 ROUGE-L on PubMed, improving over "
            "the dense baseline by 2.3 and 1.9 points respectively.",
        ),
        (
            "6 Limitations",
            "Our evaluation covers English scientific text only. We did not test on legal or "
            "clinical corpora, and we did not measure factual consistency with human raters.",
        ),
        (
            "7 Conclusion",
            "Learned sparsity makes long-document summarization practical. Future work should "
            "extend the approach to multilingual corpora and evaluate factuality directly.",
        ),
    ],
}

PAPER_TWO = {
    "title": "Retrieval Augmented Generation for Clinical Question Answering",
    "authors": "Barbara Liskov, Edsger Dijkstra",
    "abstract": (
        "Clinical question answering demands answers grounded in the medical literature. "
        "We present ClinRAG, a retrieval-augmented generation pipeline over PubMed abstracts. "
        "ClinRAG achieves 78.4 accuracy on MedQA, a 6.1 point gain over a closed-book baseline, "
        "and reduces unsupported claims by half according to clinician review."
    ),
    "sections": [
        (
            "1 Introduction",
            "Large language models hallucinate clinical facts. Retrieval grounding is the "
            "standard mitigation, but its effectiveness in medicine remains under-measured.",
        ),
        (
            "3 Method",
            "ClinRAG embeds PubMed abstracts with a biomedical sentence encoder, retrieves the "
            "top eight passages per question, and conditions the generator on those passages.",
        ),
        (
            "5 Results",
            "On MedQA ClinRAG reaches 78.4 accuracy. Clinician review of 200 answers found "
            "unsupported claims in 11 percent of outputs, down from 23 percent closed-book.",
        ),
        (
            "6 Limitations",
            "We evaluate on English MedQA only and do not measure performance on rare diseases "
            "or on languages other than English. No prospective clinical trial was conducted.",
        ),
    ],
}


def build_pdf(path: Path, spec: dict) -> Path:
    """Render a synthetic research paper to a real PDF with a text layer."""
    document = fitz.open()
    page = document.new_page()

    cursor = 72
    page.insert_text((72, cursor), spec["title"], fontsize=19, fontname="helv")
    cursor += 34
    page.insert_text((72, cursor), spec["authors"], fontsize=11, fontname="helv")
    cursor += 30
    page.insert_text((72, cursor), "Abstract", fontsize=12, fontname="hebo")
    cursor += 18

    def write_paragraph(text: str, start: float, page_ref) -> tuple[float, object]:
        words = text.split()
        line: list[str] = []
        for word in words:
            line.append(word)
            if len(" ".join(line)) > 88:
                page_ref.insert_text((72, start), " ".join(line), fontsize=10, fontname="helv")
                start += 14
                line = []
                if start > 740:
                    page_ref = document.new_page()
                    start = 72
        if line:
            page_ref.insert_text((72, start), " ".join(line), fontsize=10, fontname="helv")
            start += 14
        return start, page_ref

    cursor, page = write_paragraph(spec["abstract"], cursor, page)
    cursor += 10

    for heading, body in spec["sections"]:
        if cursor > 700:
            page = document.new_page()
            cursor = 72
        page.insert_text((72, cursor), heading, fontsize=12, fontname="hebo")
        cursor += 18
        # Repeat the body so chunking has enough material to split.
        cursor, page = write_paragraph(" ".join([body] * 3), cursor, page)
        cursor += 12

    document.save(path)
    document.close()
    return path


class FakeGemini:
    """Deterministic stand-in for GeminiClient; records every prompt it sees."""

    def __init__(self, responses: dict[str, dict] | None = None, fail: bool = False) -> None:
        self.responses = responses or {}
        self.fail = fail
        self.calls: list[dict] = []
        self.model = "fake-gemini"
        self.available = True

    def _match(self, system_instruction: str) -> dict:
        for key, payload in self.responses.items():
            if key.lower() in (system_instruction or "").lower():
                return payload
        return {}

    def generate(self, prompt, system_instruction=None, **kwargs):
        from app.services.llm import LLMResponse

        self.calls.append({"prompt": prompt, "system": system_instruction})
        if self.fail:
            raise LLMError("simulated model failure")
        return LLMResponse(text="## Introduction\nGenerated narrative.", model=self.model)

    def generate_json(self, prompt, system_instruction=None, schema=None, **kwargs):
        self.calls.append({"prompt": prompt, "system": system_instruction, "schema": schema})
        if self.fail:
            raise LLMError("simulated model failure")
        return self._match(system_instruction or "")


AGENT_RESPONSES = {
    "Question Answering Agent": {
        "answer": "SparseSum reaches 44.1 ROUGE-L on arXiv [S1].",
        "confidence": "high",
        "supporting_sources": ["S1"],
        "caveats": ["English scientific text only."],
        "follow_up_questions": ["How does routing affect latency?"],
    },
    "Summarization Agent of ResearchCompass, working in multi": {
        "overview": "Both papers make long-context generation practical.",
        "themes": [{"name": "Efficiency", "description": "Sparsity and retrieval both cut cost.", "papers": ["SparseSum"]}],
        "agreements": ["Grounding improves faithfulness."],
        "disagreements": [],
        "methodological_trends": ["Move from dense to sparse attention."],
        "shared_datasets": ["PubMed"],
    },
    "Summarization Agent of ResearchCompass. You produce": {
        "tldr": "SparseSum makes long-document summarization linear-time.",
        "problem": "Quadratic attention cost.",
        "approach": "Learned block sparsity with top-k routing.",
        "key_findings": ["44.1 ROUGE-L on arXiv"],
        "contributions": ["Learned sparsity mask"],
        "limitations": ["English only"],
        "future_work": ["Multilingual corpora"],
        "keywords": ["sparse attention", "summarization"],
    },
    "Information Extraction Agent": {
        "datasets": [{"name": "arXiv", "domain": "scientific"}, {"name": "PubMed"}],
        "methods": [{"name": "SparseSum", "type": "architecture"}],
        "metrics": [{"name": "ROUGE-L", "value": "44.1", "dataset": "arXiv"}],
        "tasks": ["summarization"],
        "research_questions": ["Does learned sparsity preserve accuracy?"],
        "limitations": ["No factuality evaluation"],
        "tools": ["PyTorch"],
        "cited_works": ["Beltagy et al. (2020)"],
    },
    "Research Gap Agent": {
        "landscape_summary": "Efficiency is solved; faithfulness is not.",
        "gaps": [
            {
                "title": "No multilingual evaluation",
                "description": "Every paper evaluates on English only.",
                "evidence": ["Our evaluation covers English scientific text only [S1]"],
                "category": "empirical",
                "severity": "high",
                "proposed_direction": "Replicate on a multilingual scientific corpus.",
                "related_papers": ["S1", "S2"],
            },
            {
                "title": "Factual consistency unmeasured",
                "description": "Neither paper measures factuality with human raters at scale.",
                "evidence": ["we did not measure factual consistency [S1]"],
                "category": "evaluation",
                "severity": "medium",
                "proposed_direction": "Run a blinded clinician annotation study.",
                "related_papers": ["S2"],
            },
        ],
        "underexplored_intersections": ["Sparse attention combined with retrieval grounding"],
        "open_questions": ["Does sparsity harm faithfulness?"],
    },
    # The Verification Agent makes two calls with two different system prompts;
    # these keys are the phrase that distinguishes them.
    "break a piece of generated text": {
        "claims": [
            {"text": "SparseSum reaches 44.1 ROUGE-L on arXiv.", "type": "numeric", "checkable": True},
            {
                "text": "SparseSum was evaluated on Portuguese legal documents.",
                "type": "factual",
                "checkable": True,
            },
        ]
    },
    "decide whether each claim is supported": {
        "judgements": [
            {
                "claim_index": 0,
                "verdict": "supported",
                "confidence": "high",
                "evidence_quote": "SparseSum reaches 44.1 ROUGE-L on arXiv",
                "explanation": "The results section states exactly this.",
            },
            {
                "claim_index": 1,
                "verdict": "unsupported",
                "confidence": "high",
                "evidence_quote": "",
                "explanation": "No passage mentions legal documents in any language.",
            },
        ]
    },
    "Knowledge Graph Agent": {
        "entities": [
            {"name": "SparseSum", "type": "Method", "description": "Sparse attention summarizer"},
            {"name": "arXiv", "type": "Dataset", "description": "Scientific paper corpus"},
            {"name": "ROUGE-L", "type": "Metric", "description": "Summarization metric"},
        ],
        "relations": [
            {"source": "SparseSum", "target": "arXiv", "type": "EVALUATED_ON", "evidence": "evaluated on arXiv"},
            {"source": "ROUGE-L", "target": "SparseSum", "type": "MEASURES", "evidence": "44.1 ROUGE-L"},
            {"source": "SparseSum", "target": "Hallucinated Entity", "type": "USES", "evidence": "n/a"},
        ],
    },
}


@pytest.fixture(scope="session", autouse=True)
def _prepare_environment():
    settings.ensure_directories()
    init_db()
    yield
    engine.dispose()
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset the database, vector store and graph between tests."""
    from sqlmodel import SQLModel

    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)

    reset_vector_store()
    get_vector_store().reset()

    reset_graph_store()
    graph_file = Path(settings.data_dir) / "knowledge_graph.json"
    graph_file.unlink(missing_ok=True)

    from app.services import ocr

    ocr.reset_engine()

    set_llm(None)
    yield
    set_llm(None)


# Dense lines that fill each column, as a real two-column paper does. Short
# lines separated by wide whitespace get merged into one detection by the OCR
# text detector, which is an artifact of a toy layout rather than a real case.
TWO_COLUMN_LEFT = [
    "Transformer models struggle with long documents",
    "because self-attention scales quadratically with",
    "sequence length. We introduce SparseSum, a",
    "sparse attention architecture that reduces the",
    "cost to linear time while preserving accuracy.",
    "On the arXiv and PubMed benchmarks SparseSum",
]
TWO_COLUMN_RIGHT = [
    "reaches 44.1 ROUGE-L, outperforming the dense",
    "baseline by 2.3 points while using 41 percent",
    "less memory. Our evaluation covers English",
    "scientific text only. We did not test on legal",
    "or clinical corpora, and we did not measure",
    "factual consistency with human raters.",
]


def build_scanned_pdf(path: Path, spec: dict, dpi: int = 200, columns: int = 1) -> Path:
    """Build a PDF with *no text layer* — a genuine scan.

    The page is composed normally, rendered to a raster image, and that image is
    placed on a fresh page. Extracting text from the result yields nothing, which
    is exactly the situation OCR exists to handle. Testing against a real
    image-only PDF is the only way to know the OCR path actually works.

    Font sizes are chosen so every line fits within the page width: PyMuPDF's
    `insert_text` does not wrap, and text past the edge is silently clipped.
    """
    source = fitz.open()
    page = source.new_page()  # 595 x 842 pt

    if columns == 2:
        page.insert_text((50, 70), spec["title"][:48], fontsize=14, fontname="helv")
        for i in range(len(TWO_COLUMN_LEFT)):
            page.insert_text((50, 110 + i * 16), TWO_COLUMN_LEFT[i], fontsize=9, fontname="helv")
            page.insert_text((310, 110 + i * 16), TWO_COLUMN_RIGHT[i], fontsize=9, fontname="helv")
    else:
        cursor = 90
        page.insert_text((50, cursor), spec["title"], fontsize=13, fontname="helv")
        cursor += 34
        page.insert_text((50, cursor), spec["authors"], fontsize=10, fontname="helv")
        cursor += 30
        page.insert_text((50, cursor), "Abstract", fontsize=11, fontname="helv")
        cursor += 22
        for sentence in spec["abstract"].split(". ")[:4]:
            page.insert_text((50, cursor), sentence.strip()[:72], fontsize=9, fontname="helv")
            cursor += 18

    pixmap = page.get_pixmap(dpi=dpi)
    png = pixmap.tobytes("png")
    source.close()

    scanned = fitz.open()
    target = scanned.new_page(width=pixmap.width * 72 / dpi, height=pixmap.height * 72 / dpi)
    target.insert_image(target.rect, stream=png)
    scanned.save(path)
    scanned.close()
    return path


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    return build_pdf(tmp_path / "sparse_sum.pdf", PAPER_ONE)


@pytest.fixture
def scanned_pdf(tmp_path) -> Path:
    """An image-only PDF: PyMuPDF finds no text in it at all."""
    return build_scanned_pdf(tmp_path / "scanned.pdf", PAPER_ONE)


@pytest.fixture
def scanned_two_column_pdf(tmp_path) -> Path:
    return build_scanned_pdf(tmp_path / "scanned_2col.pdf", PAPER_ONE, columns=2)


# Body text at 9pt with 18pt leading — the spacing at which the OCR detector's
# default polygon expansion silently dropped one line in four.
DENSE_LINES = [
    "We fine-tune a wav2vec 2.0 encoder on the UA-Speech corpus",
    "and report an accuracy of 87.4 percent across 4 severity levels",
    "SparseSum reaches 44.1 ROUGE-L, a gain of 2.3 points over BERT",
    "Trained for 100k steps on 8 A100 GPUs using PyTorch 2.1",
]


@pytest.fixture
def scanned_dense_pdf(tmp_path) -> Path:
    """An image-only page of tightly-spaced body text."""
    source = fitz.open()
    page = source.new_page()
    for i, row in enumerate(DENSE_LINES):
        page.insert_text((50, 100 + i * 18), row, fontsize=9, fontname="helv")
    pixmap = page.get_pixmap(dpi=200)
    png = pixmap.tobytes("png")
    source.close()

    scanned = fitz.open()
    target = scanned.new_page(width=pixmap.width * 72 / 200, height=pixmap.height * 72 / 200)
    target.insert_image(target.rect, stream=png)
    path = tmp_path / "scanned_dense.pdf"
    scanned.save(path)
    scanned.close()
    return path


def build_paper_with_figures(path: Path) -> Path:
    """A PDF with a real vector bar chart, a ruled table, and their captions.

    Charts in academic PDFs are vector drawings rather than embedded images, so
    the chart here is drawn with actual lines and rectangles — extracting a
    pasted bitmap would not exercise the same code path.
    """
    document = fitz.open()
    page = document.new_page()  # 595 x 842

    page.insert_text((50, 60), "Dysarthria Severity Classification", fontsize=14, fontname="helv")
    page.insert_text(
        (50, 90),
        "We evaluate three feature sets on the UA-Speech corpus across four severity levels.",
        fontsize=9,
        fontname="helv",
    )

    # --- a bar chart drawn as vectors ---------------------------------------
    chart_left, chart_bottom, chart_height = 70, 330, 190
    page.draw_line(fitz.Point(chart_left, chart_bottom), fitz.Point(chart_left, chart_bottom - chart_height))
    page.draw_line(fitz.Point(chart_left, chart_bottom), fitz.Point(chart_left + 300, chart_bottom))
    for index, (name, value) in enumerate([("MFCC", 0.62), ("Spectro", 0.78), ("wav2vec", 0.93)]):
        x = chart_left + 30 + index * 90
        height = chart_height * value
        page.draw_rect(fitz.Rect(x, chart_bottom - height, x + 55, chart_bottom), fill=(0.2, 0.4, 0.8))
        page.insert_text((x + 4, chart_bottom + 14), name, fontsize=8, fontname="helv")
        page.insert_text((x + 10, chart_bottom - height - 5), f"{value:.2f}", fontsize=8, fontname="helv")
    page.insert_text((chart_left - 45, chart_bottom - 90), "Accuracy", fontsize=8, fontname="helv")

    page.insert_text(
        (50, 370),
        "Figure 1. Classification accuracy by feature set on the UA-Speech corpus.",
        fontsize=9,
        fontname="helv",
    )

    # --- a ruled table ------------------------------------------------------
    page.insert_text(
        (50, 420),
        "Table 1. Severity level classification accuracy per feature set.",
        fontsize=9,
        fontname="helv",
    )
    rows = [
        ["Feature", "ACC", "SE", "SP"],
        ["MFCC", "0.62", "0.60", "0.64"],
        ["Spectrogram", "0.78", "0.77", "0.79"],
        ["wav2vec", "0.93", "0.92", "0.94"],
    ]
    top, row_h, col_w, left = 440, 20, 90, 50
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = fitz.Rect(left + c * col_w, top + r * row_h, left + (c + 1) * col_w, top + (r + 1) * row_h)
            page.draw_rect(rect)
            page.insert_text((rect.x0 + 5, rect.y1 - 6), cell, fontsize=8, fontname="helv")

    page.insert_text(
        (50, 570),
        "These results indicate that self-supervised representations outperform the "
        "hand-crafted baselines by a substantial margin across every severity level "
        "that we evaluated in this study, which is consistent with prior work.",
        fontsize=9,
        fontname="helv",
    )

    document.save(path)
    document.close()
    return path


@pytest.fixture
def figures_pdf(tmp_path) -> Path:
    return build_paper_with_figures(tmp_path / "with_figures.pdf")


@pytest.fixture
def figures_enabled(monkeypatch):
    monkeypatch.setattr(settings, "figures_enabled", True)
    yield


class FakeVisionClient:
    """Stands in for the vision model; records the images it was shown."""

    model = "fake-vision"
    available = True

    def __init__(self, payload: dict | None = None, fail: bool = False):
        self.calls: list[dict] = []
        self.fail = fail
        self.payload = payload or {
            "chart_type": "bar",
            "description": "A bar chart comparing classification accuracy for three feature sets.",
            "axes": {"x_label": "Feature set", "y_label": "Accuracy", "y_range": "0 to 1"},
            "series": [
                {"name": "Accuracy", "trend": "increases from MFCC to wav2vec",
                 "notable_values": ["MFCC 0.62", "wav2vec 0.93"]}
            ],
            "findings": ["wav2vec reaches 0.93 accuracy, the highest of the three."],
            "takeaway": "Self-supervised features outperform hand-crafted baselines.",
            "datasets": ["UA-Speech"],
            "methods": ["wav2vec", "MFCC"],
            "metrics": ["Accuracy"],
            "readable": True,
        }

    def generate_from_image(self, prompt, image_bytes, mime_type="image/png", **kwargs):
        import json

        from app.services.llm import LLMError

        self.calls.append({"prompt": prompt, "bytes": len(image_bytes), "mime": mime_type})
        if self.fail:
            raise LLMError("simulated vision failure")
        return json.dumps(self.payload)


@pytest.fixture
def fake_vision():
    from app.services.llm import set_llm

    client = FakeVisionClient()
    set_llm(client)
    return client


@pytest.fixture
def blank_pdf(tmp_path) -> Path:
    """No text and no images — must not be sent to OCR."""
    document = fitz.open()
    document.new_page()
    path = tmp_path / "blank.pdf"
    document.save(path)
    document.close()
    return path


@pytest.fixture
def ocr_enabled(monkeypatch):
    """Turn real OCR on for a test (the suite disables it by default).

    `settings` is a single shared instance, so patching it here reaches every
    module that imported it.
    """
    from app.services import ocr

    monkeypatch.setattr(settings, "ocr_enabled", True)
    ocr.reset_engine()
    yield
    ocr.reset_engine()


@pytest.fixture
def second_pdf(tmp_path) -> Path:
    return build_pdf(tmp_path / "clin_rag.pdf", PAPER_TWO)


@pytest.fixture
def fake_llm() -> FakeGemini:
    client = FakeGemini(AGENT_RESPONSES)
    set_llm(client)
    return client


def seed_user_id() -> str:
    """The id of the bootstrap account, creating it if a test dropped the table.

    Tests ingest through the service layer rather than the API, so they have to
    name an owner themselves; a paper with no owner belongs to nobody and is
    invisible to every scoped query.
    """
    from app.database import session_scope
    from app.services.bootstrap import ensure_seed_user

    with session_scope() as session:
        return ensure_seed_user(session).id


@pytest.fixture
def owner_id() -> str:
    return seed_user_id()


@pytest.fixture
def anon_client() -> TestClient:
    """A client with no credentials, for checking that endpoints refuse one."""
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client() -> TestClient:
    """Signed in as the seed account.

    Authenticating here rather than in each test keeps the suite about what the
    endpoints do; `anon_client` covers the unauthenticated case explicitly.
    """
    from app.main import app
    from app.services.bootstrap import SEED_PASSWORD, SEED_USERNAME

    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/auth/login",
            json={"identifier": SEED_USERNAME, "password": SEED_PASSWORD},
        )
        assert response.status_code == 200, response.text
        test_client.headers["Authorization"] = f"Bearer {response.json()['access_token']}"
        yield test_client


@pytest.fixture
def memory_graph_store(tmp_path) -> InMemoryGraphStore:
    return InMemoryGraphStore(path=tmp_path / "graph.json")
