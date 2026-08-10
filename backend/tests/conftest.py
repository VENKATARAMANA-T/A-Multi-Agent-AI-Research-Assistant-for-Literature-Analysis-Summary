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

    set_llm(None)
    yield
    set_llm(None)


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    return build_pdf(tmp_path / "sparse_sum.pdf", PAPER_ONE)


@pytest.fixture
def second_pdf(tmp_path) -> Path:
    return build_pdf(tmp_path / "clin_rag.pdf", PAPER_TWO)


@pytest.fixture
def fake_llm() -> FakeGemini:
    client = FakeGemini(AGENT_RESPONSES)
    set_llm(client)
    return client


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def memory_graph_store(tmp_path) -> InMemoryGraphStore:
    return InMemoryGraphStore(path=tmp_path / "graph.json")
