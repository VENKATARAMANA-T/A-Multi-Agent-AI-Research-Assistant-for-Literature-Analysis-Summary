"""Figure, chart and table extraction, and the vision analysis pass.

The fixture PDF contains a bar chart drawn as *vector* graphics and a ruled
table — the two forms that actually appear in academic papers. A pasted bitmap
would not exercise the same code.
"""

from __future__ import annotations

import fitz
import pytest

from app.services.figures import (
    classify_caption,
    detected_tables,
    extract_figures,
    find_captions,
    looks_like_prose,
    table_to_markdown,
)


# --- caption classification --------------------------------------------------


@pytest.mark.parametrize(
    "text,kind,label",
    [
        ("Figure 1. Classification accuracy by feature set.", "figure", "Figure 1"),
        ("Fig. 3 | ROC curves for the Stage 2 model.", "figure", "Figure 3"),
        ("FIGURE 2 . Architecture of the classifier", "figure", "Figure 2"),
        ("Table 1. Severity level accuracy per feature set.", "table", "Table 1"),
        ("TABLE III", "table", "Table III"),
        ("Algorithm 1 FedAvg. The C devices are indexed by c", "algorithm", "Algorithm 1"),
    ],
)
def test_real_captions_are_recognised(text, kind, label):
    result = classify_caption(text)
    assert result is not None, f"missed caption: {text!r}"
    assert result[0] == kind
    assert result[1] == label


@pytest.mark.parametrize(
    "text",
    [
        # Sentences in the body that merely mention a figure. Treating these as
        # captions produced regions containing nothing but prose.
        "Fig. 4 depicts the average of correlation matrices of control subjects.",
        "Table III contains the univariate statistics for ASD subjects.",
        "Table 3 illustrates the key insights from our findings.",
        "Fig. 20). The overall binary prediction performance for Stage 2.",
        "Figure 2 shows the distribution of severity levels.",
        "Fig",
    ],
)
def test_in_text_references_are_rejected(text):
    assert classify_caption(text) is None, f"false positive: {text!r}"


# --- prose detection ---------------------------------------------------------


def test_body_text_is_prose():
    text = (
        "These results indicate that self-supervised representations outperform the "
        "hand-crafted baselines by a substantial margin across every severity level "
        "that we evaluated in this study, which is consistent with prior work."
    )
    assert looks_like_prose(text) is True


def test_number_dense_table_content_is_not_prose():
    """A flattened table has no sentence breaks — "92.72" has no space after the dot.

    Judging by sentence length alone classified the table itself as prose and
    dropped the region entirely.
    """
    text = (
        "Feature ACC [%] SE SP F1 Baseline features Spectrogram 92.72 0.92 0.94 0.93 "
        "Mel-spectrogram 91.48 0.91 0.92 0.92 MFCCs 89.23 0.86 0.93 0.90 "
        "wav2vec-1 93.95 0.93 0.95 0.94 wav2vec-2 91.96 0.90 0.94 0.92"
    )
    assert looks_like_prose(text) is False


def test_short_text_is_never_prose():
    assert looks_like_prose("Feature ACC SE SP") is False


# --- caption trimming --------------------------------------------------------


def test_a_short_caption_is_left_alone():
    text = "Figure 1. Classification accuracy by feature set."
    assert classify_caption(text)[2] == text


def test_a_runaway_caption_is_cut_at_a_sentence():
    """Some PDFs merge the caption and the next paragraph into one block.

    Left uncut, the "caption" arrives hundreds of characters long with half a
    literature review attached, which inflates the vision prompt and pollutes
    the text embedded for search.
    """
    from app.services.figures import CAPTION_MAX_CHARS

    body = " ".join(
        [
            "FIGURE 1 . Speech and language disorders from review articles.",
            *["The study emphasised the absence of accepted assessment tools." for _ in range(12)],
        ]
    )
    caption = classify_caption(body)[2]

    assert len(caption) <= CAPTION_MAX_CHARS + 10
    assert caption.endswith("[…]")
    assert caption.startswith("Figure 1.")


def test_trimmed_caption_keeps_whole_words():
    from app.services.figures import trim_caption

    caption = trim_caption("Figure 2. " + "alpha beta gamma delta " * 40)
    assert "  " not in caption
    assert not caption.replace(" […]", "").endswith("-")


# --- extraction --------------------------------------------------------------


def test_finds_both_the_figure_and_the_table(figures_pdf):
    items = extract_figures(str(figures_pdf))
    labels = {item.label for item in items}

    assert "Figure 1" in labels
    assert "Table 1" in labels


def test_figure_region_is_above_its_caption(figures_pdf):
    """Figure captions sit below the artwork, table captions above it."""
    items = extract_figures(str(figures_pdf))
    figure = next(item for item in items if item.label == "Figure 1")

    with fitz.open(figures_pdf) as document:
        caption = next(c for c in find_captions(document[0]) if c.label == "Figure 1")

    assert figure.bbox[3] <= caption.rect.y0 + 2, "figure region should end above its caption"


def test_figure_captures_the_chart_not_the_prose(figures_pdf):
    items = extract_figures(str(figures_pdf))
    figure = next(item for item in items if item.label == "Figure 1")

    # The chart occupies roughly y 140-345; the intro prose sits at y~90.
    assert figure.bbox[1] > 100, "region reached up into the body text"
    assert figure.height > 80
    assert figure.image_png and figure.image_png.startswith(b"\x89PNG")


def test_table_region_does_not_swallow_the_following_prose(figures_pdf):
    """A table region that keeps growing takes the rest of the page with it."""
    items = extract_figures(str(figures_pdf))
    table = next(item for item in items if item.label == "Table 1")

    # The paragraph after the table starts at y~570.
    assert table.bbox[3] < 565, f"table region ran into the prose: {table.bbox}"


def test_regions_do_not_overlap_each_other(figures_pdf):
    items = extract_figures(str(figures_pdf))
    figure = next(i for i in items if i.label == "Figure 1")
    table = next(i for i in items if i.label == "Table 1")

    assert figure.bbox[3] <= table.bbox[1] + 2, "figure and table regions overlap"


def test_images_are_rendered_by_default(figures_pdf):
    items = extract_figures(str(figures_pdf))
    assert all(item.image_png for item in items)


def test_rendering_can_be_skipped(figures_pdf):
    items = extract_figures(str(figures_pdf), render_images=False)
    assert items and all(item.image_png is None for item in items)


def test_max_figures_is_respected(figures_pdf):
    assert len(extract_figures(str(figures_pdf), max_figures=1)) == 1


def test_paper_without_captions_yields_nothing(sample_pdf):
    assert extract_figures(str(sample_pdf)) == []


# --- table validation --------------------------------------------------------


def test_implausible_detected_tables_are_dropped(figures_pdf):
    """The table finder sometimes returns one page-wide box merging everything."""
    with fitz.open(figures_pdf) as document:
        page = document[0]
        captions = find_captions(page)
        for rect, table in detected_tables(page, captions):
            assert table.col_count <= 16
            assert rect.get_area() / page.rect.get_area() <= 0.75


def test_table_markdown_escapes_pipes():
    class FakeTable:
        def extract(self):
            return [["Feature", "Note"], ["a|b", "c"]]

    markdown, rows, cols = table_to_markdown(FakeTable())
    assert "a\\|b" in markdown
    assert (rows, cols) == (2, 2)


def test_table_markdown_handles_empty():
    class Empty:
        def extract(self):
            return []

    assert table_to_markdown(Empty()) == ("", 0, 0)


# --- ingestion integration ---------------------------------------------------


@pytest.mark.usefixtures("figures_enabled")
def test_figures_are_extracted_during_ingestion(client, figures_pdf):
    response = client.post(
        "/api/papers/upload?wait=true",
        files=[("files", (figures_pdf.name, figures_pdf.read_bytes(), "application/pdf"))],
    )
    paper = response.json()["results"][0]["paper"]

    assert paper["figure_count"] >= 2

    figures = client.get("/api/figures").json()
    assert len(figures) >= 2
    assert {f["kind"] for f in figures} >= {"figure", "table"}
    assert all(f["has_image"] for f in figures)


@pytest.mark.usefixtures("figures_enabled")
def test_figure_image_is_served(client, figures_pdf):
    client.post(
        "/api/papers/upload?wait=true",
        files=[("files", (figures_pdf.name, figures_pdf.read_bytes(), "application/pdf"))],
    )
    figure = client.get("/api/figures").json()[0]

    image = client.get(f"/api/figures/{figure['id']}/image")
    assert image.status_code == 200
    assert image.content.startswith(b"\x89PNG")
    assert client.get("/api/figures/nope/image").status_code == 404


@pytest.mark.usefixtures("figures_enabled")
def test_deleting_a_paper_removes_its_figures(client, figures_pdf):
    paper_id = client.post(
        "/api/papers/upload?wait=true",
        files=[("files", (figures_pdf.name, figures_pdf.read_bytes(), "application/pdf"))],
    ).json()["results"][0]["paper_id"]

    assert client.get("/api/figures").json()
    client.delete(f"/api/papers/{paper_id}")
    assert client.get("/api/figures").json() == []


# --- vision analysis ---------------------------------------------------------


@pytest.mark.usefixtures("figures_enabled")
def test_cost_estimate_is_shown_before_spending_quota(client, figures_pdf):
    """On the free tier the number of requests is the whole decision."""
    client.post(
        "/api/papers/upload?wait=true",
        files=[("files", (figures_pdf.name, figures_pdf.read_bytes(), "application/pdf"))],
    )
    estimate = client.get("/api/figures/estimate").json()

    assert estimate["pending"] >= 1
    assert estimate["requests_required"] == estimate["pending"]
    assert estimate["capped_at"] > 0


@pytest.mark.usefixtures("figures_enabled")
def test_analysis_reads_the_chart(client, figures_pdf, fake_vision):
    client.post(
        "/api/papers/upload?wait=true",
        files=[("files", (figures_pdf.name, figures_pdf.read_bytes(), "application/pdf"))],
    )
    payload = client.post("/api/figures/analyse", json={}).json()

    assert payload["analysed"] >= 1
    assert payload["failed"] == 0
    assert fake_vision.calls, "the vision model was never called"
    assert fake_vision.calls[0]["bytes"] > 0, "no image was sent"

    analysed = next(f for f in payload["figures"] if f["status"] == "analysed")
    assert analysed["chart_type"] == "bar"
    assert analysed["axes"]["y_label"] == "Accuracy"
    assert analysed["findings"]
    assert analysed["takeaway"]
    assert analysed["entities"]["datasets"] == ["UA-Speech"]


@pytest.mark.usefixtures("figures_enabled")
def test_tables_read_as_markdown_cost_no_vision_call(client, figures_pdf, fake_vision):
    """A table we already parsed needs no picture of itself."""
    client.post(
        "/api/papers/upload?wait=true",
        files=[("files", (figures_pdf.name, figures_pdf.read_bytes(), "application/pdf"))],
    )
    figures = client.get("/api/figures").json()
    readable = [f for f in figures if f["table_markdown"]]

    for figure in readable:
        assert figure["status"] == "skipped"

    client.post("/api/figures/analyse", json={})
    assert len(fake_vision.calls) == len([f for f in figures if f["status"] == "pending"])


@pytest.mark.usefixtures("figures_enabled")
def test_analysis_failure_is_recorded_not_raised(client, figures_pdf):
    from app.services.llm import set_llm
    from tests.conftest import FakeVisionClient

    client.post(
        "/api/papers/upload?wait=true",
        files=[("files", (figures_pdf.name, figures_pdf.read_bytes(), "application/pdf"))],
    )
    set_llm(FakeVisionClient(fail=True))

    payload = client.post("/api/figures/analyse", json={}).json()

    assert payload["analysed"] == 0
    assert payload["failed"] >= 1
    assert payload["errors"]
    assert all(f["status"] == "failed" for f in payload["figures"])


@pytest.mark.usefixtures("figures_enabled")
def test_analysing_twice_does_not_repeat_the_cost(client, figures_pdf, fake_vision):
    client.post(
        "/api/papers/upload?wait=true",
        files=[("files", (figures_pdf.name, figures_pdf.read_bytes(), "application/pdf"))],
    )
    client.post("/api/figures/analyse", json={})
    first_round = len(fake_vision.calls)

    second = client.post("/api/figures/analyse", json={}).json()

    assert second["analysed"] == 0
    assert len(fake_vision.calls) == first_round, "already-read figures were analysed again"


@pytest.mark.usefixtures("figures_enabled")
def test_analysis_makes_figures_searchable(client, figures_pdf, fake_vision):
    """A chart's meaning is in its picture; indexing it makes it findable."""
    client.post(
        "/api/papers/upload?wait=true",
        files=[("files", (figures_pdf.name, figures_pdf.read_bytes(), "application/pdf"))],
    )
    client.post("/api/figures/analyse", json={})

    hits = client.post(
        "/api/papers/search",
        json={"query": "self-supervised features outperform hand-crafted baselines", "top_k": 5},
    ).json()["hits"]

    figure_hits = [hit for hit in hits if hit["kind"] == "figure"]
    assert figure_hits, "figures were not indexed for semantic search"
    assert figure_hits[0]["figure_id"]
    assert figure_hits[0]["label"]


@pytest.mark.usefixtures("figures_enabled")
def test_an_answer_can_cite_a_figure(client, figures_pdf, fake_vision):
    """A chart is evidence; an answer drawn from one must be able to point at it."""
    from app.agents.state import format_chunk_context
    from app.services.vector_store import RetrievedChunk

    client.post(
        "/api/papers/upload?wait=true",
        files=[("files", (figures_pdf.name, figures_pdf.read_bytes(), "application/pdf"))],
    )
    client.post("/api/figures/analyse", json={})

    # The prompt must mark a figure as such, so the model knows it is reading a
    # description of an image rather than the paper's prose.
    context = format_chunk_context(
        [
            RetrievedChunk(
                chunk_id="figure:abc",
                paper_id="p1",
                text="A bar chart of accuracy.",
                score=0.9,
                page_start=1,
                paper_title="A Paper",
                kind="figure",
                figure_id="abc",
                label="Figure 1",
            )
        ]
    )
    assert "[FIGURE]" in context
    assert "Figure 1" in context


@pytest.mark.usefixtures("figures_enabled")
def test_analysed_figures_become_graph_nodes(client, figures_pdf, fake_vision):
    client.post(
        "/api/papers/upload?wait=true",
        files=[("files", (figures_pdf.name, figures_pdf.read_bytes(), "application/pdf"))],
    )
    client.post("/api/figures/analyse", json={})
    client.post("/api/agents/graph/build", json={})

    graph = client.get("/api/graph").json()
    figure_nodes = [node for node in graph["nodes"] if node["type"] == "Figure"]

    assert figure_nodes, "analysed figures did not reach the knowledge graph"
    contains = [edge for edge in graph["edges"] if edge["type"] == "CONTAINS"]
    assert contains, "no paper -> figure edge"
