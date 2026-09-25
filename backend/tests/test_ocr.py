"""OCR for scanned PDFs.

The scanned fixtures are real image-only PDFs: a page is composed, rasterised,
and the image placed on a blank page. PyMuPDF finds no text in them at all,
which is exactly the case that used to be rejected outright.
"""

from __future__ import annotations

import fitz
import pytest

from app.services import ocr, ocr_cache
from app.services.chunking import chunk_document
from app.services.ocr import OCRLine, detect_columns, order_lines
from app.services.pdf_extract import extract_pdf, image_coverage, page_needs_ocr


def line(text: str, x0: float, y0: float, width: float = 80, height: float = 14) -> OCRLine:
    return OCRLine(text=text, x0=x0, y0=y0, x1=x0 + width, y1=y0 + height, confidence=0.9)


# --- the fixtures really are text-free ---------------------------------------


def test_scanned_fixture_has_no_text_layer(scanned_pdf):
    """Guard the guard: if this ever gains a text layer the OCR tests are hollow."""
    with fitz.open(scanned_pdf) as document:
        native = "".join(page.get_text("text") for page in document)
    assert native.strip() == ""


def test_scanned_fixture_is_mostly_image(scanned_pdf):
    with fitz.open(scanned_pdf) as document:
        assert image_coverage(document[0]) > 0.9


def test_blank_fixture_has_neither_text_nor_image(blank_pdf):
    with fitz.open(blank_pdf) as document:
        assert document[0].get_text("text").strip() == ""
        assert image_coverage(document[0]) == 0.0


# --- the OCR decision --------------------------------------------------------


def test_scanned_page_is_a_candidate(scanned_pdf):
    with fitz.open(scanned_pdf) as document:
        page = document[0]
        assert page_needs_ocr(page, page.get_text("text")) is True


def test_blank_page_is_not_a_candidate(blank_pdf):
    """A blank page has no text either — but OCRing it is pure waste."""
    with fitz.open(blank_pdf) as document:
        page = document[0]
        assert page_needs_ocr(page, page.get_text("text")) is False


def test_page_with_a_text_layer_is_not_a_candidate(sample_pdf):
    with fitz.open(sample_pdf) as document:
        page = document[0]
        assert page_needs_ocr(page, page.get_text("text")) is False


# --- reading order -----------------------------------------------------------


def test_single_column_stays_one_group():
    lines = [line(f"row {i}", 60, 100 + i * 20) for i in range(8)]
    assert len(detect_columns(lines, page_width=600)) == 1


def test_two_columns_are_detected():
    left = [line(f"L{i}", 60, 100 + i * 20) for i in range(5)]
    right = [line(f"R{i}", 340, 100 + i * 20) for i in range(5)]
    columns = detect_columns(left + right, page_width=600)

    assert len(columns) == 2
    assert {item.text for item in columns[0]} == {f"L{i}" for i in range(5)}


def test_two_column_text_is_read_column_by_column():
    """Detection order interleaves the columns; reading order must not."""
    interleaved = []
    for i in range(4):
        interleaved.append(line(f"left{i}", 60, 100 + i * 20))
        interleaved.append(line(f"right{i}", 340, 100 + i * 20))

    text, columns = order_lines(interleaved, page_width=600, page_height=800)

    assert columns == 2
    assert text.index("left3") < text.index("right0"), "columns were interleaved"


def test_one_stray_line_does_not_create_a_column():
    """A single caption off to the side is not a column break."""
    lines = [line(f"row {i}", 60, 100 + i * 20) for i in range(9)]
    lines.append(line("stray", 420, 400))
    assert len(detect_columns(lines, page_width=600)) == 1


def test_narrow_gap_is_not_a_gutter():
    left = [line(f"L{i}", 60, 100 + i * 20) for i in range(5)]
    right = [line(f"R{i}", 75, 100 + i * 20) for i in range(5)]
    assert len(detect_columns(left + right, page_width=600)) == 1


def test_no_lines_yields_empty_text():
    assert order_lines([], 600, 800) == ("", 1)


# --- disabled / unavailable --------------------------------------------------


def test_scanned_pdf_yields_nothing_when_ocr_is_off(scanned_pdf):
    document = extract_pdf(scanned_pdf)

    assert document.text.strip() == ""
    assert document.ocr_pages == []
    assert document.text_source == "native"


def test_upload_explains_that_ocr_is_disabled(client, scanned_pdf):
    response = client.post(
        "/api/papers/upload?wait=true",
        files=[("files", (scanned_pdf.name, scanned_pdf.read_bytes(), "application/pdf"))],
    )
    payload = response.json()

    assert payload["failed"] == 1
    detail = payload["results"][0]["detail"]
    assert "scanned" in detail.lower()
    assert "OCR_ENABLED" in detail


def test_engine_status_reports_disabled():
    status = ocr.engine_status()
    assert status["enabled"] is False


def test_unavailable_engine_is_reported_not_raised(monkeypatch, scanned_pdf):
    """A scanned page with no engine must degrade, not crash ingestion."""
    from app.config import settings

    monkeypatch.setattr(settings, "ocr_enabled", True)
    monkeypatch.setattr(settings, "ocr_engine", "gemini")  # needs a key, none set
    ocr.reset_engine()

    document = extract_pdf(scanned_pdf)

    assert document.text.strip() == ""
    assert document.ocr_skipped_pages == [1]
    ocr.reset_engine()


# --- real recognition --------------------------------------------------------


@pytest.mark.usefixtures("ocr_enabled")
class TestRealOCR:
    """These run the actual OCR engine, so they are slower than the rest."""

    def test_text_is_recovered_from_a_scan(self, scanned_pdf):
        document = extract_pdf(scanned_pdf)

        assert document.text.strip(), "OCR recovered nothing"
        assert "Sparse Attention" in document.text
        assert document.ocr_pages == [1]
        assert document.text_source == "ocr"
        assert document.ocr_engine == "rapidocr"
        assert document.ocr_confidence and document.ocr_confidence > 0.5

    def test_title_is_recovered_without_font_metadata(self, scanned_pdf):
        """The layout heuristic needs font sizes; a scan has none."""
        document = extract_pdf(scanned_pdf)
        assert document.title and "Sparse Attention" in document.title

    def test_pages_are_marked_with_their_source(self, scanned_pdf):
        document = extract_pdf(scanned_pdf)

        assert document.pages[0].source == "ocr"
        assert document.pages[0].ocr_confidence is not None

    def test_chunks_inherit_the_ocr_source(self, scanned_pdf):
        document = extract_pdf(scanned_pdf)
        chunks = chunk_document(document, chunk_size=400, chunk_overlap=40)

        assert chunks
        assert all(chunk.source == "ocr" for chunk in chunks)

    def test_native_pdf_is_untouched_by_ocr(self, sample_pdf):
        """OCR must not run over pages whose text layer is already good."""
        document = extract_pdf(sample_pdf)

        assert document.ocr_pages == []
        assert document.text_source == "native"
        assert all(page.source == "native" for page in document.pages)

    def test_blank_page_is_skipped(self, blank_pdf):
        document = extract_pdf(blank_pdf)
        assert document.ocr_pages == []

    def test_two_column_scan_reads_column_by_column(self, scanned_two_column_pdf):
        """Detection order interleaves the columns; reading order must not."""
        document = extract_pdf(scanned_two_column_pdf)
        text = document.text

        # The last line of the left column must precede the first of the right.
        last_left = text.index("On the arXiv and PubMed")
        first_right = text.index("reaches 44.1 ROUGE-L")
        assert last_left < first_right, f"columns were interleaved:\n{text}"

    def test_results_are_cached_by_content_hash(self, scanned_pdf):
        first = extract_pdf(scanned_pdf, doc_id="doc-abc")
        assert ocr_cache.summary()["entries"] == 1

        second = extract_pdf(scanned_pdf, doc_id="doc-abc")
        assert second.text == first.text
        assert ocr_cache.summary()["pages_saved"] >= 1

    def test_cache_is_not_shared_across_documents(self, scanned_pdf):
        extract_pdf(scanned_pdf, doc_id="doc-one")
        extract_pdf(scanned_pdf, doc_id="doc-two")
        assert ocr_cache.summary()["entries"] == 2

    def test_page_limit_is_respected(self, monkeypatch, scanned_pdf):
        from app.config import settings

        monkeypatch.setattr(settings, "ocr_max_pages_per_document", 0)
        document = extract_pdf(scanned_pdf)

        assert document.ocr_pages == []
        assert document.ocr_skipped_pages == [1]

    def test_no_line_is_silently_dropped(self, scanned_dense_pdf):
        """Regression: the detector's default polygon expansion lost a line.

        With `unclip_ratio` at the engine default of 1.6, one line in four of
        tightly-spaced 9pt body text was never detected at all — the page was
        rendered correctly and the ink was there, but the line vanished. Losing
        a quarter of a paper's sentences without any error is the worst kind of
        failure, so this pins the recovery.
        """
        from tests.conftest import DENSE_LINES

        document = extract_pdf(scanned_dense_pdf)
        text = document.text

        missing = [
            row for row in DENSE_LINES
            # Compare on a distinctive fragment: recognition may differ in
            # punctuation, but a whole line must not disappear.
            if row.split(",")[0][:28] not in text
        ]
        assert not missing, f"OCR dropped {len(missing)} line(s):\n{missing}\ngot:\n{text}"

    def test_numbers_survive_recognition(self, scanned_dense_pdf):
        """Metrics are the point of a results section; digits must come through."""
        document = extract_pdf(scanned_dense_pdf)

        for value in ("87.4", "44.1", "2.3"):
            assert value in document.text, f"lost the figure {value}"

    def test_engine_status_names_the_engine(self):
        status = ocr.engine_status()
        assert status["enabled"] is True
        assert status["engine"] == "rapidocr"
        assert status["dpi"] > 0


# --- end to end through the API ---------------------------------------------


@pytest.mark.usefixtures("ocr_enabled")
def test_scanned_pdf_indexes_end_to_end(client, scanned_pdf):
    response = client.post(
        "/api/papers/upload?wait=true",
        files=[("files", (scanned_pdf.name, scanned_pdf.read_bytes(), "application/pdf"))],
    )
    payload = response.json()

    assert payload["indexed"] == 1, payload["results"][0].get("detail")
    paper = payload["results"][0]["paper"]

    assert paper["status"] == "indexed"
    assert paper["chunk_count"] > 0
    assert paper["text_source"] == "ocr"
    assert paper["ocr_pages"] == [1]
    assert paper["ocr_engine"] == "rapidocr"
    assert paper["ocr_confidence"] > 0.5


@pytest.mark.usefixtures("ocr_enabled")
def test_search_flags_ocr_derived_text(client, scanned_pdf):
    client.post(
        "/api/papers/upload?wait=true",
        files=[("files", (scanned_pdf.name, scanned_pdf.read_bytes(), "application/pdf"))],
    )
    hits = client.post("/api/papers/search", json={"query": "sparse attention", "top_k": 3}).json()

    assert hits["hits"]
    assert all(hit["source"] == "ocr" for hit in hits["hits"])


@pytest.mark.usefixtures("ocr_enabled")
def test_health_reports_the_ocr_engine(client):
    payload = client.get("/api/health").json()

    assert payload["ocr"]["enabled"] is True
    assert payload["ocr"]["engine"] == "rapidocr"
    assert "cache" in payload["ocr"]
