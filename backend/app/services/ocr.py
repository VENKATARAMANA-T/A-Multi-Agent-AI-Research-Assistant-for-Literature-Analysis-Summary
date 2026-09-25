"""Optical character recognition for scanned PDF pages.

A scanned paper has no text layer, so PyMuPDF returns nothing and the document
was previously rejected outright. This module recovers that text from a rendered
image of the page.

Three design decisions worth knowing:

*Per page, not per document.* Papers are frequently born-digital with a
photocopied appendix or a scanned figure page. Running OCR over a page whose
native text is already perfect only degrades it, so each page is scored
independently and only the ones that need it are recognised.

*Reading order is reconstructed, not assumed.* OCR returns boxes in detection
order, which on a two-column paper interleaves the columns into nonsense. Lines
are grouped into columns by their horizontal position, then read top-to-bottom
within each column.

*Engines are pluggable.* RapidOCR is the default because it installs from pip
with no system binary — Tesseract would need a separate installer on every
machine and inside the image. Gemini vision is available as a fallback, but it
costs API quota, so it is never chosen silently when RapidOCR is present.
"""

from __future__ import annotations

import hashlib
import io
import logging
import statistics
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


# --- results -----------------------------------------------------------------


@dataclass
class OCRLine:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float

    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class OCRPageResult:
    text: str
    engine: str
    lines: list[OCRLine] = field(default_factory=list)
    confidence: float = 0.0
    columns: int = 1
    cached: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class OCRUnavailable(RuntimeError):
    """No usable OCR engine is installed or configured."""


# --- reading order -----------------------------------------------------------

# A gutter narrower than this fraction of the page is not a column break.
COLUMN_GAP_RATIO = 0.06
# Each column must hold at least this share of the lines, or the "split" is just
# one short line sitting off to the side.
MIN_COLUMN_SHARE = 0.2
# Lines whose vertical centres are within this fraction of page height belong to
# the same visual row.
ROW_TOLERANCE_RATIO = 0.012


def detect_columns(lines: list[OCRLine], page_width: float) -> list[list[OCRLine]]:
    """Group lines into reading columns by finding the widest vertical gutter.

    Returns one list per column, left to right. A single-column page returns one
    group, which is the common case and costs nothing.
    """
    if len(lines) < 6 or page_width <= 0:
        return [lines]

    centers = sorted(line.x_center for line in lines)
    # The largest gap between consecutive x-centres is the candidate gutter.
    best_gap = 0.0
    split_at = 0.0
    for left, right in zip(centers, centers[1:]):
        gap = right - left
        if gap > best_gap:
            best_gap = gap
            split_at = (left + right) / 2

    if best_gap < page_width * COLUMN_GAP_RATIO:
        return [lines]

    left_column = [line for line in lines if line.x_center <= split_at]
    right_column = [line for line in lines if line.x_center > split_at]

    share = min(len(left_column), len(right_column)) / len(lines)
    if share < MIN_COLUMN_SHARE:
        return [lines]

    return [left_column, right_column]


def order_lines(lines: list[OCRLine], page_width: float, page_height: float) -> tuple[str, int]:
    """Sort recognised lines into human reading order and join them.

    Returns (text, column_count).
    """
    if not lines:
        return "", 1

    columns = detect_columns(lines, page_width)
    tolerance = max(4.0, page_height * ROW_TOLERANCE_RATIO)

    blocks: list[str] = []
    for column in columns:
        # Top to bottom, then left to right for words sharing a row.
        ordered = sorted(column, key=lambda line: (round(line.y0 / tolerance), line.x0))

        rows: list[list[OCRLine]] = []
        for line in ordered:
            if rows and abs(line.y0 - rows[-1][-1].y0) <= tolerance:
                rows[-1].append(line)
            else:
                rows.append([line])

        blocks.append(
            "\n".join(" ".join(item.text for item in sorted(row, key=lambda l: l.x0)) for row in rows)
        )

    return "\n".join(block for block in blocks if block.strip()), len(columns)


# --- engines -----------------------------------------------------------------


class OCREngine(ABC):
    name: str

    @abstractmethod
    def recognise(self, image_png: bytes, page_width: float, page_height: float) -> OCRPageResult: ...


class RapidOCREngine(OCREngine):
    """ONNX-based OCR. Ships its own models, needs no system binary."""

    name = "rapidocr"

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR  # imported lazily

        self._engine = RapidOCR()
        # RapidOCR is not documented as thread-safe and ingestion runs on a
        # worker pool, so calls are serialised.
        self._lock = threading.Lock()

    def recognise(self, image_png: bytes, page_width: float, page_height: float) -> OCRPageResult:
        import numpy as np
        from PIL import Image

        image = np.array(Image.open(io.BytesIO(image_png)).convert("RGB"))
        with self._lock:
            raw, _ = self._engine(image, unclip_ratio=settings.ocr_unclip_ratio)

        # A blank page yields None rather than an empty list.
        if not raw:
            return OCRPageResult(text="", engine=self.name, confidence=0.0)

        scale_x = page_width / image.shape[1] if image.shape[1] else 1.0
        scale_y = page_height / image.shape[0] if image.shape[0] else 1.0

        lines: list[OCRLine] = []
        for entry in raw:
            try:
                box, text, score = entry
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
            text = str(text).strip()
            if not text:
                continue
            # Scores come back as strings from this engine.
            try:
                confidence = float(score)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < settings.ocr_min_confidence:
                continue

            xs = [float(point[0]) * scale_x for point in box]
            ys = [float(point[1]) * scale_y for point in box]
            lines.append(
                OCRLine(
                    text=text,
                    x0=min(xs),
                    y0=min(ys),
                    x1=max(xs),
                    y1=max(ys),
                    confidence=confidence,
                )
            )

        if not lines:
            return OCRPageResult(text="", engine=self.name, confidence=0.0)

        text, columns = order_lines(lines, page_width, page_height)
        return OCRPageResult(
            text=text,
            engine=self.name,
            lines=lines,
            confidence=round(statistics.fmean(line.confidence for line in lines), 4),
            columns=columns,
        )


class GeminiVisionOCREngine(OCREngine):
    """Fallback that reads the page image with the LLM.

    Accurate on hard scans and handwriting, but costs one API request per page,
    so it is only used when explicitly selected or when no local engine exists.
    """

    name = "gemini-vision"

    PROMPT = (
        "Transcribe all text from this scanned page of an academic paper.\n"
        "Rules:\n"
        "- Output only the transcribed text, with no commentary.\n"
        "- Preserve the reading order. If the page has two columns, read the left "
        "column fully before the right one.\n"
        "- Keep paragraph breaks. Do not translate, summarise, or correct the text.\n"
        "- For a table, output each row on its own line with cells separated by ' | '.\n"
        "- If the page contains no readable text, output nothing."
    )

    def recognise(self, image_png: bytes, page_width: float, page_height: float) -> OCRPageResult:
        from app.services.llm import get_llm

        text = get_llm().generate_from_image(
            prompt=self.PROMPT,
            image_bytes=image_png,
            mime_type="image/png",
            temperature=0.0,
        )
        cleaned = (text or "").strip()
        return OCRPageResult(
            text=cleaned,
            engine=self.name,
            # The model reports no per-line score; treat a non-empty answer as
            # usable rather than inventing a number.
            confidence=1.0 if cleaned else 0.0,
        )


# --- engine selection --------------------------------------------------------

_engine: OCREngine | None = None
_engine_error: str | None = None
_lock = threading.Lock()


def get_engine() -> OCREngine:
    """Return the configured engine, constructing it once."""
    global _engine, _engine_error

    if _engine is not None:
        return _engine
    if _engine_error is not None:
        raise OCRUnavailable(_engine_error)

    with _lock:
        if _engine is not None:
            return _engine

        preference = (settings.ocr_engine or "auto").lower()
        if preference == "none" or not settings.ocr_enabled:
            _engine_error = "OCR is disabled (OCR_ENABLED=false or OCR_ENGINE=none)."
            raise OCRUnavailable(_engine_error)

        attempts: list[str] = []
        if preference in ("auto", "rapidocr"):
            try:
                _engine = RapidOCREngine()
                logger.info("OCR engine: rapidocr")
                return _engine
            except ImportError:
                attempts.append("rapidocr is not installed (pip install rapidocr-onnxruntime)")
            except Exception as exc:  # pragma: no cover - host dependent
                attempts.append(f"rapidocr failed to start: {exc}")

        if preference in ("auto", "gemini"):
            if settings.google_api_key:
                _engine = GeminiVisionOCREngine()
                logger.info("OCR engine: gemini-vision (costs one API request per page)")
                return _engine
            attempts.append("gemini vision needs GOOGLE_API_KEY")

        _engine_error = "No OCR engine available: " + "; ".join(attempts)
        raise OCRUnavailable(_engine_error)


def reset_engine() -> None:
    """Test hook."""
    global _engine, _engine_error
    with _lock:
        _engine = None
        _engine_error = None


def engine_status() -> dict[str, Any]:
    """Describe OCR availability for /api/health, without raising."""
    if not settings.ocr_enabled:
        return {"enabled": False, "engine": None, "reason": "OCR_ENABLED=false"}
    try:
        return {
            "enabled": True,
            "engine": get_engine().name,
            "dpi": settings.ocr_dpi,
            "max_pages": settings.ocr_max_pages_per_document,
        }
    except OCRUnavailable as exc:
        return {"enabled": True, "engine": None, "reason": str(exc)}


# --- caching -----------------------------------------------------------------


def cache_key(doc_id: str, page_number: int, engine: str) -> str:
    """Key on the document's content hash, so a cache hit skips even rendering."""
    payload = f"{doc_id}:{page_number}:{settings.ocr_dpi}:{engine}:{settings.ocr_min_confidence}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def recognise_page(
    image_png: bytes,
    page_width: float,
    page_height: float,
    doc_id: str | None = None,
    page_number: int = 0,
) -> OCRPageResult:
    """Recognise one page, consulting the cache when a document id is given."""
    engine = get_engine()

    key = None
    if doc_id and settings.ocr_cache_enabled:
        from app.services import ocr_cache

        key = cache_key(doc_id, page_number, engine.name)
        hit = ocr_cache.lookup(key)
        if hit is not None:
            return OCRPageResult(
                text=hit.text,
                engine=hit.engine,
                confidence=hit.confidence,
                columns=hit.columns,
                cached=True,
            )

    result = engine.recognise(image_png, page_width, page_height)

    if key:
        from app.services import ocr_cache

        ocr_cache.store(
            key,
            text=result.text,
            engine=result.engine,
            confidence=result.confidence,
            columns=result.columns,
        )
    return result
