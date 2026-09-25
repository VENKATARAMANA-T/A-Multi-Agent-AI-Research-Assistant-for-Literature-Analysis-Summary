"""Figure, chart and table extraction from PDF pages.

Academic charts are usually *vector* drawings rather than embedded rasters, so
`page.get_images()` alone finds almost nothing. What is reliably present is the
caption — "Fig. 3 | ROC curves for…" — which both names the figure and marks
where it sits. Regions are therefore anchored on captions:

  * a **figure** caption sits *below* its artwork, so the region is the band
    above it, narrowed to the graphics actually drawn there;
  * a **table** caption sits *above* its content, and a table is text rather
    than artwork — so tables are matched against PyMuPDF's own table finder and
    never against graphics. Unioning graphics below a table caption grabs the
    next figure on the page, which is exactly the bug this avoids.

Every region is bounded by its neighbouring captions, so one figure's box can
never swallow the one after it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import fitz

logger = logging.getLogger(__name__)

CAPTION_RE = re.compile(
    r"^\s*(fig(?:ure)?\.?|table|chart|algorithm|scheme)\s*"
    r"(\d{1,3}|[ivxlc]{1,6})\s*"
    r"([.:)|\-–,]?)\s*(.{0,400})",
    re.IGNORECASE | re.DOTALL,
)

# "Fig. 4 depicts the correlation matrices…" is a sentence in the body text that
# happens to start a block, not a caption. A caption names its subject; a
# cross-reference talks about it, and the give-away is the verb straight after
# the number.
REFERENCE_WORDS = frozenset(
    {
        "depicts", "shows", "contains", "illustrates", "presents", "displays",
        "lists", "summarizes", "summarises", "reports", "gives", "provides",
        "compares", "indicates", "describes", "demonstrates", "highlights",
        "confirms", "reveals", "shown", "seen", "above", "below", "also",
        "and", "is", "are", "was", "were", "we", "in", "of", "for", "to",
        "that", "which", "this", "these", "as", "it", "they",
    }
)

MAX_REGION_RATIO = 0.62      # a region may not exceed this share of page height
MIN_REGION_WIDTH = 60
MIN_REGION_HEIGHT = 45
MIN_GRAPHIC_AREA = 120
HORIZONTAL_OVERLAP = 0.25    # graphic must share this much width with the caption
TABLE_SEARCH_DEPTH = 160     # how far below a caption to look for its table
PAD = 4


@dataclass
class ExtractedFigure:
    kind: str                # figure | table | algorithm | chart
    label: str               # "Figure 3"
    caption: str
    page: int                # 1-based
    bbox: tuple[float, float, float, float]
    image_png: bytes | None = None
    table_markdown: str | None = None
    table_rows: int = 0
    table_cols: int = 0
    detector: str = "caption"

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "caption": self.caption,
            "page": self.page,
            "bbox": list(self.bbox),
            "table_markdown": self.table_markdown,
            "table_rows": self.table_rows,
            "table_cols": self.table_cols,
            "detector": self.detector,
            "has_image": self.image_png is not None,
        }


@dataclass
class _Caption:
    rect: fitz.Rect
    kind: str
    label: str
    text: str


# --- caption detection -------------------------------------------------------


def classify_caption(text: str) -> tuple[str, str, str] | None:
    """Return (kind, label, cleaned caption) if `text` starts a real caption."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) < 8:
        return None

    match = CAPTION_RE.match(cleaned)
    if not match:
        return None

    word, number, separator, rest = match.group(1), match.group(2), match.group(3), match.group(4) or ""

    # A reference inside a sentence, e.g. "Fig. 20)." or "Table III contains…".
    if separator in (")", ","):
        return None
    first_word = (rest.strip().split() or [""])[0].lower().strip(".,:;)")
    if first_word in REFERENCE_WORDS:
        return None

    lowered = word.lower()
    if lowered.startswith("table"):
        kind, display = "table", "Table"
    elif lowered.startswith("algorithm"):
        kind, display = "algorithm", "Algorithm"
    elif lowered.startswith("scheme"):
        kind, display = "figure", "Scheme"
    elif lowered.startswith("chart"):
        kind, display = "chart", "Chart"
    else:
        kind, display = "figure", "Figure"

    label = f"{display} {number.upper() if number.isalpha() else number}"
    return kind, label, trim_caption(f"{label}. {rest}".strip())


CAPTION_MAX_CHARS = 400


def trim_caption(text: str) -> str:
    """Cut a caption back to its own text.

    Some PDFs put the caption and the paragraph that follows it in a single
    text block, so the "caption" arrives 800 characters long with half a
    literature review attached. That inflates the vision prompt and pollutes
    the embedded search text, so it is cut at the last sentence boundary within
    a plausible caption length.
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= CAPTION_MAX_CHARS:
        return cleaned

    window = cleaned[:CAPTION_MAX_CHARS]
    # Prefer a sentence end; fall back to a word boundary.
    cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
    if cut < CAPTION_MAX_CHARS // 3:
        cut = window.rfind(" ")
    if cut <= 0:
        cut = CAPTION_MAX_CHARS
    return cleaned[: cut + 1].strip() + " […]"


def find_captions(page: fitz.Page) -> list[_Caption]:
    captions: list[_Caption] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        text = " ".join(
            span.get("text", "")
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        )
        info = classify_caption(text)
        if info is None:
            continue
        kind, label, cleaned = info
        captions.append(_Caption(rect=fitz.Rect(block["bbox"]), kind=kind, label=label, text=cleaned))
    return sorted(captions, key=lambda c: (c.rect.y0, c.rect.x0))


# --- page geometry -----------------------------------------------------------


def graphic_rects(page: fitz.Page) -> list[fitz.Rect]:
    """Rectangles of every raster image and vector drawing on the page."""
    rects: list[fitz.Rect] = []

    for image in page.get_images(full=True):
        try:
            rects.extend(page.get_image_rects(image[0]))
        except Exception:  # pragma: no cover - malformed xref
            continue

    try:
        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            if rect is not None and rect.get_area() >= MIN_GRAPHIC_AREA:
                rects.append(fitz.Rect(rect))
    except Exception:  # pragma: no cover
        pass

    return rects


MAX_TABLE_COLUMNS = 16
MAX_TABLE_AREA_RATIO = 0.75


def detected_tables(
    page: fitz.Page, captions: list[_Caption] | None = None
) -> list[tuple[fitz.Rect, Any]]:
    """PyMuPDF's own table finder, filtered to plausible tables.

    The finder sometimes returns one page-wide box that merges a chart, the
    body text and two separate tables into a single "25x37 table". Such a
    region is worse than no match at all, so implausible results are dropped:
    a genuine table does not have 37 columns, and does not contain another
    element's caption inside its own bounds.
    """
    try:
        found = page.find_tables()
    except Exception:  # pragma: no cover - depends on page content
        return []

    page_area = page.rect.get_area() or 1.0
    caption_rects = [c.rect for c in (captions or [])]
    results: list[tuple[fitz.Rect, Any]] = []

    for table in getattr(found, "tables", []):
        if table.row_count < 2 or table.col_count < 2:
            continue  # a 1xN "table" is a header or a stray rule
        if table.col_count > MAX_TABLE_COLUMNS:
            continue
        rect = fitz.Rect(table.bbox)
        if rect.get_area() / page_area > MAX_TABLE_AREA_RATIO:
            continue
        # A box swallowing two or more captions spans several page elements.
        enclosed = sum(1 for caption in caption_rects if rect.contains(caption))
        if enclosed > 1:
            continue
        results.append((rect, table))

    return results


def _horizontal_overlap(a: fitz.Rect, b: fitz.Rect) -> float:
    span = min(a.x1, b.x1) - max(a.x0, b.x0)
    width = min(a.width, b.width)
    return span / width if width > 0 else 0.0


def _bound_above(caption: fitz.Rect, others: Iterable[fitz.Rect], page_top: float) -> float:
    """Highest y a region may reach: just under the nearest element above it."""
    limit = page_top
    for other in others:
        if other.y1 <= caption.y0 and _horizontal_overlap(caption, other) > HORIZONTAL_OVERLAP:
            limit = max(limit, other.y1)
    return limit


def _bound_below(caption: fitz.Rect, others: Iterable[fitz.Rect], page_bottom: float) -> float:
    limit = page_bottom
    for other in others:
        if other.y0 >= caption.y1 and _horizontal_overlap(caption, other) > HORIZONTAL_OVERLAP:
            limit = min(limit, other.y0)
    return limit


# --- region construction -----------------------------------------------------


def _figure_region(
    page: fitz.Page,
    caption: _Caption,
    graphics: list[fitz.Rect],
    neighbours: list[fitz.Rect],
) -> fitz.Rect | None:
    """The artwork above a figure caption."""
    page_rect = page.rect
    ceiling = max(
        page_rect.y0,
        caption.rect.y0 - page_rect.height * MAX_REGION_RATIO,
        _bound_above(caption.rect, neighbours, page_rect.y0),
    )
    band = fitz.Rect(caption.rect.x0 - 14, ceiling, caption.rect.x1 + 14, caption.rect.y0 - 1)
    if band.height < MIN_REGION_HEIGHT:
        return None

    # NB: `&` returns a new rectangle. `Rect.intersect()` mutates in place, so
    # using it here silently shrank the band on every iteration and starved the
    # later graphics of any area to match against.
    inside = [
        rect
        for rect in graphics
        if (band & rect).get_area() > MIN_GRAPHIC_AREA
        and _horizontal_overlap(caption.rect, rect) > HORIZONTAL_OVERLAP
    ]

    if inside:
        union = fitz.Rect(inside[0])
        for rect in inside[1:]:
            union |= rect
        region = union & band
    else:
        # No vector or raster content found: fall back to the whole band, which
        # still captures artwork drawn in ways we did not enumerate.
        region = fitz.Rect(band)

    region = fitz.Rect(region.x0 - PAD, region.y0 - PAD, region.x1 + PAD, region.y1 + PAD)
    return region & page_rect


def _table_region(
    page: fitz.Page,
    caption: _Caption,
    tables: list[tuple[fitz.Rect, Any]],
    neighbours: list[fitz.Rect],
    claimed: set[int] | None = None,
) -> tuple[fitz.Rect | None, Any]:
    """The tabular content belonging to a table caption.

    Matched against the table finder rather than against graphics: a table is
    text, and taking "everything below the caption" would swallow the next
    figure on the page.
    """
    page_rect = page.rect

    # A caption normally sits directly above its table, but some journals place
    # it below, so look both ways — nearest wins.
    best: tuple[float, fitz.Rect, Any, int] | None = None
    for index, (rect, table) in enumerate(tables):
        # One detected table belongs to one caption; without this, two captions
        # on the same page both claim it and produce identical regions.
        if claimed is not None and index in claimed:
            continue
        if _horizontal_overlap(caption.rect, rect) < HORIZONTAL_OVERLAP:
            continue
        if rect.y0 >= caption.rect.y1:
            distance = rect.y0 - caption.rect.y1
        elif rect.y1 <= caption.rect.y0:
            distance = caption.rect.y0 - rect.y1
        else:
            distance = 0.0
        if distance <= TABLE_SEARCH_DEPTH and (best is None or distance < best[0]):
            best = (distance, rect, table, index)

    if best is not None:
        _, rect, table, index = best
        if claimed is not None:
            claimed.add(index)
        padded = fitz.Rect(rect.x0 - PAD, rect.y0 - PAD, rect.x1 + PAD, rect.y1 + PAD)
        return padded & page_rect, table

    # No structured table found. Follow the text blocks below the caption
    # instead of assuming a band: a caption like "TABLE II" is only 64pt wide,
    # so a caption-width band produces a tall sliver that misses the table
    # entirely. The blocks themselves reveal the true extent.
    floor = min(
        page_rect.y1,
        caption.rect.y1 + page_rect.height * MAX_REGION_RATIO,
        _bound_below(caption.rect, neighbours, page_rect.y1),
    )
    region = _content_below(page, caption.rect, floor)
    if region is None or region.height < MIN_REGION_HEIGHT:
        return None, None
    return region & page_rect, None


def _content_below(page: fitz.Page, caption: fitz.Rect, floor: float) -> fitz.Rect | None:
    """The table's content, whether it is drawn as text or as graphics.

    Many journals typeset tables as vector rules with no extractable text
    blocks at all, so a text-only walk finds nothing. When the space directly
    under the caption holds graphics but no text, that space *is* the table.
    """
    in_column_text = _in_column_text_blocks(page, caption, floor)
    first_text_y = in_column_text[0][0].y0 if in_column_text else floor

    # Is there a graphic-only gap between the caption and the next text?
    gap = fitz.Rect(caption.x0 - 14, caption.y1 + 1, caption.x1 + 14, min(first_text_y, floor))
    if gap.height >= MIN_REGION_HEIGHT:
        graphics = [
            rect
            for rect in graphic_rects(page)
            if (gap & rect).get_area() > MIN_GRAPHIC_AREA
            and _horizontal_overlap(caption, rect) > HORIZONTAL_OVERLAP
        ]
        if graphics:
            union = fitz.Rect(graphics[0])
            for rect in graphics[1:]:
                union |= rect
            union = union & gap
            if union.height >= MIN_REGION_HEIGHT:
                return fitz.Rect(union.x0 - PAD, union.y0 - PAD, union.x1 + PAD, union.y1 + PAD)

    return _accumulate_blocks_below(in_column_text, caption)


def _in_column_text_blocks(
    page: fitz.Page, caption: fitz.Rect, floor: float
) -> list[tuple[fitz.Rect, str]]:
    """Text blocks under the caption that share its column, top to bottom."""
    blocks: list[tuple[fitz.Rect, str]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        rect = fitz.Rect(block["bbox"])
        if rect.y0 < caption.y1 or rect.y0 >= floor:
            continue
        # Overlap is measured against the narrower box, so a 64pt-wide
        # "TABLE II" caption still matches the 452pt table beneath it, while a
        # block in the facing column does not.
        if _horizontal_overlap(caption, rect) < 0.5:
            continue
        text = " ".join(
            span.get("text", "")
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        )
        blocks.append((rect, text))
    blocks.sort(key=lambda pair: pair[0].y0)
    return blocks


PROSE_MIN_WORDS = 24
PROSE_MIN_SENTENCE_LENGTH = 11
TABLE_DIGIT_RATIO = 0.25


def looks_like_prose(text: str) -> bool:
    """Distinguish a paragraph of body text from the contents of a table.

    Without this the table region keeps growing past the table and swallows the
    rest of the page — observed capturing two columns of body text and the next
    figure along with the table it was meant to crop.

    The digit test matters as much as the sentence test. A table extracted as
    one flat block reads "Feature ACC SE SP F1 Spectrogram 92.72 0.92 0.94 …":
    long, and with no sentence breaks at all, because "92.72" has no space
    after its full stop. Measuring sentence length alone therefore classified
    the table itself as prose and dropped it. Number density separates them.
    """
    words = text.split()
    if len(words) < PROSE_MIN_WORDS:
        return False

    numeric = sum(1 for word in words if any(char.isdigit() for char in word))
    if numeric / len(words) > TABLE_DIGIT_RATIO:
        return False  # number-dense: tabular data, not prose

    sentences = [part for part in re.split(r"[.!?]\s+", text) if part.strip()]
    return len(words) / max(1, len(sentences)) >= PROSE_MIN_SENTENCE_LENGTH


def _accumulate_blocks_below(
    blocks: list[tuple[fitz.Rect, str]],
    caption: fitz.Rect,
    max_gap: float = 26.0,
) -> fitz.Rect | None:
    """Union the run of text blocks forming a table.

    Stops at the first wide vertical gap or the first paragraph of prose —
    without which the region keeps growing and swallows the rest of the page.
    """
    union: fitz.Rect | None = None
    previous_bottom = caption.y1

    for rect, text in blocks:
        if union is not None and rect.y0 - previous_bottom > max_gap:
            break
        if looks_like_prose(text):
            break
        union = fitz.Rect(rect) if union is None else (union | rect)
        previous_bottom = max(previous_bottom, rect.y1)

    if union is None:
        return None
    return fitz.Rect(union.x0 - PAD, union.y0 - PAD, union.x1 + PAD, union.y1 + PAD)


def table_to_markdown(table: Any) -> tuple[str, int, int]:
    """Render a detected table as Markdown so it can be read and embedded."""
    try:
        rows = table.extract()
    except Exception:  # pragma: no cover
        return "", 0, 0
    if not rows:
        return "", 0, 0

    def clean(cell: Any) -> str:
        return re.sub(r"\s+", " ", str(cell if cell is not None else "")).strip().replace("|", "\\|")

    header = [clean(cell) or f"col{i + 1}" for i, cell in enumerate(rows[0])]
    width = len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows[1:]:
        cells = [clean(cell) for cell in row][:width]
        cells += [""] * (width - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines), len(rows), width


# --- public API --------------------------------------------------------------


def extract_figures(
    path: str,
    dpi: int = 150,
    max_figures: int = 60,
    render_images: bool = True,
) -> list[ExtractedFigure]:
    """Find every figure, chart and table in a PDF, with its caption and image."""
    results: list[ExtractedFigure] = []

    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            captions = find_captions(page)
            if not captions:
                continue

            graphics = graphic_rects(page)
            tables = detected_tables(page, captions)
            caption_rects = [c.rect for c in captions]
            claimed: set[int] = set()

            for caption in captions:
                neighbours = [r for r in caption_rects if r != caption.rect]
                table_obj = None

                if caption.kind in ("table", "algorithm"):
                    region, table_obj = _table_region(page, caption, tables, neighbours, claimed)
                else:
                    region = _figure_region(page, caption, graphics, neighbours)

                if region is None:
                    continue
                if region.width < MIN_REGION_WIDTH or region.height < MIN_REGION_HEIGHT:
                    continue

                markdown, rows, cols = ("", 0, 0)
                if table_obj is not None:
                    markdown, rows, cols = table_to_markdown(table_obj)

                image_png = None
                if render_images:
                    try:
                        image_png = page.get_pixmap(clip=region, dpi=dpi).tobytes("png")
                    except Exception:  # pragma: no cover - clip can fail on odd pages
                        logger.warning("Could not render %s on page %d", caption.label, page_number)

                results.append(
                    ExtractedFigure(
                        kind=caption.kind,
                        label=caption.label,
                        caption=caption.text,
                        page=page_number,
                        bbox=tuple(round(v, 2) for v in region),
                        image_png=image_png,
                        table_markdown=markdown or None,
                        table_rows=rows,
                        table_cols=cols,
                        detector="table-finder" if table_obj is not None else "caption",
                    )
                )

                if len(results) >= max_figures:
                    logger.info("Figure extraction capped at %d items", max_figures)
                    return results

    return results
