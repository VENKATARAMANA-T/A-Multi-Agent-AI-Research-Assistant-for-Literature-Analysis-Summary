"""Step 2 — text + metadata extraction from PDFs (PyMuPDF).

Metadata is recovered from three sources, in order of trust:
  1. the PDF's own XMP/Info dictionary,
  2. heuristics over the first page's text layout (largest font block = title),
  3. regex sweeps for DOI / year / abstract.

Everything is best-effort: a paper that yields no metadata still ingests fine,
and the extraction agent can enrich it later with the LLM.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
ARXIV_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

SECTION_PATTERNS = [
    ("abstract", re.compile(r"^\s*(?:\d+\.?\s*)?abstract\b", re.IGNORECASE)),
    ("introduction", re.compile(r"^\s*(?:\d+\.?\s*)?introduction\b", re.IGNORECASE)),
    ("related_work", re.compile(r"^\s*(?:\d+\.?\s*)?(?:related work|background|literature review)\b", re.IGNORECASE)),
    ("method", re.compile(r"^\s*(?:\d+\.?\s*)?(?:method|methodology|approach|model|proposed\b.*)\b", re.IGNORECASE)),
    ("experiments", re.compile(r"^\s*(?:\d+\.?\s*)?(?:experiment|experimental setup|evaluation|dataset)s?\b", re.IGNORECASE)),
    ("results", re.compile(r"^\s*(?:\d+\.?\s*)?(?:results?|findings)\b", re.IGNORECASE)),
    ("discussion", re.compile(r"^\s*(?:\d+\.?\s*)?discussion\b", re.IGNORECASE)),
    ("limitations", re.compile(r"^\s*(?:\d+\.?\s*)?(?:limitations?|threats to validity)\b", re.IGNORECASE)),
    ("conclusion", re.compile(r"^\s*(?:\d+\.?\s*)?(?:conclusions?|future work|concluding remarks)\b", re.IGNORECASE)),
    ("references", re.compile(r"^\s*(?:\d+\.?\s*)?(?:references|bibliography)\b", re.IGNORECASE)),
]

# Lines that look like affiliations/emails rather than author names.
AFFILIATION_HINTS = (
    "university",
    "institute",
    "department",
    "laborator",
    "college",
    "school of",
    "inc.",
    "ltd",
    "gmbh",
    "research",
    "academy",
)


@dataclass
class PageText:
    number: int  # 1-based
    text: str


@dataclass
class ExtractedDocument:
    text: str
    pages: list[PageText] = field(default_factory=list)
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    abstract: str | None = None
    year: int | None = None
    doi: str | None = None
    venue: str | None = None
    keywords: list[str] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return len(self.text)


def _clean(value: str | None) -> str | None:
    if not value:
        return None
    collapsed = re.sub(r"\s+", " ", value).strip(" \t\n\r-—·|")
    return collapsed or None


def _largest_font_title(page: "fitz.Page") -> str | None:
    """Pick the text block with the biggest font on page 1; that is usually the title."""
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:  # pragma: no cover - malformed PDFs
        return None

    candidates: list[tuple[float, float, str]] = []
    for block in blocks:
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            size = max(span.get("size", 0) for span in spans)
            text = " ".join(span.get("text", "") for span in spans).strip()
            if not text or len(text) < 4:
                continue
            y = line.get("bbox", [0, 0, 0, 0])[1]
            candidates.append((size, y, text))

    if not candidates:
        return None

    max_size = max(size for size, _, _ in candidates)
    # Merge consecutive lines that share the dominant font size (multi-line titles).
    title_lines = [text for size, _, text in candidates if size >= max_size - 0.5]
    title = _clean(" ".join(title_lines[:4]))
    if not title or len(title) < 6:
        return None
    if title.lower().startswith(("arxiv:", "doi:", "http")):
        return None
    return title


def _guess_authors(first_page_text: str, title: str | None) -> list[str]:
    lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
    start = 0
    if title:
        title_head = title.lower()[:30]
        for i, line in enumerate(lines[:15]):
            if line.lower()[:30] in title_head or title_head.startswith(line.lower()[:30]):
                start = i + 1
                break

    for line in lines[start : start + 8]:
        lowered = line.lower()
        if lowered.startswith("abstract"):
            break
        if EMAIL_RE.search(line) or any(hint in lowered for hint in AFFILIATION_HINTS):
            continue
        # Strip affiliation markers: superscripts, digits, asterisks, daggers.
        cleaned = re.sub(r"[\d\*†‡§¶#]+", "", line)
        cleaned = re.sub(r"\s+and\s+", ", ", cleaned, flags=re.IGNORECASE)
        parts = [p.strip(" ,;") for p in cleaned.split(",")]
        names = [
            p
            for p in parts
            if 3 < len(p) < 60 and " " in p and re.match(r"^[A-Z]", p) and not any(c.isdigit() for c in p)
        ]
        if len(names) >= 1 and len(names) <= 30:
            return names[:25]
    return []


def _extract_abstract(text: str) -> str | None:
    match = re.search(
        r"abstract\b[\s:.\-—]*(.{80,4000}?)"
        r"(?:\n\s*(?:\d+\.?\s*)?(?:introduction|keywords|index terms|categories and subject)\b|\n\s*\n\s*\n)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return _clean(match.group(1))
    # Fallback: first substantial paragraph on page 1.
    match = re.search(r"abstract\b[\s:.\-—]*(.{80,2500})", text, re.IGNORECASE | re.DOTALL)
    return _clean(match.group(1)) if match else None


def _extract_keywords(text: str) -> list[str]:
    match = re.search(
        r"(?:keywords|index terms)\b[\s:.\-—]*(.{5,400}?)(?:\n\s*\n|\n\s*(?:\d+\.?\s*)?introduction\b)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    raw = re.sub(r"\s+", " ", match.group(1))
    parts = re.split(r"[;,·•]|\s{2,}", raw)
    return [p.strip(" .").title() for p in parts if 2 < len(p.strip()) < 60][:15]


def split_sections(text: str) -> dict[str, str]:
    """Slice full text into canonical sections by scanning for heading lines."""
    lines = text.splitlines()
    marks: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 90:
            continue
        for name, pattern in SECTION_PATTERNS:
            if pattern.match(stripped):
                marks.append((i, name))
                break

    if not marks:
        return {}

    sections: dict[str, str] = {}
    for idx, (line_no, name) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(lines)
        body = "\n".join(lines[line_no + 1 : end]).strip()
        if body and name not in sections:
            sections[name] = body
    return sections


def extract_pdf(path: str | Path) -> ExtractedDocument:
    """Extract text, page map and metadata from a PDF file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    with fitz.open(path) as doc:
        pages = [PageText(number=i + 1, text=page.get_text("text") or "") for i, page in enumerate(doc)]
        pdf_meta = dict(doc.metadata or {})
        first_page_title = _largest_font_title(doc[0]) if doc.page_count else None

    full_text = "\n".join(p.text for p in pages)
    first_page_text = pages[0].text if pages else ""

    title = _clean(pdf_meta.get("title"))
    if not title or len(title) < 8 or title.lower().endswith((".pdf", ".dvi", ".tex")):
        title = first_page_title or title
    if not title:
        # Last resort: first non-trivial line of page 1.
        for line in first_page_text.splitlines():
            candidate = _clean(line)
            if candidate and len(candidate) > 15:
                title = candidate
                break
    if not title:
        title = path.stem.replace("_", " ").replace("-", " ").strip()

    authors: list[str] = []
    meta_author = _clean(pdf_meta.get("author"))
    if meta_author and not EMAIL_RE.search(meta_author):
        authors = [a.strip() for a in re.split(r",| and |;", meta_author) if 3 < len(a.strip()) < 60]
    if not authors:
        authors = _guess_authors(first_page_text, title)

    abstract = _extract_abstract(first_page_text) or _extract_abstract(full_text[:20000])

    doi_match = DOI_RE.search(full_text[:20000])
    doi = doi_match.group(0).rstrip(".") if doi_match else None

    year = None
    header = first_page_text[:3000]
    arxiv = ARXIV_RE.search(header)
    if arxiv:
        yy = arxiv.group(1)[:2]
        year = 2000 + int(yy)
    if year is None:
        candidates = [int(m.group(0)) for m in YEAR_RE.finditer(header)]
        candidates = [c for c in candidates if 1950 <= c <= 2100]
        if candidates:
            year = max(candidates)
    if year is None:
        raw_date = pdf_meta.get("creationDate") or ""
        m = YEAR_RE.search(raw_date)
        if m:
            year = int(m.group(0))

    venue = None
    venue_match = re.search(
        r"(Proceedings of[^\n]{5,120}|In Proc\.[^\n]{5,120}|(?:arXiv preprint[^\n]{0,60}))",
        header,
        re.IGNORECASE,
    )
    if venue_match:
        venue = _clean(venue_match.group(1))

    return ExtractedDocument(
        text=full_text,
        pages=pages,
        title=title,
        authors=authors,
        abstract=abstract,
        year=year,
        doi=doi,
        venue=venue,
        keywords=_extract_keywords(first_page_text),
        sections=split_sections(full_text),
    )
