"""Locate a chunk's text inside its source PDF, for citation highlighting.

Clicking a citation should land on the exact passage, not merely the right
page. PyMuPDF can search a page for a phrase and return the quads it occupies,
which is far more reliable than trying to re-derive coordinates in the browser
from the PDF.js text layer.

The chunk text cannot be searched verbatim: extraction collapses ligatures,
joins hyphenated line breaks and normalises whitespace, so the stored string
often does not appear in the page byte-for-byte. Search therefore uses short
phrases from the chunk and falls back progressively.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz

logger = logging.getLogger(__name__)

# Long phrases rarely match because of soft hyphens and line breaks; short ones
# match too often. These lengths behave well on real papers.
PHRASE_WORDS = 8
MAX_PHRASES = 6
MAX_RECTS = 60


@dataclass
class Highlight:
    page: int
    rects: list[list[float]] = field(default_factory=list)
    matched_phrase: str | None = None
    page_width: float = 0.0
    page_height: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "rects": self.rects,
            "matched_phrase": self.matched_phrase,
            "page_width": round(self.page_width, 2),
            "page_height": round(self.page_height, 2),
            "found": bool(self.rects),
        }


def candidate_phrases(text: str) -> list[str]:
    """Short, distinctive phrases to search for, most promising first."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []

    # Prefer sentences: they start after a break, so they rarely begin
    # mid-hyphenation.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if len(s.strip()) > 30]

    phrases: list[str] = []
    for sentence in sentences[:MAX_PHRASES]:
        words = sentence.split()
        if len(words) >= 4:
            phrases.append(" ".join(words[:PHRASE_WORDS]))

    words = cleaned.split()
    if len(words) >= 4:
        phrases.append(" ".join(words[:PHRASE_WORDS]))
        if len(words) > PHRASE_WORDS * 2:
            middle = len(words) // 2
            phrases.append(" ".join(words[middle : middle + PHRASE_WORDS]))

    seen: set[str] = set()
    unique = []
    for phrase in phrases:
        key = phrase.lower()
        if key not in seen and len(phrase) > 12:
            seen.add(key)
            unique.append(phrase)
    return unique[:MAX_PHRASES]


def find_highlight(
    pdf_path: str | Path,
    text: str,
    page_number: int | None = None,
    search_window: int = 1,
) -> Highlight:
    """Find `text` in the PDF and return the rectangles it covers.

    `page_number` is 1-based and used as a hint; nearby pages are searched too,
    because a chunk can straddle a page break.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    phrases = candidate_phrases(text)

    with fitz.open(path) as document:
        if not document.page_count:
            return Highlight(page=page_number or 1)

        if page_number and 1 <= page_number <= document.page_count:
            order = [page_number]
            for offset in range(1, search_window + 1):
                for candidate in (page_number + offset, page_number - offset):
                    if 1 <= candidate <= document.page_count:
                        order.append(candidate)
        else:
            order = list(range(1, document.page_count + 1))

        for number in order:
            page = document[number - 1]
            for phrase in phrases:
                try:
                    quads = page.search_for(phrase, quads=False)
                except Exception:  # pragma: no cover - odd page content
                    continue
                if quads:
                    return Highlight(
                        page=number,
                        rects=[[round(v, 2) for v in rect] for rect in quads[:MAX_RECTS]],
                        matched_phrase=phrase,
                        page_width=page.rect.width,
                        page_height=page.rect.height,
                    )

        # Nothing matched: still report the page so the viewer can navigate.
        fallback = page_number or 1
        page = document[min(fallback, document.page_count) - 1]
        return Highlight(
            page=fallback,
            page_width=page.rect.width,
            page_height=page.rect.height,
        )
