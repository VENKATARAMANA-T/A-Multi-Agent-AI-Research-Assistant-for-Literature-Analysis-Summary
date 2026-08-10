"""Step 3 — document chunking with LangChain's RecursiveCharacterTextSplitter.

Chunks carry page ranges and a section label so citations can point back at a
concrete location in the source PDF.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.services.pdf_extract import ExtractedDocument, SECTION_PATTERNS


@dataclass
class TextChunk:
    index: int
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None

    @property
    def token_estimate(self) -> int:
        # ~4 characters per token is close enough for budgeting prompts.
        return max(1, len(self.text) // 4)


def _page_offsets(document: ExtractedDocument) -> tuple[list[int], list[int]]:
    """Return (cumulative start offset per page, page numbers) over the joined text."""
    offsets: list[int] = []
    numbers: list[int] = []
    cursor = 0
    for page in document.pages:
        offsets.append(cursor)
        numbers.append(page.number)
        cursor += len(page.text) + 1  # +1 for the "\n" used when joining
    return offsets, numbers


def _page_for_offset(offsets: list[int], numbers: list[int], offset: int) -> int | None:
    if not offsets:
        return None
    idx = bisect.bisect_right(offsets, offset) - 1
    idx = max(0, min(idx, len(numbers) - 1))
    return numbers[idx]


def _match_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return None
    for name, pattern in SECTION_PATTERNS:
        if pattern.match(stripped):
            return name
    return None


def _section_for_chunk(text: str, offset: int, chunk_text_value: str) -> str | None:
    """Label a chunk with its section.

    A chunk that opens with a heading belongs to *that* section, not to the one
    before it — so the chunk's own first lines are checked before falling back
    to the nearest preceding heading in the document.
    """
    for line in chunk_text_value.splitlines()[:3]:
        name = _match_heading(line)
        if name:
            return name

    best: str | None = None
    for line in text[:offset].splitlines()[-4000:]:
        name = _match_heading(line)
        if name:
            best = name
    return best


def build_splitter(chunk_size: int | None = None, chunk_overlap: int | None = None) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        length_function=len,
        keep_separator=True,
    )


def chunk_document(
    document: ExtractedDocument,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """Split an extracted document into overlapping, location-aware chunks."""
    text = document.text
    if not text.strip():
        return []

    splitter = build_splitter(chunk_size, chunk_overlap)
    pieces = [p.strip() for p in splitter.split_text(text) if p.strip()]

    offsets, numbers = _page_offsets(document)
    chunks: list[TextChunk] = []
    cursor = 0

    for index, piece in enumerate(pieces):
        # Locate the piece in the source text to derive page/section metadata.
        found = text.find(piece[:120], cursor)
        if found == -1:
            found = text.find(piece[:60], cursor)
        if found == -1:
            found = cursor
        start, end = found, found + len(piece)
        cursor = max(cursor, start + 1)

        chunks.append(
            TextChunk(
                index=index,
                text=piece,
                page_start=_page_for_offset(offsets, numbers, start),
                page_end=_page_for_offset(offsets, numbers, end),
                section=_section_for_chunk(text, start, piece),
            )
        )

    return chunks


def chunk_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
    """Convenience splitter for raw strings (used by map-reduce summarisation)."""
    if not text.strip():
        return []
    return [p.strip() for p in build_splitter(chunk_size, chunk_overlap).split_text(text) if p.strip()]
