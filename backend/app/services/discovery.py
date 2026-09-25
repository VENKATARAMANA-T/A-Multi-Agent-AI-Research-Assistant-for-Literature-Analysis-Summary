"""Related-paper discovery via OpenAlex.

A closed corpus can only tell you about what you already have. OpenAlex is a
free, open catalogue of scholarly works with no API key required, so it can
surface the papers *missing* from a corpus — which is exactly what a literature
review needs and what the gap agent cannot see.

Results are suggestions from an external service, so they are returned as
candidates to review and import, never written into the corpus automatically.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

OPENALEX = "https://api.openalex.org"
TIMEOUT = 20.0


@dataclass
class Candidate:
    """A paper OpenAlex knows about that is not (necessarily) in the corpus."""

    external_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    abstract: str | None = None
    citations: int = 0
    open_access_url: str | None = None
    relation: str = "related"   # related | cited_by | references | search
    source: str = "openalex"

    def to_dict(self) -> dict[str, Any]:
        return {
            "external_id": self.external_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "abstract": self.abstract,
            "citations": self.citations,
            "open_access_url": self.open_access_url,
            "relation": self.relation,
            "source": self.source,
        }


class DiscoveryUnavailable(RuntimeError):
    """OpenAlex could not be reached."""


def _headers() -> dict[str, str]:
    # OpenAlex asks callers to identify themselves; a contact address puts the
    # request in their faster, more reliable pool.
    agent = "ResearchCompass/1.0"
    if settings.openalex_contact_email:
        agent += f" (mailto:{settings.openalex_contact_email})"
    return {"User-Agent": agent}


def _invert_abstract(index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex stores abstracts as an inverted index; rebuild the text."""
    if not index:
        return None
    positions: list[tuple[int, str]] = []
    for word, spots in index.items():
        for spot in spots:
            positions.append((spot, word))
    if not positions:
        return None
    positions.sort()
    text = " ".join(word for _, word in positions)
    return re.sub(r"\s+", " ", text).strip()[:4000] or None


def _to_candidate(work: dict[str, Any], relation: str) -> Candidate | None:
    title = (work.get("display_name") or "").strip()
    if not title:
        return None

    authors = []
    for authorship in work.get("authorships") or []:
        name = ((authorship.get("author") or {}).get("display_name") or "").strip()
        if name:
            authors.append(name)

    location = work.get("primary_location") or {}
    venue = ((location.get("source") or {}) or {}).get("display_name")

    best_oa = work.get("best_oa_location") or {}
    oa_url = best_oa.get("pdf_url") or best_oa.get("landing_page_url")

    doi = (work.get("doi") or "").replace("https://doi.org/", "") or None

    return Candidate(
        external_id=str(work.get("id") or "").rsplit("/", 1)[-1],
        title=title,
        authors=authors[:25],
        year=work.get("publication_year"),
        venue=venue,
        doi=doi,
        abstract=_invert_abstract(work.get("abstract_inverted_index")),
        citations=int(work.get("cited_by_count") or 0),
        open_access_url=oa_url,
        relation=relation,
    )


def _get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        response = httpx.get(
            f"{OPENALEX}{path}", params=params, headers=_headers(), timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        raise DiscoveryUnavailable(f"Could not reach OpenAlex: {exc}") from exc


SELECT = (
    "id,display_name,publication_year,cited_by_count,doi,authorships,"
    "primary_location,best_oa_location,abstract_inverted_index,"
    "referenced_works,related_works"
)


def find_work(title: str | None = None, doi: str | None = None) -> dict[str, Any] | None:
    """Locate a paper in OpenAlex by DOI (exact) or title (best match)."""
    if doi:
        try:
            return _get(f"/works/https://doi.org/{doi}", {"select": SELECT})
        except DiscoveryUnavailable:
            pass  # fall through to a title search

    if not title:
        return None

    payload = _get(
        "/works",
        {"search": title[:250], "per-page": 1, "select": SELECT},
    )
    results = payload.get("results") or []
    return results[0] if results else None


def related_to(
    title: str | None = None,
    doi: str | None = None,
    limit: int = 15,
) -> list[Candidate]:
    """Papers OpenAlex links to this one, plus its references."""
    work = find_work(title=title, doi=doi)
    if work is None:
        return []

    wanted = [str(w).rsplit("/", 1)[-1] for w in (work.get("related_works") or [])][: limit]
    references = [str(w).rsplit("/", 1)[-1] for w in (work.get("referenced_works") or [])][: limit]

    candidates: list[Candidate] = []
    for ids, relation in ((wanted, "related"), (references, "references")):
        if not ids:
            continue
        payload = _get(
            "/works",
            {
                "filter": f"openalex_id:{'|'.join(ids)}",
                "per-page": min(len(ids), 50),
                "select": SELECT,
            },
        )
        for item in payload.get("results") or []:
            candidate = _to_candidate(item, relation)
            if candidate:
                candidates.append(candidate)

    candidates.sort(key=lambda c: (-c.citations, c.title))
    return candidates[:limit]


def search(query: str, limit: int = 15, year_from: int | None = None) -> list[Candidate]:
    """Free-text search across OpenAlex."""
    params: dict[str, Any] = {
        "search": query[:250],
        "per-page": min(max(limit, 1), 50),
        "select": SELECT,
        "sort": "relevance_score:desc",
    }
    if year_from:
        params["filter"] = f"from_publication_date:{year_from}-01-01"

    payload = _get("/works", params)
    results = []
    for item in payload.get("results") or []:
        candidate = _to_candidate(item, "search")
        if candidate:
            results.append(candidate)
    return results[:limit]


def deduplicate(candidates: list[Candidate], known_dois: set[str], known_titles: set[str]) -> list[Candidate]:
    """Drop candidates already present in the corpus."""
    normalised_titles = {re.sub(r"[^a-z0-9]+", "", t.lower()) for t in known_titles if t}
    lowered_dois = {d.lower() for d in known_dois if d}

    unique: list[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.doi and candidate.doi.lower() in lowered_dois:
            continue
        key = re.sub(r"[^a-z0-9]+", "", candidate.title.lower())
        if key in normalised_titles or key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique
