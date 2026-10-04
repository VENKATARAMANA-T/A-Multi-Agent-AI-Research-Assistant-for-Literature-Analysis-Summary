"""Related-paper discovery via OpenAlex.

Everything here is a *suggestion* from an external catalogue. Candidates are
returned for review, never written into the corpus automatically.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from app.api.deps import current_user, owned_paper
from app.config import settings
from app.database import get_session
from app.models import Paper, User
from app.schemas import DiscoveryResponse
from app.services import discovery

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/discover", tags=["discovery"])


def _known(session: Session, user: User) -> tuple[set[str], set[str]]:
    papers = list(session.exec(select(Paper).where(Paper.owner_id == user.id)).all())
    dois = {paper.doi for paper in papers if paper.doi}
    titles = {paper.title for paper in papers if paper.title}
    return dois, titles


def _guard() -> None:
    if not settings.discovery_enabled:
        raise HTTPException(status_code=503, detail="Discovery is disabled (DISCOVERY_ENABLED=false).")


@router.get("/related", response_model=DiscoveryResponse, summary="Papers related to one you have")
def related(
    paper_id: str = Query(...),
    limit: int = Query(default=15, ge=1, le=50),
    exclude_known: bool = Query(default=True),
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> DiscoveryResponse:
    _guard()
    paper = owned_paper(session, user, paper_id)

    try:
        candidates = discovery.related_to(title=paper.title, doi=paper.doi, limit=limit)
    except discovery.DiscoveryUnavailable as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if exclude_known:
        dois, titles = _known(session, user)
        candidates = discovery.deduplicate(candidates, dois, titles)

    return DiscoveryResponse(
        query=paper.title or paper.filename,
        source="openalex",
        candidates=[candidate.to_dict() for candidate in candidates],
        excluded_known=exclude_known,
    )


@router.get("/search", response_model=DiscoveryResponse, summary="Search the open literature")
def search(
    q: str = Query(..., min_length=3, max_length=250),
    limit: int = Query(default=15, ge=1, le=50),
    year_from: int | None = Query(default=None, ge=1800, le=2100),
    exclude_known: bool = Query(default=True),
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> DiscoveryResponse:
    _guard()
    try:
        candidates = discovery.search(q, limit=limit, year_from=year_from)
    except discovery.DiscoveryUnavailable as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if exclude_known:
        dois, titles = _known(session, user)
        candidates = discovery.deduplicate(candidates, dois, titles)

    return DiscoveryResponse(
        query=q,
        source="openalex",
        candidates=[candidate.to_dict() for candidate in candidates],
        excluded_known=exclude_known,
    )


@router.get("/gaps", response_model=DiscoveryResponse, summary="Literature your corpus is missing")
def gaps(
    paper_ids: list[str] | None = Query(
        default=None, description="Base the search on these papers; defaults to the whole corpus."
    ),
    limit: int = Query(default=20, ge=1, le=50),
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> DiscoveryResponse:
    """Related work across a set of papers, ranked by how often it recurs.

    A paper that several of yours relate to, but which you do not have, is the
    most likely thing to be missing from the review. Which papers were asked is
    reported back in `query`, because "missing from your corpus" means nothing
    without knowing what it was compared against.
    """
    _guard()
    papers = [
        p
        for p in session.exec(select(Paper).where(Paper.owner_id == user.id)).all()
        if p.title
    ]
    if paper_ids:
        wanted = set(paper_ids)
        missing = wanted - {paper.id for paper in papers}
        if missing:
            raise HTTPException(
                status_code=404, detail=f"Unknown paper id(s): {', '.join(sorted(missing))}"
            )
        papers = [paper for paper in papers if paper.id in wanted]
    if not papers:
        raise HTTPException(status_code=409, detail="No papers in the corpus yet.")

    tally: dict[str, list] = {}
    reached = 0
    for paper in papers[:12]:  # bound the number of external calls
        try:
            for candidate in discovery.related_to(title=paper.title, doi=paper.doi, limit=15):
                tally.setdefault(candidate.external_id, [candidate, 0])[1] += 1
            reached += 1
        except discovery.DiscoveryUnavailable:
            continue

    if not reached:
        raise HTTPException(status_code=502, detail="Could not reach OpenAlex.")

    dois, titles = _known(session, user)
    ranked = sorted(tally.values(), key=lambda pair: (-pair[1], -pair[0].citations))
    candidates = discovery.deduplicate([pair[0] for pair in ranked], dois, titles)

    counts = {pair[0].external_id: pair[1] for pair in ranked}
    payload = []
    for candidate in candidates[:limit]:
        item = candidate.to_dict()
        item["referenced_by_corpus"] = counts.get(candidate.external_id, 1)
        payload.append(item)

    searched = [paper.title for paper in papers[:12] if paper.title]
    return DiscoveryResponse(
        query=f"related work across {reached} paper(s)",
        source="openalex",
        candidates=payload,
        excluded_known=True,
        searched_papers=searched,
    )
