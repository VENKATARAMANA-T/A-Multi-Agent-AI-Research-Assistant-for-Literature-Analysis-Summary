"""Shared FastAPI dependencies — chiefly, who is making the request.

Every data endpoint resolves the caller through `current_user` and scopes its
queries to that account. The scoping is done at the query, not by filtering
afterwards: a corpus-wide default that forgets the owner would quietly show one
person another's papers, and that is not the kind of bug a test notices.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from sqlmodel import Session, select

from app.database import get_session
from app.models import User
from app.services.auth import ACCESS, AuthError, decode_token

UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Sign in to continue.",
    headers={"WWW-Authenticate": "Bearer"},
)


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("Authorization") or ""
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def current_user(
    request: Request, session: Session = Depends(get_session)
) -> User:
    """The signed-in account, or 401."""
    token = _bearer_token(request)
    if not token:
        raise UNAUTHENTICATED

    try:
        payload = decode_token(token, ACCESS)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user = session.get(User, payload["sub"])
    if user is None:
        # The token is valid but the account is gone — a deleted user must not
        # keep a working session until the token happens to expire.
        raise UNAUTHENTICATED
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has not been activated yet. Check your email for the link.",
        )
    return user


def owned_paper_ids(
    session: Session,
    user: User,
    requested: list[str] | None = None,
    *,
    indexed_only: bool = True,
    minimum: int = 0,
) -> list[str]:
    """Resolve the papers a request may touch, for this user only.

    The owner filter is part of every branch, including the "no ids given"
    default. A default that means "the whole corpus" is exactly where a
    multi-user bug hides: it reads naturally, it passes every single-user test,
    and it quietly hands one account another's papers.

    A paper belonging to someone else is reported as unknown rather than
    forbidden — "that id exists but is not yours" confirms it exists.
    """
    from app.models import Paper, PaperStatus

    statement = select(Paper).where(Paper.owner_id == user.id)
    if requested:
        statement = statement.where(Paper.id.in_(requested))  # type: ignore[attr-defined]

    found = list(session.exec(statement).all())

    if requested:
        missing = set(requested) - {paper.id for paper in found}
        if missing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown paper id(s): {', '.join(sorted(missing))}",
            )

    if indexed_only:
        unindexed = [p.id for p in found if p.status != PaperStatus.INDEXED]
        if requested and unindexed:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"These papers are not indexed yet: {', '.join(unindexed)}",
            )
        found = [p for p in found if p.status == PaperStatus.INDEXED]

    resolved = [paper.id for paper in found]
    if minimum and len(resolved) < minimum:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"This analysis needs at least {minimum} indexed paper(s); found {len(resolved)}.",
        )
    return resolved


def owned_paper(session: Session, user: User, paper_id: str):
    """One paper belonging to this user, or 404."""
    from app.models import Paper

    paper = session.get(Paper, paper_id)
    if paper is None or paper.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper not found.")
    return paper


def optional_user(
    request: Request, session: Session = Depends(get_session)
) -> User | None:
    """The signed-in account if there is one, for endpoints that work either way."""
    try:
        return current_user(request, session)
    except HTTPException:
        return None
