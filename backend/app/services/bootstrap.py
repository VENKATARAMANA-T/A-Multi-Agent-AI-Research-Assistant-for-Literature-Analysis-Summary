"""Give pre-existing data an owner.

Ownership arrived after the data did. Every paper, run and report created
before accounts existed has `owner_id = NULL`, and a NULL owner matches no
user's query — so without this the corpus would simply vanish from the UI the
moment sign-in was switched on.

So on startup: ensure the seed account exists, then adopt every orphaned row
into it. Adoption is deliberately one-way and runs only against NULLs, so it
cannot move a row that already belongs to somebody.
"""

from __future__ import annotations

import logging

from sqlmodel import Session, func, select

from app.models import (
    AgentRun,
    Conversation,
    Hypothesis,
    MatrixRun,
    Paper,
    Report,
    User,
    VerificationRecord,
    utcnow,
)
from app.services.auth import hash_password

logger = logging.getLogger(__name__)

SEED_USERNAME = "user1"
SEED_EMAIL = "user1@gmail.com"
SEED_PASSWORD = "123456789"
SEED_FIRST_NAME = "user1"
SEED_LAST_NAME = "user1"

OWNED_TABLES = (Paper, AgentRun, Conversation, Report, MatrixRun, Hypothesis, VerificationRecord)


def ensure_seed_user(session: Session) -> User:
    """The account that owns everything built before accounts existed.

    Pre-activated: there was no email to confirm, and leaving it inactive would
    lock the owner out of their own corpus.
    """
    existing = session.exec(
        select(User).where(func.lower(User.username) == SEED_USERNAME)
    ).first()
    if existing is not None:
        return existing

    user = User(
        username=SEED_USERNAME,
        email=SEED_EMAIL,
        first_name=SEED_FIRST_NAME,
        last_name=SEED_LAST_NAME,
        password_hash=hash_password(SEED_PASSWORD),
        is_active=True,
        activated_at=utcnow(),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    logger.info("Created the seed account %s <%s>", user.username, user.email)
    return user


def adopt_orphans(session: Session, owner: User) -> dict[str, int]:
    """Assign every ownerless row to `owner`, and report what moved."""
    moved: dict[str, int] = {}

    for model in OWNED_TABLES:
        rows = list(session.exec(select(model).where(model.owner_id.is_(None))).all())  # type: ignore[attr-defined]
        if not rows:
            continue
        for row in rows:
            row.owner_id = owner.id
            session.add(row)
        moved[model.__tablename__] = len(rows)

    if moved:
        session.commit()
        logger.info(
            "Adopted pre-existing data into %s: %s",
            owner.username,
            ", ".join(f"{count} {name}" for name, count in sorted(moved.items())),
        )
    return moved


def run(session: Session) -> dict[str, int]:
    return adopt_orphans(session, ensure_seed_user(session))
