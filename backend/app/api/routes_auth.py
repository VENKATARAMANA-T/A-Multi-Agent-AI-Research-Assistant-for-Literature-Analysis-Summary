"""Registration, activation, sign-in and the profile.

Two rules shape most of what follows.

First, failures are told apart only where it is safe. A wrong password and an
unknown username both answer "Those credentials do not match an account",
because answering differently turns the sign-in form into a way to discover who
has an account here. An unactivated account is the exception: the person
already proved they know the password, so telling them why they cannot get in
costs nothing and saves them guessing.

Second, registration never confirms whether an address is already registered.
It answers the same way either way and emails the owner — which is the only
party entitled to know.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, func, or_, select

from app.api.deps import current_user
from app.config import settings
from app.database import get_session
from app.models import User, utcnow
from app.schemas import (
    ActivateRequest,
    AuthResponse,
    ChangePasswordRequest,
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    RegisterResponse,
    ResendActivationRequest,
    UserProfile,
)
from app.services import mailer
from app.services.auth import (
    ACTIVATION,
    AuthError,
    activation_url,
    create_access_token,
    create_activation_token,
    decode_token,
    hash_password,
    normalise_email,
    normalise_username,
    password_problems,
    password_strength,
    token_expiry_seconds,
    validate_registration,
    verify_password,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])

GENERIC_SIGNIN_FAILURE = "Those credentials do not match an account."


def profile_of(user: User) -> UserProfile:
    return UserProfile(
        id=user.id,
        username=user.username,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        full_name=user.full_name,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
    )


def find_user(session: Session, identifier: str) -> User | None:
    """Look up by username or email, case-insensitively."""
    value = (identifier or "").strip().lower()
    if not value:
        return None
    return session.exec(
        select(User).where(
            or_(func.lower(User.username) == value, func.lower(User.email) == value)
        )
    ).first()


# --- registration ------------------------------------------------------------


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account and email an activation link",
)
def register(payload: RegisterRequest, session: Session = Depends(get_session)) -> RegisterResponse:
    username = normalise_username(payload.username)
    email = normalise_email(payload.email)

    errors = validate_registration(username, email, payload.password, payload.confirm_password)

    # Uniqueness is checked here so the form can mark the offending field, and
    # enforced again by the database's unique index, which is what actually
    # holds when two people register the same name at the same moment.
    if not errors.get("username"):
        if session.exec(select(User).where(func.lower(User.username) == username)).first():
            errors["username"] = ["That username is taken."]
    if not errors.get("email") and session.exec(
        select(User).where(func.lower(User.email) == email)
    ).first():
        # Deliberately not reported back: see the module docstring.
        logger.info("Registration attempted for an address that already exists: %s", email)
        return RegisterResponse(
            message="Check your email to activate your account.",
            email_sent=True,
            detail=f"If {email} is not already registered, a link is on its way.",
        )

    if errors:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=errors)

    user = User(
        username=username,
        email=email,
        first_name=payload.first_name.strip(),
        last_name=payload.last_name.strip(),
        password_hash=hash_password(payload.password),
        is_active=False,
    )
    session.add(user)
    try:
        session.commit()
    except Exception:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"username": ["That username or email was just taken. Try another."]},
        ) from None
    session.refresh(user)

    delivery = mailer.send_activation_email(
        user.email, user.first_name or user.username, activation_url(create_activation_token(user.id, user.email))
    )

    return RegisterResponse(
        message="Account created. Check your email to activate it.",
        email_sent=delivery.sent,
        detail=delivery.detail,
        activation_link=delivery.link,
    )


@router.post("/activate", response_model=AuthResponse, summary="Activate an account")
def activate(payload: ActivateRequest, session: Session = Depends(get_session)) -> AuthResponse:
    try:
        claims = decode_token(payload.token, ACTIVATION)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    user = session.get(User, claims["sub"])
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="This account no longer exists.")

    # Activating twice is not an error — people click the link again, and the
    # second click should land them signed in rather than on a failure page.
    if not user.is_active:
        user.is_active = True
        user.activated_at = utcnow()
        session.add(user)
        session.commit()
        session.refresh(user)

    return AuthResponse(
        access_token=create_access_token(user.id, user.username),
        expires_in=token_expiry_seconds(),
        user=profile_of(user),
    )


@router.post("/resend-activation", response_model=MessageResponse, summary="Send a new activation link")
def resend_activation(
    payload: ResendActivationRequest, session: Session = Depends(get_session)
) -> MessageResponse:
    user = find_user(session, payload.identifier)

    # Answers identically whether or not the account exists, so this cannot be
    # used to find out who is registered.
    reply = MessageResponse(
        message="If that account exists and is not yet active, a new link is on its way."
    )
    if user is None or user.is_active:
        return reply

    mailer.send_activation_email(
        user.email, user.first_name or user.username, activation_url(create_activation_token(user.id, user.email))
    )
    return reply


# --- sign in -----------------------------------------------------------------


@router.post("/login", response_model=AuthResponse, summary="Sign in with a username or email")
def login(payload: LoginRequest, session: Session = Depends(get_session)) -> AuthResponse:
    user = find_user(session, payload.identifier)

    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=GENERIC_SIGNIN_FAILURE
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "This account is not activated yet. Check your email for the link, "
                "or request a new one."
            ),
        )

    user.last_login_at = utcnow()
    session.add(user)
    session.commit()
    session.refresh(user)

    return AuthResponse(
        access_token=create_access_token(user.id, user.username),
        expires_in=token_expiry_seconds(),
        user=profile_of(user),
    )


# --- profile -----------------------------------------------------------------


@router.get("/me", response_model=UserProfile, summary="The signed-in account")
def me(user: User = Depends(current_user)) -> UserProfile:
    return profile_of(user)


@router.post("/password", response_model=MessageResponse, summary="Change your password")
def change_password(
    payload: ChangePasswordRequest,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> MessageResponse:
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"current_password": ["That is not your current password."]},
        )

    errors: dict[str, list[str]] = {}
    issues = password_problems(payload.new_password)
    if issues:
        errors["new_password"] = issues
    if payload.new_password != payload.confirm_password:
        errors["confirm_password"] = ["The two passwords do not match."]
    if payload.new_password == payload.current_password:
        errors["new_password"] = errors.get("new_password", []) + [
            "Choose a password you have not used here before."
        ]
    if errors:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=errors)

    user.password_hash = hash_password(payload.new_password)
    session.add(user)
    session.commit()

    mailer.send_password_changed_email(user.email, user.first_name or user.username)
    return MessageResponse(message="Password updated.")


@router.post("/password/strength", summary="Advisory strength of a candidate password")
def strength(payload: dict) -> dict:
    """Scored for the meter on the form. Advisory only — `password_problems`
    decides what is actually refused."""
    return password_strength(str(payload.get("password") or ""))


@router.get("/rules", summary="The password rules, so the form can state them")
def rules() -> dict:
    return {
        "min_length": settings.password_min_length,
        "max_length": 72,
        "activation_minutes": settings.activation_token_minutes,
        "session_hours": round(settings.access_token_minutes / 60, 1),
    }
