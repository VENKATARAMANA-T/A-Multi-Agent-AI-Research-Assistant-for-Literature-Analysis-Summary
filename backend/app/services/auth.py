"""Passwords, tokens and the rules that guard an account.

Two kinds of token are issued here and they are deliberately not
interchangeable. An access token proves who you are for the length of a
session; an activation token proves only that you control an inbox, lives five
minutes, and must never be usable as a login. Both are signed with the same
key, so the `purpose` claim is what keeps them apart — without it, an
activation link sitting in an inbox would be a valid session credential.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

from app.config import settings

logger = logging.getLogger(__name__)

ACCESS = "access"
ACTIVATION = "activation"

USERNAME_RE = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9._-]{1,30})[a-zA-Z0-9]$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]{2,}$")

# bcrypt hashes at most 72 bytes and silently ignores the rest, so a longer
# password would be truncated without anyone noticing.
MAX_PASSWORD_BYTES = 72


class AuthError(Exception):
    """A problem the user can fix, carrying a message safe to display."""


# --- passwords ---------------------------------------------------------------


def hash_password(password: str) -> str:
    encoded = password.encode("utf-8")[:MAX_PASSWORD_BYTES]
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(
            password.encode("utf-8")[:MAX_PASSWORD_BYTES], password_hash.encode("utf-8")
        )
    except ValueError:  # a malformed or truncated hash in the database
        return False


COMMON_PASSWORDS = frozenset(
    {
        "password", "password1", "12345678", "123456789", "1234567890",
        "qwerty123", "letmein", "welcome1", "admin123", "iloveyou",
    }
)


def password_problems(password: str) -> list[str]:
    """Every reason a password is *rejected*, so the form can show them at once.

    Returning a list rather than the first failure is the point: fixing one rule
    only to be told about the next is what makes sign-up forms miserable.

    Only two things block here — length and stray whitespace. Composition rules
    ("must contain a digit and a symbol") are what NIST SP 800-63B specifically
    advises against: they push people towards `Password1!` and no further.
    Everything else is advisory and lives in `password_strength`, which lets the
    form say a password is weak without refusing it.
    """
    problems: list[str] = []
    minimum = settings.password_min_length

    if len(password) < minimum:
        problems.append(f"Use at least {minimum} characters.")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        problems.append(f"Keep it under {MAX_PASSWORD_BYTES} characters.")
    if password != password.strip():
        problems.append("Remove the spaces at the start or end.")

    return problems


def password_strength(password: str) -> dict[str, Any]:
    """An advisory score, shown as a meter rather than used to refuse a sign-up."""
    suggestions: list[str] = []
    score = 0

    if len(password) >= 8:
        score += 1
    else:
        suggestions.append("Make it longer — 12 characters beats any clever substitution.")
    if len(password) >= 12:
        score += 1
    if re.search(r"[A-Za-z]", password) and re.search(r"\d", password):
        score += 1
    else:
        suggestions.append("Mix letters and numbers.")
    if re.search(r"[^A-Za-z0-9]", password):
        score += 1
    else:
        suggestions.append("A symbol or a space adds more than another letter does.")

    # A sequential run is long but trivially guessable, so it cancels the
    # credit that length alone earned.
    sequential = any(
        password[i : i + 4] in "0123456789abcdefghijklmnopqrstuvwxyz"
        for i in range(max(0, len(password) - 3))
    )
    if sequential:
        score = min(score, 1)
        suggestions.append("Avoid runs like 1234 or abcd — they are guessed first.")
    if password.lower() in COMMON_PASSWORDS:
        score = 0
        suggestions.append("This is one of the most commonly guessed passwords.")

    labels = {0: "very weak", 1: "weak", 2: "fair", 3: "good", 4: "strong"}
    return {
        "score": score,
        "label": labels[min(score, 4)],
        "suggestions": suggestions,
        "common": password.lower() in COMMON_PASSWORDS,
    }


def normalise_username(value: str) -> str:
    return (value or "").strip().lower()


def normalise_email(value: str) -> str:
    return (value or "").strip().lower()


def validate_registration(
    username: str, email: str, password: str, confirm_password: str
) -> dict[str, list[str]]:
    """Field-by-field errors, keyed by field so the form can mark each input."""
    errors: dict[str, list[str]] = {}

    if not USERNAME_RE.match(username):
        errors["username"] = [
            "Use 3-32 characters: letters, numbers, dots, underscores or hyphens, "
            "starting and ending with a letter or number."
        ]
    if not EMAIL_RE.match(email):
        errors["email"] = ["That does not look like an email address."]

    issues = password_problems(password)
    if issues:
        errors["password"] = issues
    if password != confirm_password:
        errors["confirm_password"] = ["The two passwords do not match."]

    return errors


# --- tokens ------------------------------------------------------------------


def _encode(subject: str, purpose: str, minutes: int, **claims: Any) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "purpose": purpose,
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
        **claims,
    }
    return jwt.encode(payload, settings.resolved_jwt_secret, algorithm=settings.jwt_algorithm)


def create_access_token(user_id: str, username: str) -> str:
    return _encode(user_id, ACCESS, settings.access_token_minutes, username=username)


def create_activation_token(user_id: str, email: str) -> str:
    return _encode(user_id, ACTIVATION, settings.activation_token_minutes, email=email)


def decode_token(token: str, expected_purpose: str) -> dict[str, Any]:
    """Decode and check a token, or raise AuthError with a usable reason."""
    try:
        payload = jwt.decode(
            token, settings.resolved_jwt_secret, algorithms=[settings.jwt_algorithm]
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError(
            "This link has expired. Activation links are valid for "
            f"{settings.activation_token_minutes} minutes — request a new one."
            if expected_purpose == ACTIVATION
            else "Your session has expired. Please sign in again."
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("This link is not valid.") from exc

    if payload.get("purpose") != expected_purpose:
        # An activation link must not double as a session token.
        raise AuthError("This token cannot be used here.")
    if not payload.get("sub"):
        raise AuthError("This token is missing its subject.")
    return payload


def token_expiry_seconds() -> int:
    return settings.access_token_minutes * 60


def activation_url(token: str) -> str:
    base = settings.app_base_url.rstrip("/")
    return f"{base}/activate?token={token}"
