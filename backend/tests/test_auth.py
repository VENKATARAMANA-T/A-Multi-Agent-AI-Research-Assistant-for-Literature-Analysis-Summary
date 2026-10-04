"""Accounts, tokens and the isolation between them.

The tests that matter most here are the ones about *other people's data*. An
authentication bug is loud — nobody can sign in. An authorisation bug is silent:
everything works, and one account can read another's papers. So most of this
file is a second account checking that it cannot see the first one's corpus.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest

from app.config import settings
from app.database import session_scope
from app.services.auth import (
    ACCESS,
    ACTIVATION,
    AuthError,
    create_access_token,
    create_activation_token,
    decode_token,
    hash_password,
    password_problems,
    password_strength,
    verify_password,
)
from app.services.bootstrap import SEED_EMAIL, SEED_PASSWORD, SEED_USERNAME
from app.services.ingestion import process_paper, store_upload
from tests.conftest import seed_user_id


def register_and_activate(client, username="ada", email=None, password="analytical1"):
    """Create a second account and return its Authorization header."""
    reply = client.post(
        "/api/auth/register",
        json={
            "first_name": "Ada",
            "last_name": "Lovelace",
            "username": username,
            "email": email or f"{username}@example.com",
            "password": password,
            "confirm_password": password,
        },
    )
    assert reply.status_code == 201, reply.text

    link = reply.json()["activation_link"]
    assert link, "no mail server in tests, so the link must be returned"
    token = link.split("token=")[1]

    activated = client.post("/api/auth/activate", json={"token": token})
    assert activated.status_code == 200, activated.text
    return {"Authorization": f"Bearer {activated.json()['access_token']}"}


# --- passwords ----------------------------------------------------------------


def test_a_password_hash_is_salted_and_verifiable():
    first, second = hash_password("correct horse"), hash_password("correct horse")

    assert first != second, "equal hashes would mean no salt"
    assert verify_password("correct horse", first)
    assert verify_password("correct horse", second)
    assert not verify_password("Correct Horse", first)


def test_verification_survives_a_corrupt_hash():
    assert verify_password("anything", "not-a-bcrypt-hash") is False
    assert verify_password("anything", "") is False


def test_short_passwords_are_refused():
    assert password_problems("abc")
    assert not password_problems("abcdef")


def test_surrounding_whitespace_is_refused():
    """A trailing space is invisible and locks the owner out tomorrow."""
    assert any("spaces" in problem for problem in password_problems("hunter2 "))


def test_composition_rules_do_not_block_a_sign_up():
    """NIST advises against them, and the seed account's own password has no letters."""
    assert password_problems(SEED_PASSWORD) == []


def test_strength_is_advisory_and_marks_the_obvious_ones():
    assert password_strength("123456789")["score"] == 0
    assert password_strength("123456789")["common"] is True
    assert password_strength("Tr0ub4dor&3-horse")["score"] >= 3


# --- tokens -------------------------------------------------------------------


def test_an_activation_token_cannot_be_used_as_a_session():
    """Otherwise a link sitting in an inbox is a working login."""
    token = create_activation_token("user-1", "a@b.cc")

    assert decode_token(token, ACTIVATION)["sub"] == "user-1"
    with pytest.raises(AuthError):
        decode_token(token, ACCESS)


def test_an_expired_token_is_refused_with_a_reason():
    expired = jwt.encode(
        {
            "sub": "user-1",
            "purpose": ACTIVATION,
            "exp": datetime.now(timezone.utc) - timedelta(seconds=5),
        },
        settings.resolved_jwt_secret,
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(AuthError, match="expired"):
        decode_token(expired, ACTIVATION)


def test_a_token_signed_with_another_key_is_refused():
    forged = jwt.encode({"sub": "user-1", "purpose": ACCESS}, "a-different-secret", algorithm="HS256")

    with pytest.raises(AuthError):
        decode_token(forged, ACCESS)


def test_activation_tokens_expire_in_five_minutes():
    claims = jwt.decode(
        create_activation_token("u", "a@b.cc"),
        settings.resolved_jwt_secret,
        algorithms=[settings.jwt_algorithm],
    )
    assert round((claims["exp"] - claims["iat"]) / 60) == settings.activation_token_minutes


# --- the seed account ---------------------------------------------------------


def test_the_seed_account_exists_and_owns_everything(client, sample_pdf):
    with session_scope() as session:
        paper, _ = store_upload(session, sample_pdf.name, sample_pdf.read_bytes(), seed_user_id())
        process_paper(session, paper.id)

    me = client.get("/api/auth/me").json()
    assert me["username"] == SEED_USERNAME
    assert me["email"] == SEED_EMAIL
    assert me["is_active"] is True

    assert len(client.get("/api/papers").json()) == 1


def test_the_seed_account_signs_in_with_the_documented_password(anon_client):
    response = anon_client.post(
        "/api/auth/login", json={"identifier": SEED_USERNAME, "password": SEED_PASSWORD}
    )
    assert response.status_code == 200


def test_pre_existing_data_is_adopted_rather_than_orphaned(client, sample_pdf):
    """Ownership arrived after the data did; a NULL owner belongs to nobody."""
    from app.models import Paper

    with session_scope() as session:
        paper, _ = store_upload(session, sample_pdf.name, sample_pdf.read_bytes(), None)
        process_paper(session, paper.id)
        orphan_id = paper.id

    with session_scope() as session:
        assert session.get(Paper, orphan_id).owner_id is None

    from app.services import bootstrap

    with session_scope() as session:
        moved = bootstrap.run(session)
    assert moved.get("papers") == 1

    assert any(paper["id"] == orphan_id for paper in client.get("/api/papers").json())


# --- sign in ------------------------------------------------------------------


def test_sign_in_works_with_either_username_or_email(anon_client):
    for identifier in (SEED_USERNAME, SEED_EMAIL, SEED_EMAIL.upper(), "USER1"):
        response = anon_client.post(
            "/api/auth/login", json={"identifier": identifier, "password": SEED_PASSWORD}
        )
        assert response.status_code == 200, identifier


def test_a_wrong_password_and_an_unknown_user_look_identical(anon_client):
    """Different answers would turn the form into a way to find out who has an account."""
    wrong = anon_client.post(
        "/api/auth/login", json={"identifier": SEED_USERNAME, "password": "not-it"}
    )
    unknown = anon_client.post(
        "/api/auth/login", json={"identifier": "nobody", "password": "not-it"}
    )

    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"]


def test_an_unactivated_account_is_told_why(anon_client):
    anon_client.post(
        "/api/auth/register",
        json={
            "first_name": "Grace", "last_name": "Hopper", "username": "grace",
            "email": "grace@example.com", "password": "compiler12",
            "confirm_password": "compiler12",
        },
    )

    response = anon_client.post(
        "/api/auth/login", json={"identifier": "grace", "password": "compiler12"}
    )
    assert response.status_code == 403
    assert "activat" in response.json()["detail"].lower()


def test_last_login_is_recorded(anon_client):
    before = anon_client.post(
        "/api/auth/login", json={"identifier": SEED_USERNAME, "password": SEED_PASSWORD}
    ).json()["user"]
    assert before["last_login_at"] is not None


# --- registration -------------------------------------------------------------


def test_registration_emails_a_link_and_activates(anon_client):
    headers = register_and_activate(anon_client)
    assert anon_client.get("/api/auth/me", headers=headers).json()["is_active"] is True


def test_activation_is_idempotent(anon_client):
    reply = anon_client.post(
        "/api/auth/register",
        json={
            "first_name": "Alan", "last_name": "Turing", "username": "alan",
            "email": "alan@example.com", "password": "enigma1234",
            "confirm_password": "enigma1234",
        },
    ).json()
    token = reply["activation_link"].split("token=")[1]

    # People click the link twice; the second click should not be a failure page.
    assert anon_client.post("/api/auth/activate", json={"token": token}).status_code == 200
    assert anon_client.post("/api/auth/activate", json={"token": token}).status_code == 200


def test_a_duplicate_username_is_refused(anon_client):
    register_and_activate(anon_client, username="ada")

    response = anon_client.post(
        "/api/auth/register",
        json={
            "first_name": "Another", "last_name": "Ada", "username": "ADA",
            "email": "other@example.com", "password": "analytical1",
            "confirm_password": "analytical1",
        },
    )
    assert response.status_code == 422
    assert "username" in response.json()["detail"]


def test_a_duplicate_email_is_not_confirmed_to_the_caller(anon_client):
    """Answering differently would let anyone test which addresses are registered."""
    register_and_activate(anon_client, username="ada", email="ada@example.com")

    response = anon_client.post(
        "/api/auth/register",
        json={
            "first_name": "Someone", "last_name": "Else", "username": "someone",
            "email": "ada@example.com", "password": "analytical1",
            "confirm_password": "analytical1",
        },
    )
    assert response.status_code == 201
    assert response.json().get("activation_link") is None


def test_mismatched_confirmation_is_refused(anon_client):
    response = anon_client.post(
        "/api/auth/register",
        json={
            "first_name": "A", "last_name": "B", "username": "mismatch",
            "email": "m@example.com", "password": "abcdef12",
            "confirm_password": "abcdef13",
        },
    )
    assert response.status_code == 422
    assert "confirm_password" in response.json()["detail"]


def test_a_short_password_is_refused(anon_client):
    response = anon_client.post(
        "/api/auth/register",
        json={
            "first_name": "A", "last_name": "B", "username": "shorty",
            "email": "s@example.com", "password": "abc", "confirm_password": "abc",
        },
    )
    assert response.status_code == 422
    assert "password" in response.json()["detail"]


def test_a_malformed_email_is_refused(anon_client):
    response = anon_client.post(
        "/api/auth/register",
        json={
            "first_name": "A", "last_name": "B", "username": "bademail",
            "email": "not-an-email", "password": "abcdef12",
            "confirm_password": "abcdef12",
        },
    )
    assert response.status_code == 422
    assert "email" in response.json()["detail"]


# --- the guard ----------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/papers"),
        ("get", "/api/stats"),
        ("get", "/api/graph"),
        ("get", "/api/figures"),
        ("get", "/api/reports"),
        ("get", "/api/verify"),
        ("get", "/api/lbd/hypotheses"),
        ("get", "/api/agents/runs"),
        ("post", "/api/papers/search"),
        ("post", "/api/agents/ask"),
    ],
)
def test_every_data_endpoint_refuses_an_anonymous_caller(anon_client, method, path):
    response = getattr(anon_client, method)(path, **({"json": {}} if method == "post" else {}))
    assert response.status_code == 401, f"{path} answered {response.status_code}"


def test_health_stays_open(anon_client):
    """It reports whether the backends are reachable, which is not anyone's data."""
    assert anon_client.get("/api/health").status_code == 200


def test_a_forged_token_is_refused(anon_client):
    forged = jwt.encode({"sub": "nobody", "purpose": ACCESS}, "wrong-key", algorithm="HS256")
    response = anon_client.get("/api/papers", headers={"Authorization": f"Bearer {forged}"})
    assert response.status_code == 401


def test_a_token_for_a_deleted_account_stops_working(anon_client):
    from app.models import User

    headers = register_and_activate(anon_client, username="ghost")
    assert anon_client.get("/api/auth/me", headers=headers).status_code == 200

    with session_scope() as session:
        session.delete(session.exec(__import__("sqlmodel").select(User).where(User.username == "ghost")).first())
        session.commit()

    assert anon_client.get("/api/auth/me", headers=headers).status_code == 401


def test_a_malformed_authorization_header_is_refused(anon_client):
    for header in ("", "Bearer", "Basic abc", "Bearer    "):
        response = anon_client.get("/api/papers", headers={"Authorization": header})
        assert response.status_code == 401, header


# --- isolation ----------------------------------------------------------------


def test_one_account_cannot_see_another_corpus(client, anon_client, sample_pdf):
    """The quiet failure: everything works, and the wrong person sees the papers."""
    with session_scope() as session:
        paper, _ = store_upload(session, sample_pdf.name, sample_pdf.read_bytes(), seed_user_id())
        process_paper(session, paper.id)
        paper_id = paper.id

    assert len(client.get("/api/papers").json()) == 1

    other = register_and_activate(anon_client, username="ada")
    assert anon_client.get("/api/papers", headers=other).json() == []


def test_another_account_cannot_open_your_paper_by_id(client, anon_client, sample_pdf):
    with session_scope() as session:
        paper, _ = store_upload(session, sample_pdf.name, sample_pdf.read_bytes(), seed_user_id())
        process_paper(session, paper.id)
        paper_id = paper.id

    other = register_and_activate(anon_client, username="ada")

    for path in (
        f"/api/papers/{paper_id}",
        f"/api/papers/{paper_id}/chunks",
        f"/api/papers/{paper_id}/file",
    ):
        assert anon_client.get(path, headers=other).status_code == 404, path

    assert anon_client.delete(f"/api/papers/{paper_id}", headers=other).status_code == 404


def test_search_does_not_cross_accounts(client, anon_client, sample_pdf):
    """The vector store is one collection, so the scope has to be passed every time."""
    with session_scope() as session:
        paper, _ = store_upload(session, sample_pdf.name, sample_pdf.read_bytes(), seed_user_id())
        process_paper(session, paper.id)

    mine = client.post("/api/papers/search", json={"query": "sparse attention", "top_k": 5})
    assert mine.json()["hits"], "the owner should find their own text"

    other = register_and_activate(anon_client, username="ada")
    theirs = anon_client.post(
        "/api/papers/search", json={"query": "sparse attention", "top_k": 5}, headers=other
    )
    assert theirs.json()["hits"] == []


def test_stats_count_only_your_own(client, anon_client, sample_pdf):
    with session_scope() as session:
        paper, _ = store_upload(session, sample_pdf.name, sample_pdf.read_bytes(), seed_user_id())
        process_paper(session, paper.id)

    assert client.get("/api/stats").json()["papers"] == 1

    other = register_and_activate(anon_client, username="ada")
    assert anon_client.get("/api/stats", headers=other).json()["papers"] == 0


def test_an_agent_run_is_rejected_for_another_account(client, anon_client, sample_pdf, fake_llm):
    with session_scope() as session:
        paper, _ = store_upload(session, sample_pdf.name, sample_pdf.read_bytes(), seed_user_id())
        process_paper(session, paper.id)

    run_id = client.post("/api/agents/ask", json={"question": "What is the method?"}).json()["run_id"]
    assert client.get(f"/api/agents/runs/{run_id}").status_code == 200

    other = register_and_activate(anon_client, username="ada")
    assert anon_client.get(f"/api/agents/runs/{run_id}", headers=other).status_code == 404
    assert anon_client.get("/api/agents/runs", headers=other).json() == []


def test_uploading_the_same_pdf_gives_each_account_its_own_copy(client, anon_client, sample_pdf):
    """Matching on the content hash alone would hand the second person the first one's row."""
    data = sample_pdf.read_bytes()

    mine = client.post(
        "/api/papers/upload?wait=true", files={"files": (sample_pdf.name, data, "application/pdf")}
    ).json()
    assert mine["results"][0]["status"] == "indexed"

    other = register_and_activate(anon_client, username="ada")
    theirs = anon_client.post(
        "/api/papers/upload?wait=true",
        files={"files": (sample_pdf.name, data, "application/pdf")},
        headers=other,
    ).json()

    assert theirs["results"][0]["status"] == "indexed", "not a duplicate — different owner"
    assert theirs["results"][0]["paper_id"] != mine["results"][0]["paper_id"]


# --- changing a password ------------------------------------------------------


def test_changing_a_password_requires_the_current_one(client):
    response = client.post(
        "/api/auth/password",
        json={
            "current_password": "wrong", "new_password": "newpass12",
            "confirm_password": "newpass12",
        },
    )
    assert response.status_code == 400
    assert "current_password" in response.json()["detail"]


def test_a_changed_password_takes_effect(client, anon_client):
    assert client.post(
        "/api/auth/password",
        json={
            "current_password": SEED_PASSWORD, "new_password": "brand-new-99",
            "confirm_password": "brand-new-99",
        },
    ).status_code == 200

    assert anon_client.post(
        "/api/auth/login", json={"identifier": SEED_USERNAME, "password": SEED_PASSWORD}
    ).status_code == 401
    assert anon_client.post(
        "/api/auth/login", json={"identifier": SEED_USERNAME, "password": "brand-new-99"}
    ).status_code == 200


def test_the_new_password_must_be_different(client):
    response = client.post(
        "/api/auth/password",
        json={
            "current_password": SEED_PASSWORD, "new_password": SEED_PASSWORD,
            "confirm_password": SEED_PASSWORD,
        },
    )
    assert response.status_code == 422


def test_the_new_password_must_be_confirmed(client):
    response = client.post(
        "/api/auth/password",
        json={
            "current_password": SEED_PASSWORD, "new_password": "abcdefgh1",
            "confirm_password": "abcdefgh2",
        },
    )
    assert response.status_code == 422
    assert "confirm_password" in response.json()["detail"]


def test_a_short_new_password_is_refused(client):
    response = client.post(
        "/api/auth/password",
        json={"current_password": SEED_PASSWORD, "new_password": "abc", "confirm_password": "abc"},
    )
    assert response.status_code == 422


# --- resending ----------------------------------------------------------------


def test_resending_never_reveals_whether_an_account_exists(anon_client):
    unknown = anon_client.post("/api/auth/resend-activation", json={"identifier": "nobody"})
    known = anon_client.post("/api/auth/resend-activation", json={"identifier": SEED_USERNAME})

    assert unknown.status_code == known.status_code == 200
    assert unknown.json() == known.json()
