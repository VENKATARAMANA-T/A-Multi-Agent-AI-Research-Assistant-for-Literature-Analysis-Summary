"""Phase 7: the Verification Agent.

The property worth defending here is independence. A verifier handed the same
context that produced the text will agree with itself, and a check that always
passes is worse than no check — so several of these tests exist purely to prove
that the evidence is re-retrieved from the corpus, one query per claim.
"""

from __future__ import annotations

import pytest

from app.agents import verification
from app.agents.verification import (
    VERDICT_SCORES,
    extract_claims,
    extract_text,
    judge_claims,
    verify,
)
from app.database import session_scope
from app.services.ingestion import process_paper, store_upload
from app.services.llm import LLMError, set_llm
from tests.conftest import AGENT_RESPONSES, FakeGemini


def ingest(path) -> str:
    with session_scope() as session:
        paper, _ = store_upload(session, path.name, path.read_bytes())
        process_paper(session, paper.id)
        return paper.id


ANSWER = (
    "SparseSum reaches 44.1 ROUGE-L on arXiv, and it was also evaluated on "
    "Portuguese legal documents."
)


# --- independence -------------------------------------------------------------


def test_each_claim_is_retrieved_for_separately(sample_pdf, fake_llm, monkeypatch):
    """The verifier searches the corpus with the claim, not with the original text.

    This is the whole design: reusing the first agent's context would make the
    check a restatement of it.
    """
    ingest(sample_pdf)

    queries: list[str] = []
    real_search = verification.gather_evidence

    from app.agents import retrieval

    original = retrieval.search

    def recording_search(question, paper_ids, top_k):
        queries.append(question)
        return original(question, paper_ids, top_k)

    monkeypatch.setattr(retrieval, "search", recording_search)

    result = verify(ANSWER)

    assert real_search is verification.gather_evidence  # not monkeypatched away
    assert queries == [
        "SparseSum reaches 44.1 ROUGE-L on arXiv.",
        "SparseSum was evaluated on Portuguese legal documents.",
    ], "each claim must drive its own retrieval"
    assert result.checked == 2


def test_the_judge_never_sees_the_original_text(sample_pdf, fake_llm):
    """Only the claims and their freshly retrieved passages reach the judge."""
    ingest(sample_pdf)
    verify(ANSWER)

    judge_call = next(
        call for call in fake_llm.calls if "decide whether each claim" in (call["system"] or "")
    )
    assert ANSWER not in judge_call["prompt"]
    assert "Evidence retrieved for it" in judge_call["prompt"]


def test_evidence_comes_from_the_corpus_not_the_claim(sample_pdf, fake_llm):
    ingest(sample_pdf)
    result = verify(ANSWER)

    supported = result.claims[0]
    assert supported["evidence"], "a claim about the corpus should retrieve passages"
    assert all(item["paper_id"] for item in supported["evidence"])
    assert all(item["text"] for item in supported["evidence"])


# --- verdicts -----------------------------------------------------------------


def test_an_unsupported_claim_is_flagged(sample_pdf, fake_llm):
    """The invented half of the answer must not pass."""
    ingest(sample_pdf)
    result = verify(ANSWER)

    verdicts = {claim["text"]: claim["verdict"] for claim in result.claims}
    assert verdicts["SparseSum reaches 44.1 ROUGE-L on arXiv."] == "supported"
    assert verdicts["SparseSum was evaluated on Portuguese legal documents."] == "unsupported"

    assert len(result.problems) == 1
    assert result.problems[0]["text"].startswith("SparseSum was evaluated on Portuguese")
    assert result.score == pytest.approx(0.5)
    assert result.status == "completed"


def test_score_weights_partial_support(sample_pdf):
    ingest(sample_pdf)
    set_llm(
        FakeGemini(
            {
                **AGENT_RESPONSES,
                "decide whether each claim is supported": {
                    "judgements": [
                        {"claim_index": 0, "verdict": "supported"},
                        {"claim_index": 1, "verdict": "partially_supported"},
                    ]
                },
            }
        )
    )

    result = verify(ANSWER)
    assert result.score == pytest.approx(0.75)
    assert result.counts == {"supported": 1, "partially_supported": 1}
    assert result.problems == []


def test_a_contradicted_claim_counts_as_a_problem(sample_pdf):
    ingest(sample_pdf)
    set_llm(
        FakeGemini(
            {
                **AGENT_RESPONSES,
                "decide whether each claim is supported": {
                    "judgements": [
                        {"claim_index": 0, "verdict": "contradicted", "explanation": "89.7, not 44.1."},
                        {"claim_index": 1, "verdict": "supported"},
                    ]
                },
            }
        )
    )

    result = verify(ANSWER)
    assert [claim["verdict"] for claim in result.problems] == ["contradicted"]
    assert result.score == pytest.approx(0.5)


def test_a_skipped_judgement_does_not_silently_pass(sample_pdf):
    """A claim the model forgot to judge must not be counted as supported."""
    ingest(sample_pdf)
    set_llm(
        FakeGemini(
            {
                **AGENT_RESPONSES,
                "decide whether each claim is supported": {
                    "judgements": [{"claim_index": 0, "verdict": "supported"}]
                },
            }
        )
    )

    result = verify(ANSWER)
    assert result.claims[1]["verdict"] == "unverifiable"
    assert result.score == pytest.approx(0.5)


def test_unknown_verdicts_fall_back_to_unsupported():
    """A verdict outside the vocabulary must fail closed, not open."""
    set_llm(
        FakeGemini(
            {
                "decide whether each claim is supported": {
                    "judgements": [{"claim_index": 0, "verdict": "looks_fine_to_me"}]
                }
            }
        )
    )

    judged = judge_claims([{"text": "A claim about SparseSum.", "evidence": []}])
    assert judged[0]["verdict"] == "unsupported"


def test_non_checkable_claims_are_not_sent_to_the_judge(sample_pdf):
    ingest(sample_pdf)
    fake = FakeGemini(
        {
            **AGENT_RESPONSES,
            "break a piece of generated text": {
                "claims": [
                    {"text": "SparseSum reaches 44.1 ROUGE-L.", "checkable": True},
                    {"text": "Further work in this area is needed.", "checkable": False},
                ]
            },
            "decide whether each claim is supported": {
                "judgements": [{"claim_index": 0, "verdict": "supported"}]
            },
        }
    )
    set_llm(fake)

    result = verify(ANSWER)

    judge_call = next(c for c in fake.calls if "decide whether each claim" in (c["system"] or ""))
    assert "Further work in this area" not in judge_call["prompt"]

    assert result.checked == 1
    assert result.score == pytest.approx(1.0)
    hedge = next(c for c in result.claims if c["text"].startswith("Further work"))
    assert hedge["verdict"] == "unverifiable"


# --- cost ---------------------------------------------------------------------


def test_cost_is_two_calls_regardless_of_claim_count(sample_pdf):
    """Judging is batched: length must not multiply the bill."""
    ingest(sample_pdf)
    fake = FakeGemini(
        {
            **AGENT_RESPONSES,
            "break a piece of generated text": {
                "claims": [{"text": f"Claim number {i} about SparseSum."} for i in range(12)]
            },
            "decide whether each claim is supported": {
                "judgements": [{"claim_index": i, "verdict": "supported"} for i in range(12)]
            },
        }
    )
    set_llm(fake)

    result = verify(ANSWER)

    assert result.checked == 12
    assert result.llm_calls == 2
    assert len(fake.calls) == 2


def test_claims_are_capped(sample_pdf):
    ingest(sample_pdf)
    set_llm(
        FakeGemini(
            {
                **AGENT_RESPONSES,
                "break a piece of generated text": {
                    "claims": [{"text": f"Claim number {i} about SparseSum."} for i in range(60)]
                },
                "decide whether each claim is supported": {"judgements": []},
            }
        )
    )

    result = verify(ANSWER)
    assert len(result.claims) == verification.MAX_CLAIMS


# --- failures -----------------------------------------------------------------


def test_missing_api_key_is_reported_not_raised(sample_pdf):
    ingest(sample_pdf)
    result = verify(ANSWER)

    assert result.status == "failed"
    assert result.claims == []
    assert any("GOOGLE_API_KEY" in error for error in result.errors)


def test_model_failure_during_judgement_is_reported(sample_pdf, monkeypatch):
    ingest(sample_pdf)
    set_llm(FakeGemini(AGENT_RESPONSES))

    def boom(claims):
        raise LLMError("simulated model failure")

    monkeypatch.setattr(verification, "judge_claims", boom)
    result = verify(ANSWER)

    assert result.status == "failed"
    assert any("simulated model failure" in error for error in result.errors)


def test_retrieval_failure_leaves_the_claim_judgeable(sample_pdf, fake_llm, monkeypatch):
    """A broken index must not abort the check — it becomes a claim with no evidence."""
    ingest(sample_pdf)

    from app.agents import retrieval

    monkeypatch.setattr(
        retrieval, "search", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("index down"))
    )

    result = verify(ANSWER)
    assert result.checked == 2
    assert all(claim["evidence"] == [] for claim in result.claims)


def test_short_text_is_rejected_before_spending_a_call(fake_llm):
    result = verify("Yes.")
    assert result.claims == []
    assert fake_llm.calls == []
    assert any("too short" in error for error in result.errors)


def test_blank_claims_are_dropped(sample_pdf):
    ingest(sample_pdf)
    set_llm(
        FakeGemini(
            {
                **AGENT_RESPONSES,
                "break a piece of generated text": {
                    "claims": [{"text": "  "}, {"text": "short"}, "A real claim about SparseSum."]
                },
                "decide whether each claim is supported": {
                    "judgements": [{"claim_index": 0, "verdict": "supported"}]
                },
            }
        )
    )

    claims = extract_claims("irrelevant")
    assert [claim["text"] for claim in claims] == ["A real claim about SparseSum."]


# --- pulling text out of a stored run -----------------------------------------


def test_extract_text_reads_a_qa_result():
    text = extract_text(
        {"answer": {"answer": "SparseSum is linear-time.", "caveats": ["English only."]}}
    )
    assert "SparseSum is linear-time." in text
    assert "English only." in text


def test_extract_text_reads_a_summary_and_a_gap_result():
    text = extract_text(
        {
            "summary": {
                "tldr": "Sparsity works.",
                "key_findings": ["44.1 ROUGE-L"],
                "themes": [{"name": "Efficiency", "description": "Both cut cost."}],
            },
            "gaps": {
                "landscape_summary": "Faithfulness is unmeasured.",
                "gaps": [{"title": "No multilingual evaluation", "description": "English only."}],
            },
        }
    )
    for expected in (
        "Sparsity works.",
        "44.1 ROUGE-L",
        "Both cut cost.",
        "Faithfulness is unmeasured.",
        "No multilingual evaluation",
    ):
        assert expected in text


def test_extract_text_of_an_empty_result_is_empty():
    assert extract_text({}) == ""
    assert extract_text({"graph": {"nodes": []}}) == ""


# --- scoring table ------------------------------------------------------------


def test_only_supported_claims_score_full_marks():
    assert VERDICT_SCORES["supported"] == 1.0
    assert VERDICT_SCORES["partially_supported"] == 0.5
    for verdict in ("unsupported", "contradicted", "unverifiable"):
        assert VERDICT_SCORES[verdict] == 0.0


# --- API ----------------------------------------------------------------------


def test_verify_endpoint_checks_pasted_text(client, sample_pdf, fake_llm):
    ingest(sample_pdf)

    response = client.post("/api/verify", json={"text": ANSWER, "source": "text"})
    assert response.status_code == 200

    body = response.json()
    assert body["checked"] == 2
    assert body["score"] == pytest.approx(0.5)
    assert body["llm_calls"] == 2
    assert len(body["problems"]) == 1
    assert body["id"], "the verification should be saved"


def test_verify_endpoint_reads_a_stored_agent_run(client, sample_pdf, fake_llm):
    ingest(sample_pdf)

    ask = client.post("/api/agents/ask", json={"question": "What ROUGE-L does SparseSum reach?"})
    assert ask.status_code == 200
    run_id = ask.json()["run_id"]
    assert run_id

    response = client.post("/api/verify", json={"agent_run_id": run_id})
    assert response.status_code == 200

    body = response.json()
    # Source, papers and subject are inherited from the run.
    assert body["source"] == "qa"
    assert body["subject"] == "What ROUGE-L does SparseSum reach?"
    assert body["checked"] == 2


def test_verify_endpoint_rejects_an_unknown_run(client, fake_llm):
    response = client.post("/api/verify", json={"agent_run_id": "does-not-exist"})
    assert response.status_code == 404


def test_verify_endpoint_rejects_empty_text(client, fake_llm):
    response = client.post("/api/verify", json={"text": "no"})
    assert response.status_code == 400


def test_verifications_are_listed_and_retrievable(client, sample_pdf, fake_llm):
    ingest(sample_pdf)
    created = client.post("/api/verify", json={"text": ANSWER, "subject": "Ad-hoc check"}).json()

    listing = client.get("/api/verify").json()
    assert [row["id"] for row in listing] == [created["id"]]
    assert listing[0]["subject"] == "Ad-hoc check"

    detail = client.get(f"/api/verify/{created['id']}").json()
    assert detail["text"] == ANSWER
    assert len(detail["claims"]) == 2
    assert detail["claims"][0]["evidence"]

    assert client.delete(f"/api/verify/{created['id']}").status_code == 204
    assert client.get(f"/api/verify/{created['id']}").status_code == 404


def test_verify_endpoint_can_skip_saving(client, sample_pdf, fake_llm):
    ingest(sample_pdf)
    body = client.post("/api/verify", json={"text": ANSWER, "save": False}).json()

    assert body["id"] is None
    assert client.get("/api/verify").json() == []


def test_verify_endpoint_reports_a_model_failure_without_a_500(client, sample_pdf):
    ingest(sample_pdf)
    set_llm(FakeGemini(fail=True))

    response = client.post("/api/verify", json={"text": ANSWER})
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "failed"
    assert body["claims"] == []
    assert any("simulated model failure" in error for error in body["errors"])
