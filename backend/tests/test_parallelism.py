"""Parallel agent execution and the state reducers that make it safe.

Two things have to hold for the concurrent review pipeline to be correct:
the work must genuinely overlap, and no branch's bookkeeping may be lost when
several of them write `trace`/`errors`/`llm_calls` at the same time.
"""

from __future__ import annotations

import threading
import time

from app.agents.parallel import map_parallel, run_parallel
from app.agents.workflow import REVIEW_BRANCHES, run_workflow
from app.database import session_scope
from app.services.ingestion import process_paper, store_upload


def ingest(path) -> str:
    from tests.conftest import seed_user_id

    with session_scope() as session:
        paper, _ = store_upload(session, path.name, path.read_bytes(), seed_user_id())
        process_paper(session, paper.id)
        return paper.id


# --- map_parallel ------------------------------------------------------------


def test_results_keep_input_order_regardless_of_completion_order():
    """Fast items finish first, but output must still match input order."""

    def work(n: int) -> int:
        time.sleep(0.05 if n % 2 == 0 else 0.01)
        return n * 10

    assert map_parallel(work, [0, 1, 2, 3, 4], max_workers=5) == [0, 10, 20, 30, 40]


def test_work_actually_overlaps():
    """Five 100 ms sleeps must take far less than the 500 ms they would serially."""
    started = time.perf_counter()
    map_parallel(lambda _: time.sleep(0.1), list(range(5)), max_workers=5)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.35, f"expected overlap, took {elapsed:.2f}s"


def test_concurrency_is_capped():
    """max_workers must be honoured — the provider's rate limit depends on it."""
    peak = 0
    current = 0
    lock = threading.Lock()

    def work(_: int) -> None:
        nonlocal peak, current
        with lock:
            current += 1
            peak = max(peak, current)
        time.sleep(0.05)
        with lock:
            current -= 1

    map_parallel(work, list(range(10)), max_workers=2)
    assert peak <= 2


def test_a_failing_item_does_not_abort_the_others():
    def work(n: int) -> int:
        if n == 1:
            raise ValueError("boom")
        return n

    assert map_parallel(work, [0, 1, 2]) == [0, None, 2]


def test_empty_input_returns_empty():
    assert map_parallel(lambda x: x, []) == []


def test_run_parallel_returns_named_results():
    results = run_parallel(
        [("a", lambda: 1), ("b", lambda: 2), ("c", lambda: 1 / 0)], max_workers=3
    )

    assert results == {"a": 1, "b": 2, "c": None}


# --- reducers under concurrency ----------------------------------------------


def test_review_runs_every_branch_and_keeps_all_bookkeeping(sample_pdf, second_pdf, fake_llm):
    """The four review branches run concurrently; none may lose its trace."""
    ingest(sample_pdf)
    ingest(second_pdf)

    result = run_workflow("review")

    nodes = [event["node"] for event in result["trace"]]
    assert "router" in nodes and "loader" in nodes
    for branch in ("multi_summarize", "extract", "gap", "graph"):
        assert branch in nodes, f"{branch} missing from the merged trace"

    # All four outputs survived the concurrent merge.
    assert result["summary"] and result["extraction"] and result["gaps"] and result["graph"]

    # 1 summary + 2 extractions + 1 gap + 2 graph calls, none double-counted.
    assert result["llm_calls"] == 6
    assert result["status"] == "completed"


def test_review_branches_are_declared_parallel():
    from app.agents.workflow import route_after_load

    state = {"intent": "review", "documents": [{"id": "x"}]}
    assert route_after_load(state) == REVIEW_BRANCHES
    assert len(REVIEW_BRANCHES) == 4


def test_single_intent_enters_exactly_one_branch(sample_pdf, fake_llm):
    ingest(sample_pdf)
    result = run_workflow("gap")

    nodes = [event["node"] for event in result["trace"]]
    assert "gap" in nodes
    assert "extract" not in nodes
    assert "graph" not in nodes


def test_errors_from_concurrent_branches_are_all_retained(sample_pdf, second_pdf):
    """With no API key every branch fails; every failure must be reported."""
    ingest(sample_pdf)
    ingest(second_pdf)

    result = run_workflow("review")

    assert result["status"] == "partial"
    # One error per branch, with extraction and graph failing once per paper.
    assert len(result["errors"]) >= 4
    assert any("multi_summarize" in error for error in result["errors"])
    assert any("gap" in error for error in result["errors"])


def test_trace_is_not_duplicated_by_the_reducer(sample_pdf, fake_llm):
    """A node returning a delta must not re-emit entries already in the trace."""
    ingest(sample_pdf)
    result = run_workflow("qa", question="What is the method?")

    nodes = [event["node"] for event in result["trace"]]
    assert nodes.count("router") == 1
    assert nodes.count("retrieval") == 1
    assert nodes.count("graph_retrieval") == 1
    assert nodes.count("qa") == 1


def test_per_paper_extraction_is_counted_once_each(sample_pdf, second_pdf, fake_llm):
    ingest(sample_pdf)
    ingest(second_pdf)

    result = run_workflow("extract")

    assert result["llm_calls"] == 2
    assert len(result["extraction"]["papers"]) == 2
