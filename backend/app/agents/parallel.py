"""Bounded parallel execution for per-paper agent calls.

`extract` and `build_graph` make one LLM call per paper. Run serially, a
five-paper corpus costs five round trips end to end; the calls are independent,
so nearly all of that is idle waiting.

The Gemini SDK call is blocking I/O, so threads are the right tool — they let us
parallelise without converting the whole LangGraph into async. Concurrency is
capped because the free tier limits requests *per minute* as well as per day,
and an unbounded fan-out trips that immediately.

Results are returned in input order regardless of completion order, so output is
deterministic and diffable even though execution is not.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Sequence, TypeVar

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def map_parallel(
    fn: Callable[[T], R],
    items: Sequence[T],
    max_workers: int | None = None,
    label: str = "task",
) -> list[R | None]:
    """Apply `fn` to each item concurrently, preserving input order.

    An item that raises yields None in its slot and is logged — one failing
    paper must not abort the others, matching the workflow's per-agent failure
    isolation.
    """
    if not items:
        return []

    workers = max(1, min(max_workers or settings.llm_max_concurrency, len(items)))
    if workers == 1 or len(items) == 1:
        return [_safe_call(fn, item, index, label) for index, item in enumerate(items)]

    logger.debug("Running %d %s(s) with %d workers", len(items), label, workers)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=label) as pool:
        futures = [pool.submit(_safe_call, fn, item, index, label) for index, item in enumerate(items)]
        return [future.result() for future in futures]


def _safe_call(fn: Callable[[T], R], item: T, index: int, label: str) -> R | None:
    try:
        return fn(item)
    except Exception:  # pragma: no cover - defensive; nodes handle their own errors
        logger.exception("%s #%d failed", label, index)
        return None


def run_parallel(
    tasks: Iterable[tuple[str, Callable[[], Any]]],
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Run named zero-argument callables concurrently; returns {name: result}.

    A task that raises stores None under its name rather than propagating.
    """
    task_list = list(tasks)
    if not task_list:
        return {}

    workers = max(1, min(max_workers or settings.llm_max_concurrency, len(task_list)))
    results: dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agent") as pool:
        futures = {name: pool.submit(_guard, name, fn) for name, fn in task_list}
        for name, future in futures.items():
            results[name] = future.result()
    return results


def _guard(name: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:  # pragma: no cover
        logger.exception("parallel task '%s' failed", name)
        return None
