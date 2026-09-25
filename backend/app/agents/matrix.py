"""Custom extraction columns — the "AI matrix".

The built-in extraction agent pulls a fixed set of entities. A researcher
comparing papers almost always wants something else as well: sample size,
ethics approval, hardware used, whether code was released. This agent lets the
columns be defined at query time and fills them across every selected paper,
producing the comparison table the literature review actually needs.

One call per paper, run concurrently. Every answer must be grounded, and a
column the paper does not address comes back explicitly "not reported" rather
than invented — a fabricated sample size is worse than a blank cell.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.agents.parallel import map_parallel
from app.agents.state import condense_paper_text, trace_event
from app.services.llm import LLMError, get_llm, summarise_provider_error

logger = logging.getLogger(__name__)

NOT_REPORTED = "not reported"

MATRIX_SYSTEM = (
    "You are the Comparison Agent of ResearchCompass. You fill in a table of "
    "user-defined columns for one research paper, reading only that paper.\n"
    "Every cell must be supported by the paper's own text. If the paper does "
    f'not address a column, answer exactly "{NOT_REPORTED}" — never guess, '
    "never infer from convention, and never carry over a value from a typical "
    "study in the field. A blank cell is useful; an invented one is harmful."
)

# Papers are trimmed to this many characters for a matrix pass. Columns are
# usually answered from the method and results sections, so the full text is
# rarely needed and costs latency.
PAPER_BUDGET = 40_000


def build_schema(columns: list[dict[str, Any]]) -> dict[str, Any]:
    """A JSON schema with one property per requested column."""
    properties: dict[str, Any] = {}
    for column in columns:
        key = column["key"]
        kind = (column.get("type") or "text").lower()
        if kind == "number":
            spec: dict[str, Any] = {"type": "string"}  # kept as text: "n = 120 (60 per arm)"
        elif kind == "boolean":
            spec = {"type": "string", "enum": ["yes", "no", NOT_REPORTED]}
        elif kind == "list":
            spec = {"type": "array", "items": {"type": "string"}}
        else:
            spec = {"type": "string"}
        properties[key] = spec
        properties[f"{key}__evidence"] = {"type": "string"}

    return {"type": "object", "properties": properties, "required": [c["key"] for c in columns]}


def build_prompt(title: str, text: str, columns: list[dict[str, Any]]) -> str:
    lines = []
    for column in columns:
        description = column.get("description") or ""
        kind = (column.get("type") or "text").lower()
        hint = {
            "number": "Give the figure with its units, exactly as the paper states it.",
            "boolean": 'Answer "yes", "no", or "not reported".',
            "list": "Return an array of short strings.",
            "text": "Answer in one short phrase, not a sentence.",
        }[kind if kind in ("number", "boolean", "list") else "text"]
        lines.append(f'- `{column["key"]}` — {column["name"]}. {description} {hint}')
        lines.append(f'- `{column["key"]}__evidence` — the phrase from the paper that supports it.')

    columns_block = "\n".join(lines)

    return f"""Fill in these columns for the paper below.

## Paper
{title}

## Content
{text}

## Columns
{columns_block}

## Rules
- Use only this paper. Do not use background knowledge about the field.
- If the paper does not state something, answer "{NOT_REPORTED}" and leave its
  evidence empty. Do not estimate, and do not infer from what similar studies
  usually do.
- Quote the supporting phrase verbatim in the `__evidence` field so each cell
  can be checked.

Return JSON only."""


def normalise_key(name: str) -> str:
    """A safe JSON key derived from a user-supplied column name."""
    import re

    key = re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")
    return key or "column"


def prepare_columns(columns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign unique keys and drop unusable definitions."""
    prepared: list[dict[str, Any]] = []
    used: set[str] = set()
    for column in columns:
        name = str(column.get("name") or "").strip()
        if not name:
            continue
        key = normalise_key(name)
        candidate, suffix = key, 2
        while candidate in used:
            candidate, suffix = f"{key}_{suffix}", suffix + 1
        used.add(candidate)
        prepared.append(
            {
                "key": candidate,
                "name": name,
                "description": str(column.get("description") or "").strip(),
                "type": (column.get("type") or "text").lower(),
            }
        )
    return prepared


def fill_row(document: dict[str, Any], columns: list[dict[str, Any]]) -> dict[str, Any] | None:
    """One paper, one LLM call."""
    title = document.get("title") or document.get("id") or "Untitled"
    text = condense_paper_text(document.get("text") or "", document.get("sections"), PAPER_BUDGET)

    payload = get_llm().generate_json(
        prompt=build_prompt(title, text, columns),
        system_instruction=MATRIX_SYSTEM,
        schema=build_schema(columns),
        temperature=0.0,
    )

    cells: dict[str, Any] = {}
    for column in columns:
        key = column["key"]
        value = payload.get(key)
        if isinstance(value, list):
            value = [str(v).strip() for v in value if str(v).strip()]
        elif value is not None:
            value = str(value).strip()
        cells[key] = {
            "value": value if value not in ("", None, []) else NOT_REPORTED,
            "evidence": str(payload.get(f"{key}__evidence") or "").strip() or None,
            "reported": bool(value) and str(value).lower() != NOT_REPORTED,
        }

    return {"paper_id": document.get("id"), "paper_title": title, "cells": cells}


def run_matrix(
    documents: list[dict[str, Any]],
    columns: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fill the matrix for every document, concurrently."""
    started = time.perf_counter()
    prepared = prepare_columns(columns)

    if not prepared:
        return {"columns": [], "rows": [], "errors": ["No usable columns were supplied."],
                "llm_calls": 0, "trace": []}
    if not documents:
        return {"columns": prepared, "rows": [], "errors": ["No papers to compare."],
                "llm_calls": 0, "trace": []}

    errors: list[str] = []

    def one(document: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return fill_row(document, prepared)
        except LLMError as exc:
            message = summarise_provider_error(str(exc))
            errors.append(f"{document.get('title') or document.get('id')}: {message}")
            return None
        except Exception as exc:  # pragma: no cover
            logger.exception("Matrix row failed")
            errors.append(f"{document.get('title') or document.get('id')}: {exc}")
            return None

    results = map_parallel(one, documents, label="matrix")
    rows = [row for row in results if row]

    return {
        "columns": prepared,
        "rows": rows,
        "errors": errors,
        "llm_calls": len(rows),
        "trace": [
            trace_event(
                "matrix",
                "ok" if rows else "error",
                started,
                papers=len(rows),
                columns=len(prepared),
            )
        ],
    }


def to_csv(columns: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    """Render the matrix as CSV for a spreadsheet."""
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["Paper"] + [column["name"] for column in columns])

    for row in rows:
        line = [row.get("paper_title") or row.get("paper_id")]
        for column in columns:
            cell = (row.get("cells") or {}).get(column["key"]) or {}
            value = cell.get("value")
            if isinstance(value, list):
                value = "; ".join(value)
            line.append(value if value is not None else "")
        writer.writerow(line)

    return buffer.getvalue()
