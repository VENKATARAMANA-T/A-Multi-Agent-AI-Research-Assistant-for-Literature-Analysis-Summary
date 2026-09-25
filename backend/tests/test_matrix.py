"""The AI matrix — user-defined comparison columns."""

from __future__ import annotations

import json

import pytest

from app.agents.matrix import NOT_REPORTED, build_schema, normalise_key, prepare_columns, to_csv
from app.services.llm import set_llm

COLUMNS = [
    {"name": "Sample size", "description": "Number of participants.", "type": "number"},
    {"name": "Code released?", "description": "Did the authors publish code?", "type": "boolean"},
    {"name": "Datasets", "description": "Corpora used.", "type": "list"},
]


class MatrixClient:
    """Returns a filled row, and records the prompts it was given."""

    model = "fake"
    available = True

    def __init__(self, payload=None, fail=False):
        self.calls = []
        self.fail = fail
        self.payload = payload or {
            "sample_size": "120 participants",
            "sample_size__evidence": "We recruited 120 participants.",
            "code_released": "yes",
            "code_released__evidence": "Code is available at github.com/example.",
            "datasets": ["UA-Speech", "TORGO"],
            "datasets__evidence": "evaluated on UA-Speech and TORGO",
        }

    def generate_json(self, prompt, system_instruction=None, schema=None, **kwargs):
        from app.services.llm import LLMError

        self.calls.append({"prompt": prompt, "system": system_instruction, "schema": schema})
        if self.fail:
            raise LLMError("simulated failure")
        return self.payload

    def generate(self, *args, **kwargs):
        raise NotImplementedError


# --- column preparation ------------------------------------------------------


def test_keys_are_derived_from_names():
    assert normalise_key("Sample size") == "sample_size"
    assert normalise_key("Code released?") == "code_released"
    assert normalise_key("  ") == "column"


def test_duplicate_names_get_distinct_keys():
    prepared = prepare_columns([{"name": "Size"}, {"name": "Size"}, {"name": "size"}])
    keys = [column["key"] for column in prepared]
    assert len(set(keys)) == 3


def test_nameless_columns_are_dropped():
    assert prepare_columns([{"name": ""}, {"description": "orphan"}]) == []


def test_schema_has_a_field_and_an_evidence_field_per_column():
    prepared = prepare_columns(COLUMNS)
    schema = build_schema(prepared)

    for column in prepared:
        assert column["key"] in schema["properties"]
        assert f"{column['key']}__evidence" in schema["properties"]
    assert schema["required"] == [c["key"] for c in prepared]


def test_boolean_columns_offer_a_not_reported_option():
    """A yes/no column must be able to say the paper is silent."""
    schema = build_schema(prepare_columns([{"name": "Code released?", "type": "boolean"}]))
    assert NOT_REPORTED in schema["properties"]["code_released"]["enum"]


# --- CSV ---------------------------------------------------------------------


def test_csv_has_a_header_and_one_row_per_paper():
    columns = prepare_columns(COLUMNS)
    rows = [
        {
            "paper_title": "Paper A",
            "cells": {
                "sample_size": {"value": "120"},
                "code_released": {"value": "yes"},
                "datasets": {"value": ["UA-Speech", "TORGO"]},
            },
        }
    ]
    csv_text = to_csv(columns, rows)
    lines = csv_text.strip().splitlines()

    assert lines[0] == "Paper,Sample size,Code released?,Datasets"
    assert "Paper A" in lines[1]
    assert "UA-Speech; TORGO" in lines[1]


def test_csv_escapes_commas_in_values():
    columns = prepare_columns([{"name": "Notes"}])
    rows = [{"paper_title": "A, B", "cells": {"notes": {"value": "one, two"}}}]
    assert '"A, B"' in to_csv(columns, rows)


# --- HTTP --------------------------------------------------------------------


def upload(client, path):
    return client.post(
        "/api/papers/upload?wait=true",
        files=[("files", (path.name, path.read_bytes(), "application/pdf"))],
    ).json()["results"][0]["paper_id"]


def test_matrix_fills_one_row_per_paper(client, sample_pdf, second_pdf):
    upload(client, sample_pdf)
    upload(client, second_pdf)
    fake = MatrixClient()
    set_llm(fake)

    payload = client.post("/api/matrix", json={"columns": COLUMNS}).json()

    assert len(payload["rows"]) == 2
    assert payload["llm_calls"] == 2
    assert len(payload["columns"]) == 3

    cell = payload["rows"][0]["cells"]["sample_size"]
    assert cell["value"] == "120 participants"
    assert cell["evidence"]
    assert cell["reported"] is True


def test_every_cell_carries_its_evidence(client, sample_pdf):
    """A cell that cannot be checked is not much use in a literature review."""
    upload(client, sample_pdf)
    set_llm(MatrixClient())

    row = client.post("/api/matrix", json={"columns": COLUMNS}).json()["rows"][0]

    for key in ("sample_size", "code_released", "datasets"):
        assert "evidence" in row["cells"][key]


def test_a_silent_paper_is_marked_not_reported(client, sample_pdf):
    """An invented sample size is worse than a blank cell."""
    upload(client, sample_pdf)
    set_llm(MatrixClient(payload={"sample_size": NOT_REPORTED, "sample_size__evidence": ""}))

    row = client.post(
        "/api/matrix", json={"columns": [{"name": "Sample size", "type": "number"}]}
    ).json()["rows"][0]

    assert row["cells"]["sample_size"]["value"] == NOT_REPORTED
    assert row["cells"]["sample_size"]["reported"] is False


def test_the_prompt_forbids_guessing(client, sample_pdf):
    upload(client, sample_pdf)
    fake = MatrixClient()
    set_llm(fake)

    client.post("/api/matrix", json={"columns": [{"name": "Sample size"}]})

    system = fake.calls[0]["system"].lower()
    assert "never guess" in system
    assert NOT_REPORTED in system


def test_columns_are_required(client, sample_pdf):
    upload(client, sample_pdf)
    assert client.post("/api/matrix", json={"columns": []}).status_code == 400


def test_too_many_columns_is_rejected(client, sample_pdf):
    upload(client, sample_pdf)
    many = [{"name": f"Column {i}"} for i in range(20)]
    assert client.post("/api/matrix", json={"columns": many}).status_code == 400


def test_matrix_requires_indexed_papers(client):
    assert client.post("/api/matrix", json={"columns": COLUMNS}).status_code == 409


def test_failures_are_reported_not_raised(client, sample_pdf):
    upload(client, sample_pdf)
    set_llm(MatrixClient(fail=True))

    payload = client.post("/api/matrix", json={"columns": COLUMNS}).json()

    assert payload["rows"] == []
    assert payload["errors"]
    assert payload["status"] == "partial"


def test_saved_runs_can_be_listed_fetched_and_exported(client, sample_pdf):
    upload(client, sample_pdf)
    set_llm(MatrixClient())

    created = client.post("/api/matrix", json={"columns": COLUMNS, "name": "My comparison"}).json()
    run_id = created["id"]

    listing = client.get("/api/matrix").json()
    assert any(run["id"] == run_id for run in listing)
    assert listing[0]["column_count"] == 3

    assert client.get(f"/api/matrix/{run_id}").json()["name"] == "My comparison"

    csv_response = client.get(f"/api/matrix/{run_id}/csv")
    assert csv_response.status_code == 200
    assert "Sample size" in csv_response.text
    assert "attachment" in csv_response.headers["content-disposition"]

    assert client.delete(f"/api/matrix/{run_id}").status_code == 204
    assert client.get(f"/api/matrix/{run_id}").status_code == 404
