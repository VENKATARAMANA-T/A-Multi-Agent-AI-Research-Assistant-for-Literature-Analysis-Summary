"""Figure Agent — reads charts, diagrams and tables with the vision model.

A results chart carries the paper's actual findings, and text extraction throws
all of it away: a line plot becomes nothing, and a table becomes a run of
numbers with no column headings. This agent looks at the rendered image and
returns what the figure shows, in a structure the rest of the system can index,
search and cite.

Cost matters here. Every figure is one vision request, so a ten-figure paper
costs ten requests against a free tier that allows twenty a day. Analysis is
therefore never automatic: extraction happens during ingestion for free, and
the vision pass is an explicit, costed action. Tables whose contents were
already recovered as Markdown are skipped entirely — there is nothing a vision
call would add.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.agents.parallel import map_parallel
from app.agents.state import trace_event
from app.services.llm import LLMError, get_llm, summarise_provider_error

logger = logging.getLogger(__name__)

FIGURE_SYSTEM = (
    "You are the Figure Agent of ResearchCompass. You read figures from academic "
    "papers — line plots, bar charts, ROC curves, box plots, heatmaps, confusion "
    "matrices, architecture diagrams, flowcharts and tables — and report exactly "
    "what they show.\n"
    "Report only what is visible in the image. Read values off the axes and "
    "labels; never estimate a number that is not shown, and never infer a result "
    "the figure does not display. If part of the image is unreadable, say so."
)

FIGURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chart_type": {
            "type": "string",
            "enum": [
                "line", "bar", "scatter", "box", "roc", "heatmap",
                "confusion_matrix", "pie", "histogram", "architecture",
                "flowchart", "table", "photograph", "screenshot", "equation",
                "waveform", "spectrogram", "other",
            ],
        },
        "description": {"type": "string"},
        "axes": {
            "type": "object",
            "properties": {
                "x_label": {"type": "string"},
                "x_units": {"type": "string"},
                "x_range": {"type": "string"},
                "y_label": {"type": "string"},
                "y_units": {"type": "string"},
                "y_range": {"type": "string"},
            },
        },
        "series": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "trend": {"type": "string"},
                    "notable_values": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name"],
            },
        },
        "findings": {"type": "array", "items": {"type": "string"}},
        "takeaway": {"type": "string"},
        "datasets": {"type": "array", "items": {"type": "string"}},
        "methods": {"type": "array", "items": {"type": "string"}},
        "metrics": {"type": "array", "items": {"type": "string"}},
        "readable": {"type": "boolean"},
    },
    "required": ["chart_type", "description", "takeaway"],
}


def build_prompt(label: str, caption: str, paper_title: str | None) -> str:
    context = f"Paper: {paper_title}\n" if paper_title else ""
    return f"""{context}This image is {label} from that paper.

Its printed caption reads:
"{caption}"

Describe what the image actually shows.

- `chart_type`: what kind of figure this is.
- `description`: 2-5 sentences a researcher could read instead of seeing the
  figure. Name the axes, the series, and what the data does.
- `axes`: the axis labels, units and ranges exactly as printed. Omit for a
  diagram or photograph that has no axes.
- `series`: one entry per plotted line, bar group or matrix row, with its trend
  and any values readable from the image (for example "AUC = 0.95", "peaks at
  epoch 30"). For an architecture diagram, list the components instead.
- `findings`: specific, checkable statements supported by the image.
- `takeaway`: the single thing this figure is in the paper to demonstrate.
- `datasets`, `methods`, `metrics`: names that appear in the image itself.
- `readable`: false if the image is too small, cropped or blurred to interpret;
  then say why in `description` rather than guessing.

Do not repeat the caption back. Report what the caption does not say.

Return JSON only."""


def analyse_figure(
    image_png: bytes,
    label: str,
    caption: str,
    paper_title: str | None = None,
) -> dict[str, Any]:
    """One vision call for one figure. Raises LLMError on failure."""
    from app.services.llm import parse_json_object

    text = get_llm().generate_from_image(
        prompt=build_prompt(label, caption, paper_title),
        image_bytes=image_png,
        mime_type="image/png",
        temperature=0.1,
    )
    if not text:
        raise LLMError("The vision model returned an empty response.")
    return parse_json_object(text)


def normalise_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    """Coerce the model's output into the columns we store."""

    def strings(key: str) -> list[str]:
        return [str(v).strip() for v in (payload.get(key) or []) if str(v).strip()]

    series = []
    for item in payload.get("series") or []:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        series.append({k: v for k, v in item.items() if v})

    axes = payload.get("axes") if isinstance(payload.get("axes"), dict) else {}

    return {
        "chart_type": str(payload.get("chart_type") or "other"),
        "description": str(payload.get("description") or "").strip() or None,
        "takeaway": str(payload.get("takeaway") or "").strip() or None,
        "axes": {k: str(v) for k, v in (axes or {}).items() if v},
        "series": series,
        "findings": strings("findings"),
        "entities": {
            "datasets": strings("datasets"),
            "methods": strings("methods"),
            "metrics": strings("metrics"),
        },
        "readable": bool(payload.get("readable", True)),
    }


def analyse_many(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any] | None], dict[str, Any]]:
    """Analyse several figures concurrently.

    Each item needs `image_png`, `label`, `caption` and optionally
    `paper_title`. Returns (results aligned with input, trace delta).
    """
    started = time.perf_counter()
    if not items:
        return [], {"trace": [trace_event("figures", "skipped", started, reason="nothing to analyse")]}

    errors: list[str] = []

    def one(item: dict[str, Any]) -> dict[str, Any] | None:
        try:
            payload = analyse_figure(
                image_png=item["image_png"],
                label=item.get("label", "Figure"),
                caption=item.get("caption", ""),
                paper_title=item.get("paper_title"),
            )
            return normalise_analysis(payload)
        except LLMError as exc:
            message = summarise_provider_error(str(exc))
            logger.warning("Figure analysis failed for %s: %s", item.get("label"), message)
            errors.append(f"{item.get('label', 'figure')}: {message}")
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Figure analysis crashed for %s", item.get("label"))
            errors.append(f"{item.get('label', 'figure')}: {exc}")
            return None

    results = map_parallel(one, items, label="figure")
    succeeded = sum(1 for r in results if r)

    return results, {
        "trace": [
            trace_event(
                "figures",
                "ok" if succeeded else "error",
                started,
                analysed=succeeded,
                failed=len(items) - succeeded,
            )
        ],
        "errors": errors,
        "llm_calls": succeeded,
    }
