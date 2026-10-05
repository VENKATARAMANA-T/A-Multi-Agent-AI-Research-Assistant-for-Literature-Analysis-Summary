"""Literature Review Agent — a full review, written section by section.

This is not the structural report in `services/report.py`, which assembles
tables from what the other agents already produced. This writes prose: ten
sections of narrative review on a topic you name, every claim carrying an `[S#]`
marker back to the paper it came from.

Two decisions shape the implementation.

**Three calls, not ten and not one.** One call cannot hold a ten-section review
inside the output limit — it truncates somewhere in the middle, usually the part
you wanted. Ten calls would be thorough and would spend half a free-tier day on
one document. The sections are written in three groups that genuinely need each
other's context: what the field is and how it got here, what the evidence
actually says, and what is missing. References are the tenth section and cost
nothing — they come from metadata already extracted.

**Citations are resolved, not trusted.** The model is given numbered papers and
asked to cite them, but a marker pointing at `[S12]` when only nine papers were
supplied is a fabrication. Every marker is checked against the real list and
dropped if it does not resolve, so a citation in the finished review always
points at a paper that exists.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.agents.state import condense_paper_text, trace_event
from app.services.llm import LLMError, get_llm, summarise_provider_error

logger = logging.getLogger(__name__)

# Enough for a full paragraph of context per paper without pushing the prompt
# past the point where the model starts skimming.
PAPER_BUDGET = 14_000
MAX_PAPERS = 40

CITATION_RE = re.compile(r"\[S(\d{1,3})\]")

SYSTEM = (
    "You are the Literature Review Agent of ResearchCompass. You write the "
    "literature review section of an academic paper.\n"
    "Write prose, not bullet summaries of each paper in turn: a review that "
    "walks through the corpus one paper at a time is an annotated bibliography, "
    "not a review. Group by idea, compare approaches, and say where the evidence "
    "disagrees.\n"
    "Cite with [S1], [S2] markers after the claim they support. Cite only the "
    "numbered papers given to you — never invent a number. An uncited claim "
    "reads as the author's own assertion, so cite anything that came from a paper.\n"
    "Where the corpus cannot answer something, say so plainly instead of "
    "filling the gap with general knowledge."
)

# The ten sections, written in three passes. Grouped by what needs what: you
# cannot compare methods without having described them, and you cannot argue a
# gap without having shown the evidence.
GROUPS: list[dict[str, Any]] = [
    {
        "key": "framing",
        "label": "Introduction, evolution and approaches",
        "sections": [
            ("introduction", "1. Introduction",
             "Frame the topic, why it matters, and what this review covers. "
             "State the scope honestly: how many papers, what years, what they have in common."),
            ("evolution", "2. Research Evolution",
             "How the field moved. Earlier approaches, what changed, what drove the change. "
             "Use the publication years to order the story where they are available."),
            ("approaches", "3. Existing Approaches",
             "The families of method present in this corpus, what distinguishes them, "
             "and which papers take which position."),
        ],
    },
    {
        "key": "evidence",
        "label": "Datasets, comparison and disagreements",
        "sections": [
            ("datasets", "4. Dataset Landscape",
             "The datasets used, their size and domain, which are shared across papers "
             "and which are used by only one. Note where a result rests on a single dataset."),
            ("comparison", "5. Method Comparison",
             "Compare the methods on what they actually report: metrics, numbers, conditions. "
             "Say where a comparison is not possible because the setups differ."),
            ("conflicts", "6. Conflicting Findings",
             "Where the papers disagree, and on what. If they do not conflict, say that "
             "explicitly and explain why — a corpus too small or too homogeneous to conflict "
             "is itself a finding."),
        ],
    },
    {
        "key": "forward",
        "label": "Gaps, open problems and directions",
        "sections": [
            ("gaps", "7. Research Gaps",
             "What this literature has not established. Be specific about the missing "
             "evidence, not generically 'more research is needed'."),
            ("open_problems", "8. Open Problems",
             "The hard problems the field has not solved, as distinct from gaps in coverage."),
            ("directions", "9. Proposed Research Directions",
             "Concrete studies that would close the gaps above. Each should name a design, "
             "a dataset and an outcome that would settle the question."),
        ],
    },
]

SECTION_ORDER = [key for group in GROUPS for key, _, _ in group["sections"]]
SECTION_TITLES = {key: title for group in GROUPS for key, title, _ in group["sections"]}


@dataclass
class ReviewResult:
    topic: str
    sections: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    markdown: str = ""
    llm_calls: int = 0
    errors: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.sections:
            return "failed"
        return "partial" if self.errors else "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "sections": self.sections,
            "citations": self.citations,
            "markdown": self.markdown,
            "llm_calls": self.llm_calls,
            "errors": self.errors,
            "trace": self.trace,
            "status": self.status,
        }


def numbered_context(documents: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Render each paper under its own [S#] marker, and return the key to them.

    The same numbering is handed to the model and used to resolve the markers
    afterwards, so `[S3]` means one thing throughout.
    """
    blocks: list[str] = []
    citations: list[dict[str, Any]] = []

    for index, doc in enumerate(documents[:MAX_PAPERS], start=1):
        marker = f"S{index}"
        authors = doc.get("authors") or []
        meta = []
        if authors:
            meta.append(", ".join(authors[:6]) + (" et al." if len(authors) > 6 else ""))
        if doc.get("year"):
            meta.append(str(doc["year"]))
        if doc.get("venue"):
            meta.append(doc["venue"])

        body = condense_paper_text(doc.get("text") or "", doc.get("sections"), PAPER_BUDGET)
        blocks.append(
            f"[{marker}] {doc.get('title') or doc.get('id')}\n"
            f"{' · '.join(meta) or 'No metadata'}\n\n{body}"
        )
        citations.append(
            {
                "marker": marker,
                "paper_id": doc.get("id"),
                "title": doc.get("title") or doc.get("id"),
                "authors": authors,
                "year": doc.get("year"),
                "venue": doc.get("venue"),
                "doi": doc.get("doi"),
            }
        )

    return "\n\n========\n\n".join(blocks), citations


def group_prompt(group: dict[str, Any], topic: str, context: str, focus: str | None) -> str:
    wanted = "\n".join(
        f"- `{key}` — **{title}**: {brief}" for key, title, brief in group["sections"]
    )
    focus_line = f"\n## Particular focus\n{focus}\n" if focus else ""

    return f"""Write these sections of a literature review on: **{topic}**
{focus_line}
## Sections to write
{wanted}

## The papers
{context}

## Rules
- Prose, in paragraphs. No bullet lists except where genuinely enumerating.
- Cite with [S1], [S2] after the claim. Only the markers above exist.
- 150-350 words per section. Say less rather than padding.
- Do not repeat a section's heading inside its own text.
- Write about the corpus you were given, not the field in general.

Return JSON with one key per section id above, each a string of Markdown prose."""


def schema_for(group: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {key: {"type": "string"} for key, _, _ in group["sections"]},
        "required": [key for key, _, _ in group["sections"]],
    }


def resolve_citations(text: str, citations: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Drop markers that point at a paper which was never supplied.

    A model asked to cite will occasionally produce `[S12]` from a corpus of
    nine. Leaving it in the finished review would be worse than having no
    citation at all: it looks authoritative and resolves to nothing.
    """
    valid = {item["marker"] for item in citations}
    used: list[str] = []

    def replace(match: re.Match) -> str:
        marker = f"S{int(match.group(1))}"
        if marker not in valid:
            logger.info("Dropped a citation to %s, which is not in this corpus", marker)
            return ""
        if marker not in used:
            used.append(marker)
        return f"[{marker}]"

    cleaned = CITATION_RE.sub(replace, text)
    # Tidy the spacing a dropped marker leaves behind.
    cleaned = re.sub(r" +([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip(), used


def format_reference(citation: dict[str, Any]) -> str:
    """One reference line. Built from stored metadata, so it costs nothing."""
    authors = citation.get("authors") or []
    if len(authors) > 3:
        who = f"{authors[0]} et al."
    elif authors:
        who = ", ".join(authors)
    else:
        who = "Unknown authors"

    parts = [f"**[{citation['marker']}]** {who}"]
    if citation.get("year"):
        parts.append(f"({citation['year']})")
    parts.append(f"*{citation['title']}*")
    if citation.get("venue"):
        parts.append(citation["venue"])
    if citation.get("doi"):
        parts.append(f"doi:{citation['doi']}")
    return ". ".join(parts).replace("..", ".")


def build_markdown(topic: str, sections: list[dict[str, Any]], citations: list[dict[str, Any]]) -> str:
    lines = [f"# Literature Review: {topic}", ""]
    for section in sections:
        lines += [f"## {section['title']}", "", section["text"], ""]

    lines += ["## 10. References", ""]
    lines += [format_reference(citation) for citation in citations]
    lines.append("")
    return "\n".join(lines)


def generate(
    topic: str,
    documents: list[dict[str, Any]],
    focus: str | None = None,
) -> ReviewResult:
    """Write the review. Three model calls plus a free reference list."""
    started = time.perf_counter()
    result = ReviewResult(topic=topic)

    if not documents:
        result.errors.append("literature_review: no indexed papers to review.")
        result.trace.append(trace_event("literature_review", "skipped", started, reason="no papers"))
        return result

    context, citations = numbered_context(documents)
    result.citations = citations

    produced: dict[str, str] = {}
    for group in GROUPS:
        group_started = time.perf_counter()
        try:
            payload = get_llm().generate_json(
                prompt=group_prompt(group, topic, context, focus),
                system_instruction=SYSTEM,
                schema=schema_for(group),
                temperature=0.35,
            )
            result.llm_calls += 1
        except LLMError as exc:
            message = summarise_provider_error(str(exc))
            # One failed group must not lose the two that worked: the review is
            # returned with a note where the missing sections would be.
            result.errors.append(f"literature_review[{group['key']}]: {message}")
            result.trace.append(
                trace_event("literature_review", "error", group_started, group=group["key"], error=message)
            )
            continue

        for key, _title, _brief in group["sections"]:
            body = str(payload.get(key) or "").strip()
            if body:
                produced[key] = body

        result.trace.append(
            trace_event(
                "literature_review", "ok", group_started,
                group=group["key"], sections=len([k for k, _, _ in group["sections"] if k in produced]),
            )
        )

    used_markers: list[str] = []
    for key in SECTION_ORDER:
        body = produced.get(key)
        if not body:
            continue
        cleaned, used = resolve_citations(body, citations)
        for marker in used:
            if marker not in used_markers:
                used_markers.append(marker)
        result.sections.append({"key": key, "title": SECTION_TITLES[key], "text": cleaned})

    # Only papers the review actually cites belong in its reference list.
    cited = [c for c in citations if c["marker"] in used_markers] or citations
    result.citations = citations
    result.markdown = build_markdown(topic, result.sections, cited)

    result.trace.append(
        trace_event(
            "literature_review", result.status, started,
            sections=len(result.sections), cited=len(cited), llm_calls=result.llm_calls,
        )
    )
    return result
