"""Step 8 — literature review reports: Markdown, comparison tables, PDF export."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agents.prompts import REPORT_SYSTEM, report_prompt
from app.config import settings
from app.services.llm import LLMError, get_llm, summarise_provider_error

logger = logging.getLogger(__name__)

MAX_CELL = 90


def _escape(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value if value is not None else "")).strip()
    return text.replace("|", "\\|")


def _truncate(value: Any, limit: int = MAX_CELL) -> str:
    text = _escape(value)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _bullets(items: list[Any] | None, empty: str = "_None reported._") -> str:
    if not items:
        return empty
    return "\n".join(f"- {_escape(item)}" for item in items)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No data available._"
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_truncate(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, divider, *body])


def build_corpus_table(papers: list[dict]) -> str:
    rows = [
        [
            paper.get("title") or paper.get("id"),
            ", ".join((paper.get("authors") or [])[:3]) + (" et al." if len(paper.get("authors") or []) > 3 else ""),
            paper.get("year") or "—",
            paper.get("venue") or "—",
            paper.get("page_count") or "—",
        ]
        for paper in papers
    ]
    return markdown_table(["Paper", "Authors", "Year", "Venue", "Pages"], rows)


def build_comparison_table(extraction: dict | None) -> str:
    """One row per paper: task, method, dataset, headline metric."""
    if not extraction or not extraction.get("papers"):
        return "_Run the extraction agent to populate this table._"

    rows = []
    for record in extraction["papers"]:
        methods = ", ".join(item["name"] for item in record.get("methods", [])[:3]) or "—"
        datasets = ", ".join(item["name"] for item in record.get("datasets", [])[:3]) or "—"
        metrics = record.get("metrics", [])
        metric_text = "—"
        if metrics:
            metric = metrics[0]
            value = metric.get("value")
            metric_text = f"{metric['name']}: {value}" if value else metric["name"]
        tasks = ", ".join(record.get("tasks", [])[:2]) or "—"
        rows.append([record.get("paper_title"), tasks, methods, datasets, metric_text])

    return markdown_table(["Paper", "Task", "Method(s)", "Dataset(s)", "Headline metric"], rows)


def build_entity_table(extraction: dict | None, field: str, label: str) -> str:
    aggregate = (extraction or {}).get("aggregate") or {}
    items = aggregate.get(field) or []
    if not items:
        return f"_No {label.lower()} extracted._"
    rows = [[item["name"], len(item.get("papers", [])), ", ".join(item.get("papers", [])[:4])] for item in items[:25]]
    return markdown_table([label, "# Papers", "Appears in"], rows)


def build_gap_section(gaps: dict | None) -> str:
    if not gaps or not gaps.get("gaps"):
        return "_Run the research gap agent to populate this section._"

    parts = []
    summary = gaps.get("landscape_summary")
    if summary:
        parts.append(_escape(summary) + "\n")

    for i, gap in enumerate(gaps["gaps"], start=1):
        severity = str(gap.get("severity", "medium")).upper()
        category = str(gap.get("category", "general")).replace("_", " ").title()
        parts.append(f"### Gap {i}: {_escape(gap.get('title'))}")
        parts.append(f"**Severity:** {severity} · **Category:** {category}\n")
        parts.append(_escape(gap.get("description")) + "\n")
        if gap.get("evidence"):
            parts.append("**Evidence:**\n" + _bullets(gap["evidence"]) + "\n")
        if gap.get("proposed_direction"):
            parts.append(f"**Proposed direction:** {_escape(gap['proposed_direction'])}\n")
        if gap.get("related_papers"):
            parts.append("**Related papers:** " + ", ".join(_escape(p) for p in gap["related_papers"]) + "\n")

    if gaps.get("underexplored_intersections"):
        parts.append("### Under-explored Intersections\n" + _bullets(gaps["underexplored_intersections"]) + "\n")
    if gaps.get("open_questions"):
        parts.append("### Open Questions\n" + _bullets(gaps["open_questions"]) + "\n")

    return "\n".join(parts)


def build_synthesis_section(summary: dict | None) -> str:
    if not summary:
        return "_Run the summarization agent to populate this section._"

    parts = []
    if summary.get("overview"):
        parts.append(_escape(summary["overview"]) + "\n")
    for theme in summary.get("themes") or []:
        parts.append(f"### {_escape(theme.get('name'))}")
        parts.append(_escape(theme.get("description")))
        if theme.get("papers"):
            parts.append("\n_Papers:_ " + ", ".join(_escape(p) for p in theme["papers"]))
        parts.append("")
    if summary.get("agreements"):
        parts.append("### Points of Agreement\n" + _bullets(summary["agreements"]) + "\n")
    if summary.get("disagreements"):
        parts.append("### Points of Disagreement\n" + _bullets(summary["disagreements"]) + "\n")
    if summary.get("methodological_trends"):
        parts.append("### Methodological Trends\n" + _bullets(summary["methodological_trends"]) + "\n")
    return "\n".join(parts)


def build_references(papers: list[dict]) -> str:
    entries = []
    for paper in sorted(papers, key=lambda p: ((p.get("authors") or [""])[0], p.get("year") or 0)):
        authors = paper.get("authors") or []
        author_text = ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else "") if authors else "Unknown author"
        year = paper.get("year") or "n.d."
        title = paper.get("title") or paper.get("filename")
        venue = f" _{paper['venue']}_." if paper.get("venue") else ""
        doi = f" https://doi.org/{paper['doi']}" if paper.get("doi") else ""
        entries.append(f"1. {_escape(author_text)} ({year}). {_escape(title)}.{venue}{doi}")
    return "\n".join(entries) if entries else "_No papers in this report._"


def generate_narrative(context: str, focus: str | None) -> str | None:
    """Ask Gemini for the prose body. Returns None if the LLM is unavailable."""
    try:
        response = get_llm().generate(
            prompt=report_prompt(context, focus),
            system_instruction=REPORT_SYSTEM,
            temperature=0.3,
        )
        return response.text.strip()
    except LLMError as exc:
        logger.warning("Narrative generation skipped: %s", summarise_provider_error(str(exc)))
        return None


def build_notes_block(notes: list[str] | None) -> list[str]:
    """Explain, in the report itself, why sections may be missing."""
    if not notes:
        return []
    lines = [
        "> **Incomplete report.** Some agents did not finish, so the sections they",
        "> populate are empty below. Re-run the report once the issue is resolved.",
        ">",
    ]
    lines += [f"> - {_escape(note)}" for note in notes[:10]]
    lines.append("")
    return lines


def build_markdown_report(
    title: str,
    papers: list[dict],
    summary: dict | None = None,
    extraction: dict | None = None,
    gaps: dict | None = None,
    narrative: str | None = None,
    graph_stats: dict | None = None,
    notes: list[str] | None = None,
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    years = [p["year"] for p in papers if p.get("year")]
    span = f"{min(years)}–{max(years)}" if years else "unknown"

    sections: list[str] = [
        f"# {title}",
        "",
        f"_Generated by ResearchCompass on {generated} · {len(papers)} papers · publication years {span}_",
        "",
        *build_notes_block(notes),
        "## 1. Corpus Overview",
        "",
        build_corpus_table(papers),
        "",
    ]

    if narrative:
        sections += ["## 2. Literature Synthesis", "", narrative, ""]
        next_index = 3
    else:
        sections += ["## 2. Literature Synthesis", "", build_synthesis_section(summary), ""]
        next_index = 3

    sections += [
        f"## {next_index}. Comparative Analysis",
        "",
        build_comparison_table(extraction),
        "",
        f"### {next_index}.1 Datasets Across the Corpus",
        "",
        build_entity_table(extraction, "datasets", "Dataset"),
        "",
        f"### {next_index}.2 Methods Across the Corpus",
        "",
        build_entity_table(extraction, "methods", "Method"),
        "",
        f"### {next_index}.3 Reported Metrics",
        "",
        build_entity_table(extraction, "metrics", "Metric"),
        "",
        f"## {next_index + 1}. Research Gaps and Future Directions",
        "",
        build_gap_section(gaps),
        "",
    ]

    if graph_stats:
        sections += [
            f"## {next_index + 2}. Knowledge Graph Summary",
            "",
            f"The knowledge graph contains **{graph_stats.get('node_count', 0)} nodes** and "
            f"**{graph_stats.get('edge_count', 0)} relationships** "
            f"(backend: {graph_stats.get('backend', 'n/a')}).",
            "",
            markdown_table(
                ["Node type", "Count"], [[k, v] for k, v in (graph_stats.get("node_types") or {}).items()]
            ),
            "",
        ]
        references_index = next_index + 3
    else:
        references_index = next_index + 2

    sections += [f"## {references_index}. References", "", build_references(papers), ""]
    return "\n".join(sections)


# --- PDF export --------------------------------------------------------------

INLINE_CODE_RE = re.compile(r"`([^`]+)`")
BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)|_([^_]+)_")


def _inline_to_rl(text: str) -> str:
    """Convert the small subset of Markdown we emit into ReportLab markup."""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    escaped = escaped.replace("\\|", "|")
    escaped = INLINE_CODE_RE.sub(r"<font face='Courier'>\1</font>", escaped)
    escaped = BOLD_RE.sub(r"<b>\1</b>", escaped)
    escaped = ITALIC_RE.sub(lambda m: f"<i>{m.group(1) or m.group(2)}</i>", escaped)
    return escaped


def export_pdf(markdown_text: str, output_path: str | Path, title: str) -> Path:
    """Render the Markdown report to a paginated PDF."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.5, leading=14, alignment=TA_JUSTIFY)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, spaceBefore=6, spaceAfter=12)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13.5, spaceBefore=14, spaceAfter=8)
    h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11.5, spaceBefore=10, spaceAfter=6)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=14, bulletIndent=4, spaceAfter=3)
    cell = ParagraphStyle("Cell", parent=body, fontSize=7.8, leading=10, alignment=0)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        title=title,
        author="ResearchCompass",
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    flowables: list = []
    table_buffer: list[list[str]] = []

    def flush_table() -> None:
        nonlocal table_buffer
        if not table_buffer:
            return
        rows = [[Paragraph(_inline_to_rl(c), cell) for c in row] for row in table_buffer]
        column_count = max(len(row) for row in rows)
        available = doc.width
        table = Table(rows, colWidths=[available / column_count] * column_count, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c4d0")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f8")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        flowables.extend([table, Spacer(1, 10)])
        table_buffer = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()

        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= {"-", ":", " "} and c for c in cells):
                continue  # markdown divider row
            table_buffer.append(cells)
            continue
        flush_table()

        if not line.strip():
            flowables.append(Spacer(1, 5))
        elif line.startswith("# "):
            flowables.append(Paragraph(_inline_to_rl(line[2:]), h1))
        elif line.startswith("## "):
            flowables.append(Paragraph(_inline_to_rl(line[3:]), h2))
        elif line.startswith("### "):
            flowables.append(Paragraph(_inline_to_rl(line[4:]), h3))
        elif line.startswith(("- ", "* ")):
            flowables.append(Paragraph(_inline_to_rl(line[2:]), bullet, bulletText="•"))
        elif re.match(r"^\d+\.\s", line):
            flowables.append(Paragraph(_inline_to_rl(re.sub(r"^\d+\.\s", "", line)), bullet, bulletText="•"))
        elif line.strip() == "---":
            flowables.append(PageBreak())
        else:
            flowables.append(Paragraph(_inline_to_rl(line), body))

    flush_table()
    doc.build(flowables)
    return output_path


def report_path(report_id: str) -> Path:
    return Path(settings.report_dir) / f"{report_id}.pdf"
