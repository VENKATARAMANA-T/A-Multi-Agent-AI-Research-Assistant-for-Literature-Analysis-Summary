"""Knowledge Graph Agent — LLM relation extraction, then persistence to Neo4j."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from app.agents.base import llm_node, merge_deltas
from app.agents.cleanup import clean_name
from app.agents.parallel import map_parallel
from app.agents.prompts import GRAPH_SCHEMA, GRAPH_SYSTEM, graph_prompt
from app.agents.state import AgentState, format_paper_context, trace_event

logger = logging.getLogger(__name__)

ENTITY_TYPES = {"Concept", "Method", "Dataset", "Metric", "Task", "Author", "Tool", "Application"}
RELATION_TYPES = {
    "USES",
    "EVALUATED_ON",
    "PROPOSES",
    "IMPROVES",
    "COMPARES_WITH",
    "PART_OF",
    "APPLIED_TO",
    "MEASURES",
    "EXTENDS",
    "RELATED_TO",
}


def canonical_key(name: str) -> str:
    """Normalise an entity name so 'BERT', 'bert' and 'BERT [S1]' become one node."""
    cleaned = re.sub(r"\s+", " ", clean_name(name)).strip().strip(".,;:")
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.lower()


def build_graph_node(state: AgentState) -> dict[str, Any]:
    """Extract entities/relations per paper, merge them, and persist the graph."""
    started = time.perf_counter()
    documents = state.get("documents") or []
    if not documents:
        return {
            "errors": ["graph: no document loaded"],
            "trace": [trace_event("graph", "skipped", started, reason="no document")],
            "llm_calls": 0,
        }

    entities: dict[str, dict] = {}
    relations: dict[tuple[str, str, str], dict] = {}
    paper_nodes: list[dict] = []

    def extract_one(document: dict) -> dict[str, Any]:
        title = document.get("title") or document.get("id") or "Untitled"
        # Fresh bookkeeping per paper, otherwise the parent's counters are
        # copied in and then added back, double-counting every call.
        single_state: AgentState = {
            **state,
            "context": format_paper_context([document]),
            "trace": [],
            "errors": [],
            "llm_calls": 0,
        }
        return llm_node(
            name=f"graph[{title[:40]}]",
            state=single_state,
            output_key="graph",
            build_prompt=lambda s, t=title: graph_prompt(t, s.get("context", "")),
            system_instruction=GRAPH_SYSTEM,
            schema=GRAPH_SCHEMA,
            temperature=0.1,
        )

    # Fan the independent per-paper calls out, then merge sequentially below:
    # merging mutates the shared entity/relation maps, so it stays on one thread
    # and processes results in input order for a deterministic graph.
    outcomes = map_parallel(extract_one, documents, label="graph")
    merged = merge_deltas(outcomes)

    for document, single in zip(documents, outcomes):
        paper_id = document.get("id")
        title = document.get("title") or paper_id or "Untitled"
        paper_nodes.append(
            {
                "id": f"paper:{paper_id}",
                "name": title,
                "type": "Paper",
                "paper_id": paper_id,
                "year": document.get("year"),
                "authors": document.get("authors") or [],
            }
        )

        if single is not None:
            _merge_fragment(single.get("graph") or {}, paper_id, title, entities, relations)

        # Authors become first-class nodes so co-authorship is visible.
        for author in (document.get("authors") or [])[:12]:
            key = canonical_key(author)
            if not key:
                continue
            node = entities.setdefault(
                key,
                {"id": f"entity:{key}", "name": author.strip(), "type": "Author", "papers": [], "description": ""},
            )
            if paper_id not in node["papers"]:
                node["papers"].append(paper_id)
            relations.setdefault(
                (f"paper:{paper_id}", node["id"], "AUTHORED_BY"),
                {
                    "source": f"paper:{paper_id}",
                    "target": node["id"],
                    "type": "AUTHORED_BY",
                    "papers": [paper_id],
                    "evidence": "",
                },
            )

    # Figures that the vision agent has read know which datasets, methods and
    # metrics they depict. Linking them in means a chart is reachable from the
    # concepts it illustrates, not just from its paper.
    _merge_figures(documents, entities, relations, paper_nodes)

    graph = {
        "nodes": paper_nodes + list(entities.values()),
        "edges": list(relations.values()),
        "paper_ids": [doc.get("id") for doc in documents],
    }
    errors = list(merged["errors"])
    persisted = False
    try:
        from app.services.graph_store import get_graph_store

        persisted = get_graph_store().persist(graph)
    except Exception as exc:  # pragma: no cover - Neo4j optional
        logger.warning("Could not persist knowledge graph: %s", exc)
        errors.append(f"graph_store: {exc}")

    return {
        "graph": graph,
        "errors": errors,
        "llm_calls": merged["llm_calls"],
        "trace": [
            *merged["trace"],
            trace_event(
                "graph",
                "ok",
                started,
                nodes=len(graph["nodes"]),
                edges=len(graph["edges"]),
                persisted=persisted,
            ),
        ],
    }


FIGURE_ENTITY_RELATIONS = {
    "datasets": "EVALUATED_ON",
    "methods": "USES",
    "metrics": "MEASURES",
}


def _merge_figures(
    documents: list[dict],
    entities: dict[str, dict],
    relations: dict[tuple[str, str, str], dict],
    paper_nodes: list[dict],
) -> None:
    """Add analysed figures to the graph, linked to what they depict."""
    try:
        from sqlmodel import select

        from app.database import session_scope
        from app.models import Figure, FigureStatus
    except Exception:  # pragma: no cover
        return

    paper_ids = [doc.get("id") for doc in documents if doc.get("id")]
    if not paper_ids:
        return

    try:
        with session_scope() as session:
            figures = list(
                session.exec(
                    select(Figure).where(
                        Figure.paper_id.in_(paper_ids),  # type: ignore[attr-defined]
                        Figure.status == FigureStatus.ANALYSED,
                    )
                ).all()
            )
    except Exception:  # pragma: no cover
        logger.warning("Could not load figures for the graph", exc_info=True)
        return

    for figure in figures:
        node_id = f"figure:{figure.id}"
        paper_nodes.append(
            {
                "id": node_id,
                "name": f"{figure.label} — {figure.chart_type or figure.kind}",
                "type": "Figure",
                "paper_id": figure.paper_id,
                "description": (figure.takeaway or figure.caption or "")[:400],
                "papers": [figure.paper_id],
            }
        )
        relations.setdefault(
            (f"paper:{figure.paper_id}", node_id, "CONTAINS"),
            {
                "source": f"paper:{figure.paper_id}",
                "target": node_id,
                "type": "CONTAINS",
                "papers": [figure.paper_id],
                "evidence": figure.label,
            },
        )

        # A figure depicting a dataset or method connects to that entity, so
        # the chart is reachable from the concept it illustrates.
        for field, relation in FIGURE_ENTITY_RELATIONS.items():
            for name in (figure.entities or {}).get(field, []) or []:
                key = canonical_key(name)
                if not key:
                    continue
                target = entities.get(key)
                if target is None:
                    continue  # only link concepts the paper itself established
                relations.setdefault(
                    (node_id, target["id"], relation),
                    {
                        "source": node_id,
                        "target": target["id"],
                        "type": relation,
                        "papers": [figure.paper_id],
                        "evidence": f"depicted in {figure.label}",
                    },
                )


def _merge_fragment(
    fragment: dict,
    paper_id: str | None,
    paper_title: str,
    entities: dict[str, dict],
    relations: dict[tuple[str, str, str], dict],
) -> None:
    """Fold one paper's entities/relations into the corpus-wide graph."""
    local_ids: dict[str, str] = {}

    for entity in fragment.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        name = clean_name(entity.get("name"))
        if not name or len(name) > 120:
            continue
        entity_type = str(entity.get("type") or "Concept")
        if entity_type not in ENTITY_TYPES:
            entity_type = "Concept"

        key = canonical_key(name)
        node = entities.get(key)
        if node is None:
            node = {
                "id": f"entity:{key}",
                "name": name,
                "type": entity_type,
                "description": str(entity.get("description") or "")[:400],
                "papers": [],
            }
            entities[key] = node
        elif not node.get("description") and entity.get("description"):
            node["description"] = str(entity["description"])[:400]

        if paper_id and paper_id not in node["papers"]:
            node["papers"].append(paper_id)
        local_ids[key] = node["id"]

        # Connect the paper to each entity it mentions.
        if paper_id:
            relations.setdefault(
                (f"paper:{paper_id}", node["id"], "MENTIONS"),
                {
                    "source": f"paper:{paper_id}",
                    "target": node["id"],
                    "type": "MENTIONS",
                    "papers": [paper_id],
                    "evidence": "",
                },
            )

    for relation in fragment.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        source_key = canonical_key(relation.get("source") or "")
        target_key = canonical_key(relation.get("target") or "")
        relation_type = str(relation.get("type") or "RELATED_TO").upper()
        if relation_type not in RELATION_TYPES:
            relation_type = "RELATED_TO"
        if not source_key or not target_key or source_key == target_key:
            continue

        # A relation may reference the paper itself as the subject.
        paper_key = canonical_key(paper_title)
        source_id = local_ids.get(source_key) or entities.get(source_key, {}).get("id")
        target_id = local_ids.get(target_key) or entities.get(target_key, {}).get("id")
        if source_key == paper_key and paper_id:
            source_id = f"paper:{paper_id}"
        if target_key == paper_key and paper_id:
            target_id = f"paper:{paper_id}"
        if not source_id or not target_id:
            continue  # hallucinated endpoint — drop the edge

        edge_key = (source_id, target_id, relation_type)
        edge = relations.get(edge_key)
        if edge is None:
            relations[edge_key] = {
                "source": source_id,
                "target": target_id,
                "type": relation_type,
                "evidence": str(relation.get("evidence") or "")[:300],
                "papers": [paper_id] if paper_id else [],
            }
        elif paper_id and paper_id not in edge["papers"]:
            edge["papers"].append(paper_id)
