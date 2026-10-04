"""Knowledge graph read endpoints for the React Force Graph visualisation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlmodel import Session

from app.api.deps import current_user, owned_paper_ids
from app.database import get_session
from app.models import User
from app.schemas import GraphResponse
from app.services.graph_store import get_graph_store

router = APIRouter(prefix="/api/graph", tags=["graph"])

EMPTY = GraphResponse(
    nodes=[], edges=[], backend="none", stats={"node_count": 0, "edge_count": 0}
)


def _scope(session: Session, user: User, paper_ids: list[str] | None) -> list[str]:
    """The papers whose graph this user may see.

    One graph database holds every account's entities, so the paper filter is
    the only thing separating them — it is never optional here.
    """
    return owned_paper_ids(session, user, paper_ids or None)


@router.get("", response_model=GraphResponse, summary="Fetch the knowledge graph")
def get_graph(
    paper_ids: list[str] | None = Query(default=None, description="Restrict to these papers"),
    limit: int = Query(default=2000, ge=10, le=10000),
    node_types: list[str] | None = Query(default=None, description="Restrict to these node types"),
    min_degree: int = Query(default=0, ge=0, le=50, description="Drop nodes with fewer connections"),
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> GraphResponse:
    scope = _scope(session, user, paper_ids)
    if not scope:
        return EMPTY

    store = get_graph_store()
    graph = store.fetch(paper_ids=scope, limit=limit)

    nodes = graph["nodes"]
    edges = graph["edges"]

    if node_types:
        wanted = set(node_types)
        nodes = [node for node in nodes if node.get("type") in wanted]
        node_ids = {node["id"] for node in nodes}
        edges = [edge for edge in edges if edge["source"] in node_ids and edge["target"] in node_ids]

    if min_degree > 0:
        degree: dict[str, int] = {}
        for edge in edges:
            degree[edge["source"]] = degree.get(edge["source"], 0) + 1
            degree[edge["target"]] = degree.get(edge["target"], 0) + 1
        nodes = [node for node in nodes if degree.get(node["id"], 0) >= min_degree]
        node_ids = {node["id"] for node in nodes}
        edges = [edge for edge in edges if edge["source"] in node_ids and edge["target"] in node_ids]

    node_types_count: dict[str, int] = {}
    edge_types_count: dict[str, int] = {}
    for node in nodes:
        key = node.get("type", "Concept")
        node_types_count[key] = node_types_count.get(key, 0) + 1
    for edge in edges:
        key = edge.get("type", "RELATED_TO")
        edge_types_count[key] = edge_types_count.get(key, 0) + 1

    return GraphResponse(
        nodes=nodes,
        edges=edges,
        backend=graph.get("backend", "unknown"),
        stats={
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_types": dict(sorted(node_types_count.items(), key=lambda kv: -kv[1])),
            "edge_types": dict(sorted(edge_types_count.items(), key=lambda kv: -kv[1])),
        },
    )


@router.get("/stats", summary="Knowledge graph statistics")
def graph_stats(
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> dict:
    scope = _scope(session, user, None)
    if not scope:
        return {"backend": get_graph_store().stats().get("backend", "unknown"),
                "node_count": 0, "edge_count": 0, "node_types": {}, "edge_types": {}}

    graph = get_graph_store().fetch(paper_ids=scope, limit=10000)
    node_types: dict[str, int] = {}
    edge_types: dict[str, int] = {}
    for node in graph["nodes"]:
        key = node.get("type", "Concept")
        node_types[key] = node_types.get(key, 0) + 1
    for edge in graph["edges"]:
        key = edge.get("type", "RELATED_TO")
        edge_types[key] = edge_types.get(key, 0) + 1

    return {
        "backend": graph.get("backend", "unknown"),
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "node_types": dict(sorted(node_types.items(), key=lambda kv: -kv[1])),
        "edge_types": dict(sorted(edge_types.items(), key=lambda kv: -kv[1])),
    }


@router.get("/neighbors/{node_id:path}", summary="Neighbourhood of a single node")
def neighbors(
    node_id: str,
    depth: int = Query(default=1, ge=1, le=3),
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> GraphResponse:
    scope = _scope(session, user, None)
    if not scope:
        raise HTTPException(status_code=404, detail="Node not found.")
    graph = get_graph_store().fetch(paper_ids=scope, limit=10000)
    by_id = {node["id"]: node for node in graph["nodes"]}
    if node_id not in by_id:
        raise HTTPException(status_code=404, detail="Node not found.")

    frontier = {node_id}
    visited = {node_id}
    for _ in range(depth):
        next_frontier: set[str] = set()
        for edge in graph["edges"]:
            if edge["source"] in frontier and edge["target"] not in visited:
                next_frontier.add(edge["target"])
            if edge["target"] in frontier and edge["source"] not in visited:
                next_frontier.add(edge["source"])
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break

    nodes = [by_id[i] for i in visited if i in by_id]
    edges = [e for e in graph["edges"] if e["source"] in visited and e["target"] in visited]
    return GraphResponse(
        nodes=nodes,
        edges=edges,
        backend=graph.get("backend", "unknown"),
        stats={"node_count": len(nodes), "edge_count": len(edges)},
    )


@router.delete("", status_code=204, response_class=Response, summary="Clear your knowledge graph")
def clear_graph(
    session: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> Response:
    """Drop this account's contribution, paper by paper.

    Deliberately not `store.clear()`: one graph database holds everybody's
    entities, so emptying it would delete other accounts' work to satisfy one
    person's "clear" button.
    """
    store = get_graph_store()
    for paper_id in _scope(session, user, None):
        store.delete_paper(paper_id)
    return Response(status_code=204)
