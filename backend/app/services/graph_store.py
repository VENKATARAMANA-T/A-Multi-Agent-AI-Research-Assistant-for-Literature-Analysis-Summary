"""Step 7 — knowledge graph persistence.

Neo4j is the primary store. When it is unreachable (no container running, or
NEO4J_ENABLED=false) we transparently fall back to a JSON-file-backed in-memory
graph so the visualisation still works on a laptop with nothing but Python.
The active backend is reported by /api/health and /api/graph.
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

MERGE_PAPER = """
MERGE (p:Paper {id: $id})
SET p.title = $title, p.year = $year, p.authors = $authors, p.paper_id = $paper_id
"""

MERGE_ENTITY = """
MERGE (e:Entity {id: $id})
SET e.name = $name, e.type = $type, e.description = $description, e.papers = $papers
"""

MERGE_RELATION = """
MATCH (a {id: $source})
MATCH (b {id: $target})
MERGE (a)-[r:RELATION {type: $type}]->(b)
SET r.evidence = $evidence, r.papers = $papers
"""

FETCH_GRAPH = """
MATCH (n)
OPTIONAL MATCH (n)-[r:RELATION]->(m)
RETURN n, r, m
LIMIT $limit
"""


class GraphStore(ABC):
    backend: str

    @abstractmethod
    def persist(self, graph: dict[str, Any]) -> bool: ...

    @abstractmethod
    def fetch(self, paper_ids: list[str] | None = None, limit: int = 2000) -> dict[str, Any]: ...

    @abstractmethod
    def delete_paper(self, paper_id: str) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...

    def stats(self) -> dict[str, Any]:
        graph = self.fetch()
        node_types: dict[str, int] = {}
        edge_types: dict[str, int] = {}
        for node in graph["nodes"]:
            node_types[node.get("type", "Unknown")] = node_types.get(node.get("type", "Unknown"), 0) + 1
        for edge in graph["edges"]:
            edge_types[edge.get("type", "RELATED_TO")] = edge_types.get(edge.get("type", "RELATED_TO"), 0) + 1
        return {
            "backend": self.backend,
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "node_types": dict(sorted(node_types.items(), key=lambda kv: -kv[1])),
            "edge_types": dict(sorted(edge_types.items(), key=lambda kv: -kv[1])),
        }


class InMemoryGraphStore(GraphStore):
    """File-backed graph store — the fallback when Neo4j is unavailable."""

    backend = "in-memory"

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (Path(settings.data_dir) / "knowledge_graph.json")
        self._lock = threading.Lock()
        self._nodes: dict[str, dict] = {}
        self._edges: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            self._nodes = {node["id"]: node for node in payload.get("nodes", [])}
            self._edges = {self._edge_key(edge): edge for edge in payload.get("edges", [])}
        except Exception:  # pragma: no cover - corrupt cache is not fatal
            logger.warning("Could not read %s; starting with an empty graph.", self._path)

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"nodes": list(self._nodes.values()), "edges": list(self._edges.values())}
        self._path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _edge_key(edge: dict) -> str:
        return f"{edge.get('source')}|{edge.get('type')}|{edge.get('target')}"

    def persist(self, graph: dict[str, Any]) -> bool:
        with self._lock:
            for node in graph.get("nodes", []):
                existing = self._nodes.get(node["id"])
                if existing:
                    papers = sorted(set(existing.get("papers", [])) | set(node.get("papers", [])))
                    self._nodes[node["id"]] = {**existing, **node, "papers": papers}
                else:
                    self._nodes[node["id"]] = dict(node)
            for edge in graph.get("edges", []):
                if edge.get("source") not in self._nodes or edge.get("target") not in self._nodes:
                    continue
                key = self._edge_key(edge)
                existing = self._edges.get(key)
                if existing:
                    papers = sorted(set(existing.get("papers", [])) | set(edge.get("papers", [])))
                    self._edges[key] = {**existing, **edge, "papers": papers}
                else:
                    self._edges[key] = dict(edge)
            self._flush()
        return True

    def fetch(self, paper_ids: list[str] | None = None, limit: int = 2000) -> dict[str, Any]:
        with self._lock:
            nodes = list(self._nodes.values())
            edges = list(self._edges.values())

        if paper_ids:
            wanted = set(paper_ids)
            nodes = [
                node
                for node in nodes
                if wanted & set(node.get("papers", []) or [])
                or node.get("paper_id") in wanted
            ]
            node_ids = {node["id"] for node in nodes}
            edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]

        nodes = nodes[:limit]
        node_ids = {node["id"] for node in nodes}
        edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]
        return {"nodes": nodes, "edges": edges, "backend": self.backend}

    def delete_paper(self, paper_id: str) -> None:
        with self._lock:
            self._nodes.pop(f"paper:{paper_id}", None)
            for node_id, node in list(self._nodes.items()):
                papers = [p for p in node.get("papers", []) if p != paper_id]
                if node.get("papers") and not papers:
                    self._nodes.pop(node_id, None)  # entity only came from this paper
                else:
                    node["papers"] = papers
            for key, edge in list(self._edges.items()):
                if edge["source"] not in self._nodes or edge["target"] not in self._nodes:
                    self._edges.pop(key, None)
                    continue
                papers = [p for p in edge.get("papers", []) if p != paper_id]
                if edge.get("papers") and not papers:
                    self._edges.pop(key, None)
                else:
                    edge["papers"] = papers
            self._flush()

    def clear(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._flush()


class Neo4jGraphStore(GraphStore):
    backend = "neo4j"

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        from neo4j import GraphDatabase

        self._database = database
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()
        self._ensure_constraints()

    def _ensure_constraints(self) -> None:
        with self._driver.session(database=self._database) as session:
            session.run("CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
            session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")

    def persist(self, graph: dict[str, Any]) -> bool:
        with self._driver.session(database=self._database) as session:
            for node in graph.get("nodes", []):
                if node.get("type") == "Paper":
                    session.run(
                        MERGE_PAPER,
                        id=node["id"],
                        title=node.get("name"),
                        year=node.get("year"),
                        authors=node.get("authors") or [],
                        paper_id=node.get("paper_id"),
                    )
                else:
                    session.run(
                        MERGE_ENTITY,
                        id=node["id"],
                        name=node.get("name"),
                        type=node.get("type", "Concept"),
                        description=node.get("description", ""),
                        papers=node.get("papers") or [],
                    )
            for edge in graph.get("edges", []):
                session.run(
                    MERGE_RELATION,
                    source=edge["source"],
                    target=edge["target"],
                    type=edge.get("type", "RELATED_TO"),
                    evidence=edge.get("evidence", ""),
                    papers=edge.get("papers") or [],
                )
        return True

    def fetch(self, paper_ids: list[str] | None = None, limit: int = 2000) -> dict[str, Any]:
        nodes: dict[str, dict] = {}
        edges: dict[str, dict] = {}

        with self._driver.session(database=self._database) as session:
            for record in session.run(FETCH_GRAPH, limit=limit):
                for key in ("n", "m"):
                    raw = record.get(key)
                    if raw is None:
                        continue
                    props = dict(raw)
                    node_id = props.get("id")
                    if not node_id:
                        continue
                    is_paper = "Paper" in set(raw.labels)
                    nodes[node_id] = {
                        "id": node_id,
                        "name": props.get("title") if is_paper else props.get("name"),
                        "type": "Paper" if is_paper else props.get("type", "Concept"),
                        "description": props.get("description", ""),
                        "papers": list(props.get("papers") or []),
                        "paper_id": props.get("paper_id"),
                        "year": props.get("year"),
                        "authors": list(props.get("authors") or []),
                    }
                relation = record.get("r")
                source = record.get("n")
                target = record.get("m")
                if relation is not None and source is not None and target is not None:
                    props = dict(relation)
                    source_id = dict(source).get("id")
                    target_id = dict(target).get("id")
                    edge_type = props.get("type", "RELATED_TO")
                    edges[f"{source_id}|{edge_type}|{target_id}"] = {
                        "source": source_id,
                        "target": target_id,
                        "type": edge_type,
                        "evidence": props.get("evidence", ""),
                        "papers": list(props.get("papers") or []),
                    }

        node_list = list(nodes.values())
        if paper_ids:
            wanted = set(paper_ids)
            node_list = [
                node
                for node in node_list
                if wanted & set(node.get("papers") or []) or node.get("paper_id") in wanted
            ]
        node_ids = {node["id"] for node in node_list}
        edge_list = [e for e in edges.values() if e["source"] in node_ids and e["target"] in node_ids]
        return {"nodes": node_list, "edges": edge_list, "backend": self.backend}

    def delete_paper(self, paper_id: str) -> None:
        with self._driver.session(database=self._database) as session:
            session.run("MATCH (p:Paper {id: $id}) DETACH DELETE p", id=f"paper:{paper_id}")
            session.run(
                """
                MATCH (e:Entity)
                WHERE $pid IN e.papers
                SET e.papers = [x IN e.papers WHERE x <> $pid]
                WITH e WHERE size(e.papers) = 0
                DETACH DELETE e
                """,
                pid=paper_id,
            )

    def clear(self) -> None:
        with self._driver.session(database=self._database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def close(self) -> None:
        self._driver.close()


_store: GraphStore | None = None
_lock = threading.Lock()


def get_graph_store() -> GraphStore:
    global _store
    if _store is not None:
        return _store

    with _lock:
        if _store is not None:
            return _store

        if settings.neo4j_enabled:
            try:
                _store = Neo4jGraphStore(
                    settings.neo4j_uri,
                    settings.neo4j_user,
                    settings.neo4j_password,
                    settings.neo4j_database,
                )
                logger.info("Connected to Neo4j at %s", settings.neo4j_uri)
                return _store
            except Exception as exc:
                logger.warning(
                    "Neo4j unavailable at %s (%s). Using the in-memory graph store instead.",
                    settings.neo4j_uri,
                    exc,
                )

        _store = InMemoryGraphStore()
        return _store


def reset_graph_store() -> None:
    """Test hook."""
    global _store
    with _lock:
        if isinstance(_store, Neo4jGraphStore):
            try:
                _store.close()
            except Exception:  # pragma: no cover
                pass
        _store = None
