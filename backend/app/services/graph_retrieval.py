"""GraphRAG — retrieve context by traversing the knowledge graph.

Vector search finds passages that *resemble* the question. Some questions have
no such passage: "which methods were evaluated on the same dataset as wav2vec?"
is answered by the relationships between chunks, not by any one of them, and no
amount of similarity search will surface it.

This module answers those by walking the graph instead. The question is linked
to entities, the neighbourhood around them is collected, and the edges are
serialised as facts the model can read and cite — each carrying the papers that
assert it, so a graph-derived answer is as checkable as a text-derived one.

Entity linking is the whole game. A wrongly linked entity produces confidently
wrong context, so matching is deliberately conservative: exact surface forms
first, including acronym expansions, then embedding similarity only above a
confidence floor. When nothing matches, this returns empty and the caller falls
back to vector search rather than inventing a subgraph.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.agents.cleanup import name_variants
from app.services.embeddings import cosine_similarity, get_embedder

logger = logging.getLogger(__name__)

# Structural edges say nothing a reader needs: "paper MENTIONS entity" and
# "paper AUTHORED_BY person" are bookkeeping, not findings, and including them
# fills the context with noise in an answer about how two methods relate.
STRUCTURAL_RELATIONS = frozenset({"MENTIONS", "CONTAINS", "AUTHORED_BY"})

# Entity kinds that cannot answer a research question.
NON_FACTUAL_TYPES = frozenset({"Paper", "Figure", "Author"})

STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have how i if in
    into is it its may might must of on or should than that the their then there these they
    this those to use used uses using was we were what when where which while who why will
    with would you your about across after also any because before between both during each
    more most other over same some such only own very
    """.split()
)

MAX_NGRAM = 4
MIN_TERM_LENGTH = 3
# Below this the match is a coincidence; a wrong entity is worse than none.
EMBEDDING_FLOOR = 0.55
MAX_SEEDS = 6
DEFAULT_HOPS = 2
MAX_FACTS = 60


@dataclass
class EntityMatch:
    node_id: str
    name: str
    type: str
    confidence: float
    method: str  # "exact" | "alias" | "embedding"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "type": self.type,
            "confidence": round(self.confidence, 3),
            "method": self.method,
        }


@dataclass
class GraphFact:
    source: str
    relation: str
    target: str
    papers: list[str] = field(default_factory=list)
    paper_titles: list[str] = field(default_factory=list)
    evidence: str = ""
    hops: int = 1

    def sentence(self) -> str:
        readable = self.relation.replace("_", " ").lower()
        return f"{self.source} {readable} {self.target}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "papers": self.papers,
            "paper_titles": self.paper_titles,
            "evidence": self.evidence,
            "hops": self.hops,
            "sentence": self.sentence(),
        }


@dataclass
class GraphContext:
    matches: list[EntityMatch] = field(default_factory=list)
    facts: list[GraphFact] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    paper_ids: list[str] = field(default_factory=list)
    reason: str | None = None

    @property
    def found(self) -> bool:
        return bool(self.facts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matches": [match.to_dict() for match in self.matches],
            "facts": [fact.to_dict() for fact in self.facts],
            "node_ids": self.node_ids,
            "paper_ids": self.paper_ids,
            "reason": self.reason,
            "found": self.found,
        }


# --- entity linking ----------------------------------------------------------


def candidate_terms(question: str) -> list[str]:
    """Word n-grams from the question that could name an entity."""
    words = [
        word
        for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/+.]*", question or "")
        if word
    ]
    terms: list[str] = []
    for size in range(min(MAX_NGRAM, len(words)), 0, -1):
        for index in range(len(words) - size + 1):
            gram = " ".join(words[index : index + size])
            if len(gram) < MIN_TERM_LENGTH:
                continue
            if size == 1 and gram.lower() in STOPWORDS:
                continue
            terms.append(gram)
    return terms


def _surface_index(nodes: Iterable[dict]) -> dict[str, str]:
    """Every surface form an entity may be written as, mapped to its node id."""
    index: dict[str, str] = {}
    for node in nodes:
        name = node.get("name")
        if not name:
            continue
        for variant in name_variants(name):
            key = variant.strip().lower()
            if len(key) >= MIN_TERM_LENGTH:
                # A longer, more specific name wins a clash.
                existing = index.get(key)
                if existing is None:
                    index[key] = node["id"]
    return index


def link_entities(
    question: str,
    nodes: list[dict],
    max_seeds: int = MAX_SEEDS,
    use_embeddings: bool = True,
) -> list[EntityMatch]:
    """Resolve the question to entities in the graph, conservatively."""
    if not question or not nodes:
        return []

    by_id = {node["id"]: node for node in nodes if node.get("name")}
    index = _surface_index(by_id.values())

    matches: dict[str, EntityMatch] = {}

    for term in candidate_terms(question):
        node_id = index.get(term.lower())
        if node_id is None or node_id in matches:
            continue
        node = by_id[node_id]
        # A longer literal match is stronger evidence than a one-word hit.
        confidence = min(1.0, 0.75 + 0.05 * len(term.split()))
        exact = term.strip().lower() == (node.get("name") or "").strip().lower()
        matches[node_id] = EntityMatch(
            node_id=node_id,
            name=node["name"],
            type=node.get("type", "Concept"),
            confidence=confidence,
            method="exact" if exact else "alias",
        )

    if use_embeddings and len(matches) < max_seeds:
        matches.update(_embedding_matches(question, by_id, exclude=set(matches)))

    ranked = sorted(matches.values(), key=lambda match: -match.confidence)
    return ranked[:max_seeds]


def _embedding_matches(
    question: str, by_id: dict[str, dict], exclude: set[str]
) -> dict[str, EntityMatch]:
    """Semantic fallback for entities the question paraphrases."""
    candidates = [(node_id, node) for node_id, node in by_id.items() if node_id not in exclude]
    if not candidates:
        return {}

    try:
        embedder = get_embedder()
        question_vector = embedder.embed_query(question)
        name_vectors = embedder.embed_documents([node["name"] for _, node in candidates])
    except Exception:  # pragma: no cover - embedder problems surface elsewhere
        logger.warning("Entity linking fell back to exact matching only", exc_info=True)
        return {}

    found: dict[str, EntityMatch] = {}
    for (node_id, node), vector in zip(candidates, name_vectors):
        score = cosine_similarity(question_vector, vector)
        if score < EMBEDDING_FLOOR:
            continue
        found[node_id] = EntityMatch(
            node_id=node_id,
            name=node["name"],
            type=node.get("type", "Concept"),
            confidence=float(score),
            method="embedding",
        )
    return found


# --- subgraph ----------------------------------------------------------------


def expand(
    graph: dict[str, Any],
    seeds: list[str],
    hops: int = DEFAULT_HOPS,
    max_facts: int = MAX_FACTS,
) -> tuple[list[GraphFact], list[str]]:
    """Collect the edges within `hops` of the seed entities, nearest first."""
    from app.services.lbd import resolve_entities

    nodes = {node["id"]: node for node in graph.get("nodes", []) if node.get("name")}
    titles = {
        node.get("paper_id"): node.get("name")
        for node in graph.get("nodes", [])
        if node.get("type") == "Paper" and node.get("paper_id")
    }

    # The same concept arrives under several names ("CNN" and its expansion),
    # which would otherwise emit the identical fact three times and waste the
    # model's attention on repetition.
    factual = {
        node_id: node
        for node_id, node in nodes.items()
        if node.get("type") not in NON_FACTUAL_TYPES
    }
    alias = resolve_entities(factual)
    canonical_name = {
        node_id: max(
            (nodes[other].get("name") or "" for other, root in alias.items() if root == alias[node_id]),
            key=len,
            default=nodes[node_id].get("name") or node_id,
        )
        for node_id in factual
    }

    def label(node_id: str) -> str:
        return canonical_name.get(node_id) or nodes.get(node_id, {}).get("name") or node_id

    def canonical(node_id: str) -> str:
        return alias.get(node_id, node_id)

    adjacency: dict[str, list[dict]] = {}
    for edge in graph.get("edges", []):
        if edge.get("type") in STRUCTURAL_RELATIONS:
            continue
        source, target = edge.get("source"), edge.get("target")
        if source not in nodes or target not in nodes:
            continue
        adjacency.setdefault(canonical(source), []).append(edge)
        adjacency.setdefault(canonical(target), []).append(edge)

    facts: list[GraphFact] = []
    seen_edges: set[tuple[str, str, str]] = set()
    visited = {canonical(seed) for seed in seeds}
    frontier = sorted(visited)

    for depth in range(1, hops + 1):
        next_frontier: list[str] = []
        for node_id in frontier:
            for edge in adjacency.get(node_id, []):
                source, target = canonical(edge["source"]), canonical(edge["target"])
                relation = edge.get("type", "RELATED_TO")
                if source == target:
                    continue  # collapsed into one entity by resolution
                key = (label(source), relation, label(target))
                if key in seen_edges:
                    continue
                seen_edges.add(key)

                papers = list(edge.get("papers") or [])
                facts.append(
                    GraphFact(
                        source=label(source),
                        relation=relation,
                        target=label(target),
                        papers=papers,
                        paper_titles=[titles.get(paper) for paper in papers if titles.get(paper)],
                        evidence=str(edge.get("evidence") or "")[:200],
                        hops=depth,
                    )
                )
                for other in (source, target):
                    if other not in visited:
                        visited.add(other)
                        next_frontier.append(other)

                if len(facts) >= max_facts:
                    return facts, sorted(visited)
        frontier = next_frontier
        if not frontier:
            break

    return facts, sorted(visited)


def format_facts(facts: list[GraphFact], max_chars: int = 12_000) -> str:
    """Number the facts as [G1], [G2]… so an answer can cite them."""
    lines: list[str] = []
    used = 0
    for index, fact in enumerate(facts, start=1):
        source = ", ".join(fact.paper_titles[:2]) or "unattributed"
        line = f"[G{index}] {fact.sentence()} (from: {source})"
        if fact.evidence:
            line += f" — “{fact.evidence}”"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


# --- entry point -------------------------------------------------------------


def retrieve(
    question: str,
    graph: dict[str, Any] | None = None,
    hops: int = DEFAULT_HOPS,
    max_facts: int = MAX_FACTS,
    paper_ids: list[str] | None = None,
) -> GraphContext:
    """Build graph context for a question, or explain why there is none."""
    if graph is None:
        from app.services.graph_store import get_graph_store

        graph = get_graph_store().fetch(paper_ids=paper_ids or None)

    nodes = graph.get("nodes", [])
    if not nodes:
        return GraphContext(reason="The knowledge graph is empty; build it first.")

    entities = [node for node in nodes if node.get("type") not in NON_FACTUAL_TYPES]
    matches = link_entities(question, entities)

    if not matches:
        return GraphContext(
            reason="No entity in the question matched the knowledge graph.",
        )

    facts, visited = expand(graph, [match.node_id for match in matches], hops, max_facts)
    if not facts:
        return GraphContext(
            matches=matches,
            node_ids=visited,
            reason="The matched entities have no relationships recorded.",
        )

    papers = sorted({paper for fact in facts for paper in fact.papers})
    return GraphContext(matches=matches, facts=facts, node_ids=visited, paper_ids=papers)
