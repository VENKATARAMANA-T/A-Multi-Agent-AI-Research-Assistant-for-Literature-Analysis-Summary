"""Literature-Based Discovery — Swanson's ABC model over the knowledge graph.

If one paper links A→B and a different paper links B→C, but nothing links A→C
and no single paper mentions both A and C, then A→C is a connection the
literature implies without ever stating. Swanson found the fish-oil/Raynaud's
link this way in 1986, from two bodies of work that did not cite each other.

Two rules decide whether a candidate is a discovery rather than an artefact:

*Disjointness.* A and C must share no paper at all. If they co-occur even once
the connection is already known, and proposing it is noise.

*Hub suppression.* An intermediate term connected to everything — "model",
"accuracy", "dataset" — links everything to everything. Each B is therefore
weighted by the inverse of its degree, so a term shared by two entities counts
for far more than one shared by forty. Without this the ranking is meaningless;
it is the classic failure mode of naive ABC implementations.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Entity types that can take part in a hypothesis. Papers, authors and figures
# are structural: two papers sharing an author is not a scientific discovery.
SCIENTIFIC_TYPES = frozenset(
    {"Method", "Dataset", "Concept", "Task", "Metric", "Tool", "Application"}
)

# Relations that carry no scientific meaning for this purpose.
STRUCTURAL_RELATIONS = frozenset({"MENTIONS", "AUTHORED_BY", "CONTAINS"})

MIN_SUPPORT = 1
DEFAULT_LIMIT = 25


def _resolve_entities(nodes: dict[str, dict]) -> dict[str, str]:
    """Map each node id to the id of the entity it really is.

    Two nodes are the same entity when any of their surface forms match — the
    full name, or the acronym a paper introduced it with. The node with the
    most papers wins as the representative, so merged entities keep the
    best-attested identity.
    """
    from app.agents.cleanup import name_variants

    by_variant: dict[str, list[str]] = defaultdict(list)
    for node_id, node in nodes.items():
        for variant in name_variants(node.get("name")):
            key = variant.strip().lower()
            if len(key) >= 2:
                by_variant[key].append(node_id)

    # Union-find over nodes that share any surface form.
    parent: dict[str, str] = {node_id: node_id for node_id in nodes}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return
        # Prefer the entity attested in more papers as the representative.
        papers_a = len(nodes[root_a].get("papers") or [])
        papers_b = len(nodes[root_b].get("papers") or [])
        if papers_b > papers_a:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a

    for group in by_variant.values():
        for other in group[1:]:
            union(group[0], other)

    return {node_id: find(node_id) for node_id in nodes}


@dataclass
class Link:
    """One A→B or B→C step, with the relation and the papers asserting it."""

    other: str
    relation: str
    papers: list[str] = field(default_factory=list)
    direction: str = "out"


@dataclass
class Chain:
    """A single A→B→C path."""

    b_id: str
    b_name: str
    b_type: str
    a_to_b: str
    b_to_c: str
    a_papers: list[str] = field(default_factory=list)
    c_papers: list[str] = field(default_factory=list)
    weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "b_id": self.b_id,
            "b_name": self.b_name,
            "b_type": self.b_type,
            "a_to_b": self.a_to_b,
            "b_to_c": self.b_to_c,
            "a_papers": self.a_papers,
            "c_papers": self.c_papers,
            "weight": round(self.weight, 4),
        }


@dataclass
class Candidate:
    """A proposed A→C connection, with the chains that imply it."""

    a_id: str
    a_name: str
    c_id: str
    c_name: str
    c_type: str
    chains: list[Chain] = field(default_factory=list)
    score: float = 0.0
    a_paper_count: int = 0
    c_paper_count: int = 0

    @property
    def support(self) -> int:
        """Number of distinct intermediate terms linking A and C."""
        return len({chain.b_id for chain in self.chains})

    def to_dict(self) -> dict[str, Any]:
        return {
            "a_id": self.a_id,
            "a_name": self.a_name,
            "c_id": self.c_id,
            "c_name": self.c_name,
            "c_type": self.c_type,
            "support": self.support,
            "score": round(self.score, 4),
            "chains": [chain.to_dict() for chain in self.chains],
            "a_paper_count": self.a_paper_count,
            "c_paper_count": self.c_paper_count,
        }


class DiscoveryGraph:
    """Adjacency over the knowledge graph, restricted to scientific entities.

    Entities are resolved before traversal. Papers introduce a term in full and
    then use its acronym, so "Convolutional Neural Network (CNN)" in one paper
    and "CNN" in another arrive as two unconnected nodes. Nothing then bridges
    the two papers, every chain stays inside a single paper, and the
    disjointness rule correctly rejects all of them — the method produces
    nothing. Merging aliases is what creates the cross-paper bridges the
    technique depends on.
    """

    def __init__(
        self,
        graph: dict[str, Any],
        allowed_types: Iterable[str] | None = None,
        resolve_entities: bool = True,
    ):
        allowed = frozenset(allowed_types) if allowed_types else SCIENTIFIC_TYPES

        raw = {
            node["id"]: node
            for node in graph.get("nodes", [])
            if node.get("type") in allowed and node.get("name")
        }

        self._alias: dict[str, str] = (
            _resolve_entities(raw) if resolve_entities else {node_id: node_id for node_id in raw}
        )

        # Collapse merged nodes, unioning their papers and keeping the longest
        # name as the representative label.
        self.nodes: dict[str, dict] = {}
        for node_id, node in raw.items():
            canonical = self._alias[node_id]
            existing = self.nodes.get(canonical)
            if existing is None:
                self.nodes[canonical] = {**node, "id": canonical, "papers": list(node.get("papers") or []),
                                         "aliases": [node.get("name")]}
                continue
            existing["papers"] = sorted(set(existing["papers"]) | set(node.get("papers") or []))
            existing["aliases"].append(node.get("name"))
            if len(node.get("name") or "") > len(existing.get("name") or ""):
                existing["name"] = node["name"]

        self.adjacency: dict[str, list[Link]] = defaultdict(list)
        self._edge_keys: set[tuple[str, str]] = set()
        seen: set[tuple[str, str, str]] = set()

        for edge in graph.get("edges", []):
            relation = edge.get("type", "RELATED_TO")
            if relation in STRUCTURAL_RELATIONS:
                continue
            source = self._alias.get(edge.get("source"))
            target = self._alias.get(edge.get("target"))
            if source is None or target is None or source == target:
                continue
            if (source, target, relation) in seen:
                continue
            seen.add((source, target, relation))

            papers = list(edge.get("papers") or [])
            self.adjacency[source].append(Link(target, relation, papers, "out"))
            self.adjacency[target].append(Link(source, relation, papers, "in"))
            # Undirected for the "already connected?" test: a stated link in
            # either direction means A and C are not strangers.
            self._edge_keys.add((source, target))
            self._edge_keys.add((target, source))

    def __len__(self) -> int:
        return len(self.nodes)

    def degree(self, node_id: str) -> int:
        return len(self.adjacency.get(node_id, []))

    def connected(self, a: str, c: str) -> bool:
        return (a, c) in self._edge_keys

    def papers_of(self, node_id: str) -> set[str]:
        return set(self.nodes.get(node_id, {}).get("papers") or [])

    def name_of(self, node_id: str) -> str:
        return self.nodes.get(node_id, {}).get("name") or node_id

    def find(self, term: str) -> str | None:
        """Resolve a user-supplied term to a node id."""
        needle = (term or "").strip().lower()
        if not needle:
            return None
        if needle in self.nodes:
            return needle
        for node_id, node in self.nodes.items():
            if (node.get("name") or "").strip().lower() == needle:
                return node_id
        # Fall back to a containment match, longest name first so "wav2vec"
        # does not match "wav2vec feature extraction pipeline" by accident.
        matches = [
            node_id
            for node_id, node in sorted(
                self.nodes.items(), key=lambda kv: len(kv[1].get("name") or "")
            )
            if needle in (node.get("name") or "").lower()
        ]
        return matches[0] if matches else None

    def hub_weight(self, node_id: str) -> float:
        """Inverse-degree weight: a widely-connected term carries little signal."""
        degree = self.degree(node_id)
        if degree <= 0:
            return 0.0
        return 1.0 / math.log2(degree + 2)


def open_discovery(
    graph: dict[str, Any],
    source_term: str,
    limit: int = DEFAULT_LIMIT,
    min_support: int = MIN_SUPPORT,
    allowed_types: Iterable[str] | None = None,
    require_disjoint: bool = True,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Given A, propose every C the literature implies but never states.

    Returns (candidates, diagnostics). The diagnostics explain *why* a small
    corpus produced few results, which matters because the technique needs a
    large literature to be productive.
    """
    discovery = DiscoveryGraph(graph, allowed_types)
    diagnostics: dict[str, Any] = {
        "entities": len(discovery),
        "source_resolved": None,
        "b_terms": 0,
        "c_examined": 0,
        "rejected_already_linked": 0,
        "rejected_shared_paper": 0,
    }

    a_id = discovery.find(source_term)
    if a_id is None:
        diagnostics["reason"] = f"'{source_term}' is not in the knowledge graph."
        return [], diagnostics

    diagnostics["source_resolved"] = discovery.name_of(a_id)
    a_papers = discovery.papers_of(a_id)

    b_links = discovery.adjacency.get(a_id, [])
    diagnostics["b_terms"] = len(b_links)

    grouped: dict[str, Candidate] = {}

    for a_to_b in b_links:
        b_id = a_to_b.other
        weight = discovery.hub_weight(b_id)

        for b_to_c in discovery.adjacency.get(b_id, []):
            c_id = b_to_c.other
            if c_id == a_id or c_id == b_id:
                continue
            diagnostics["c_examined"] += 1

            # Already stated: not a discovery.
            if discovery.connected(a_id, c_id):
                diagnostics["rejected_already_linked"] += 1
                continue

            c_papers = discovery.papers_of(c_id)
            if require_disjoint and (a_papers & c_papers):
                diagnostics["rejected_shared_paper"] += 1
                continue

            candidate = grouped.get(c_id)
            if candidate is None:
                candidate = Candidate(
                    a_id=a_id,
                    a_name=discovery.name_of(a_id),
                    c_id=c_id,
                    c_name=discovery.name_of(c_id),
                    c_type=discovery.nodes[c_id].get("type", "Concept"),
                    a_paper_count=len(a_papers),
                    c_paper_count=len(c_papers),
                )
                grouped[c_id] = candidate

            candidate.chains.append(
                Chain(
                    b_id=b_id,
                    b_name=discovery.name_of(b_id),
                    b_type=discovery.nodes[b_id].get("type", "Concept"),
                    a_to_b=a_to_b.relation,
                    b_to_c=b_to_c.relation,
                    a_papers=sorted(a_to_b.papers),
                    c_papers=sorted(b_to_c.papers),
                    weight=weight,
                )
            )

    candidates = []
    for candidate in grouped.values():
        if candidate.support < min_support:
            continue
        # Score on distinct intermediates, each discounted by how hub-like it
        # is, so many weak paths never outrank a few specific ones.
        best_per_b: dict[str, float] = {}
        for chain in candidate.chains:
            best_per_b[chain.b_id] = max(best_per_b.get(chain.b_id, 0.0), chain.weight)
        candidate.score = sum(best_per_b.values())
        candidate.chains.sort(key=lambda chain: -chain.weight)
        candidates.append(candidate)

    candidates.sort(key=lambda item: (-item.score, -item.support, item.c_name))
    diagnostics["candidates_found"] = len(candidates)
    return candidates[:limit], diagnostics


def closed_discovery(
    graph: dict[str, Any],
    source_term: str,
    target_term: str,
    allowed_types: Iterable[str] | None = None,
) -> tuple[list[Chain], dict[str, Any]]:
    """Given A and C, find every B that links them.

    Used to interrogate a specific hunch rather than to generate new ones.
    """
    discovery = DiscoveryGraph(graph, allowed_types)
    diagnostics: dict[str, Any] = {"entities": len(discovery)}

    a_id = discovery.find(source_term)
    c_id = discovery.find(target_term)

    if a_id is None or c_id is None:
        missing = source_term if a_id is None else target_term
        diagnostics["reason"] = f"'{missing}' is not in the knowledge graph."
        return [], diagnostics

    diagnostics.update(
        {
            "source_resolved": discovery.name_of(a_id),
            "target_resolved": discovery.name_of(c_id),
            "already_linked": discovery.connected(a_id, c_id),
            "shared_papers": sorted(discovery.papers_of(a_id) & discovery.papers_of(c_id)),
        }
    )

    from_a = {link.other: link for link in discovery.adjacency.get(a_id, [])}
    chains: list[Chain] = []

    for link in discovery.adjacency.get(c_id, []):
        b_id = link.other
        if b_id in (a_id, c_id) or b_id not in from_a:
            continue
        a_to_b = from_a[b_id]
        chains.append(
            Chain(
                b_id=b_id,
                b_name=discovery.name_of(b_id),
                b_type=discovery.nodes[b_id].get("type", "Concept"),
                a_to_b=a_to_b.relation,
                b_to_c=link.relation,
                a_papers=sorted(a_to_b.papers),
                c_papers=sorted(link.papers),
                weight=discovery.hub_weight(b_id),
            )
        )

    chains.sort(key=lambda chain: -chain.weight)
    diagnostics["links_found"] = len(chains)
    return chains, diagnostics


def rankable_terms(graph: dict[str, Any], allowed_types: Iterable[str] | None = None,
                   limit: int = 40) -> list[dict[str, Any]]:
    """Entities worth using as a starting point, most connected first."""
    discovery = DiscoveryGraph(graph, allowed_types)
    entries = [
        {
            "id": node_id,
            "name": discovery.name_of(node_id),
            "type": node.get("type", "Concept"),
            "degree": discovery.degree(node_id),
            "papers": len(discovery.papers_of(node_id)),
        }
        for node_id, node in discovery.nodes.items()
        if discovery.degree(node_id) > 0
    ]
    entries.sort(key=lambda entry: (-entry["degree"], entry["name"]))
    return entries[:limit]
