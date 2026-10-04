"""Literature-Based Discovery — Swanson's ABC model.

The fixture graph reproduces Swanson's original 1986 finding in miniature: one
body of work links dietary fish oil to blood viscosity, a separate body links
blood viscosity to Raynaud's disease, and no paper mentions both ends. The
connection is implied by the literature without ever being stated, which is
exactly what the method exists to surface.
"""

from __future__ import annotations

import pytest

from app.agents.cleanup import clean_name, expand_acronym, name_variants
from app.agents.hypothesis import VERDICT_ORDER, assess_many, build_prompt, normalise
from app.services.lbd import (
    DiscoveryGraph,
    closed_discovery,
    open_discovery,
    rankable_terms,
)


def node(node_id, name, kind, papers):
    return {"id": node_id, "name": name, "type": kind, "papers": papers, "description": ""}


def edge(source, target, relation, papers):
    return {"source": source, "target": target, "type": relation, "papers": papers, "evidence": ""}


@pytest.fixture
def swanson_graph():
    """Two disjoint literatures joined only by a shared intermediate."""
    return {
        "nodes": [
            node("e:fish oil", "Fish Oil", "Concept", ["p1"]),
            node("e:blood viscosity", "Blood Viscosity", "Concept", ["p1", "p2"]),
            node("e:raynaud", "Raynaud's Disease", "Concept", ["p2"]),
            # A hub: connected to everything, so it must be discounted.
            node("e:measurement", "Measurement", "Concept", ["p1", "p2", "p3", "p4"]),
            node("e:unrelated", "Crop Rotation", "Concept", ["p3"]),
            node("e:soil", "Soil Nitrogen", "Concept", ["p4"]),
        ],
        "edges": [
            edge("e:fish oil", "e:blood viscosity", "IMPROVES", ["p1"]),
            edge("e:blood viscosity", "e:raynaud", "APPLIED_TO", ["p2"]),
            edge("e:fish oil", "e:measurement", "USES", ["p1"]),
            edge("e:raynaud", "e:measurement", "USES", ["p2"]),
            edge("e:unrelated", "e:measurement", "USES", ["p3"]),
            edge("e:soil", "e:measurement", "USES", ["p4"]),
        ],
    }


# --- the core finding --------------------------------------------------------


def test_finds_the_implied_connection(swanson_graph):
    candidates, diagnostics = open_discovery(swanson_graph, "Fish Oil")

    names = [c.c_name for c in candidates]
    assert "Raynaud's Disease" in names, diagnostics


def test_the_implied_link_outranks_the_hub_route(swanson_graph):
    """Both Raynaud's and Crop Rotation are two hops away — only one matters.

    They are reachable through "Measurement", a term connected to everything.
    Without inverse-degree weighting, an unrelated agriculture paper scores the
    same as the real finding.
    """
    candidates, _ = open_discovery(swanson_graph, "Fish Oil")
    by_name = {c.c_name: c for c in candidates}

    assert "Raynaud's Disease" in by_name
    if "Crop Rotation" in by_name:
        assert by_name["Raynaud's Disease"].score > by_name["Crop Rotation"].score


def test_the_specific_intermediate_is_named(swanson_graph):
    candidates, _ = open_discovery(swanson_graph, "Fish Oil")
    raynaud = next(c for c in candidates if c.c_name == "Raynaud's Disease")

    intermediates = {chain.b_name for chain in raynaud.chains}
    assert "Blood Viscosity" in intermediates


# --- the two rules that make it a discovery ----------------------------------


def test_an_already_stated_link_is_not_a_discovery(swanson_graph):
    """If a paper states A→C directly, proposing it is noise."""
    graph = {
        "nodes": swanson_graph["nodes"],
        "edges": swanson_graph["edges"] + [edge("e:fish oil", "e:raynaud", "APPLIED_TO", ["p9"])],
    }
    candidates, diagnostics = open_discovery(graph, "Fish Oil")

    assert "Raynaud's Disease" not in [c.c_name for c in candidates]
    assert diagnostics["rejected_already_linked"] >= 1


def test_concepts_sharing_a_paper_are_not_disjoint_literatures(swanson_graph):
    """Swanson's criterion: the two literatures must not overlap at all."""
    graph = {
        "nodes": [
            n if n["id"] != "e:raynaud" else node("e:raynaud", "Raynaud's Disease", "Concept", ["p1", "p2"])
            for n in swanson_graph["nodes"]
        ],
        "edges": swanson_graph["edges"],
    }
    candidates, diagnostics = open_discovery(graph, "Fish Oil", require_disjoint=True)

    assert "Raynaud's Disease" not in [c.c_name for c in candidates]
    assert diagnostics["rejected_shared_paper"] >= 1


def test_relaxed_mode_keeps_pairs_that_merely_share_a_paper(swanson_graph):
    """A small corpus has no disjoint literatures; unstated links still help."""
    graph = {
        "nodes": [
            n if n["id"] != "e:raynaud" else node("e:raynaud", "Raynaud's Disease", "Concept", ["p1", "p2"])
            for n in swanson_graph["nodes"]
        ],
        "edges": swanson_graph["edges"],
    }
    candidates, _ = open_discovery(graph, "Fish Oil", require_disjoint=False)

    assert "Raynaud's Disease" in [c.c_name for c in candidates]


def test_structural_relations_are_ignored(swanson_graph):
    """Two papers sharing an author is not a scientific connection."""
    graph = {
        "nodes": swanson_graph["nodes"] + [node("e:author", "Some Author", "Concept", ["p1", "p3"])],
        "edges": swanson_graph["edges"]
        + [
            edge("e:fish oil", "e:author", "MENTIONS", ["p1"]),
            edge("e:author", "e:soil", "MENTIONS", ["p3"]),
        ],
    }
    candidates, _ = open_discovery(graph, "Fish Oil")
    intermediates = {chain.b_name for c in candidates for chain in c.chains}

    assert "Some Author" not in intermediates


def test_unknown_term_reports_why(swanson_graph):
    candidates, diagnostics = open_discovery(swanson_graph, "Nonexistent Concept")

    assert candidates == []
    assert "not in the knowledge graph" in diagnostics["reason"]


def test_min_support_filters_weakly_linked_pairs(swanson_graph):
    candidates, _ = open_discovery(swanson_graph, "Fish Oil", min_support=2)
    for candidate in candidates:
        assert candidate.support >= 2


# --- hub weighting -----------------------------------------------------------


def test_a_widely_connected_term_carries_less_weight(swanson_graph):
    discovery = DiscoveryGraph(swanson_graph)
    hub = discovery.hub_weight("e:measurement")   # degree 4
    specific = discovery.hub_weight("e:blood viscosity")  # degree 2

    assert specific > hub


def test_an_isolated_node_has_no_weight(swanson_graph):
    discovery = DiscoveryGraph(swanson_graph)
    assert discovery.hub_weight("e:nothing") == 0.0


# --- entity resolution -------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Convolutional Neural Network (CNN)", ("Convolutional Neural Network", "CNN")),
        # Earlier extraction dropped the closing bracket; these must still resolve.
        ("Convolutional Neural Network (CNN", ("Convolutional Neural Network", "CNN")),
        ("Gaussian Mixture Models (GMMs", ("Gaussian Mixture Models", "GMM")),
    ],
)
def test_acronyms_are_recognised(raw, expected):
    assert expand_acronym(raw) == expected


def test_plain_names_have_no_acronym():
    assert expand_acronym("BiLSTM") is None
    assert name_variants("BiLSTM") == ["BiLSTM"]


def test_bracket_stripping_does_not_mangle_an_acronym():
    """An unconditional trailing-bracket strip produced "...Network (CNN"."""
    assert clean_name("Convolutional Neural Network (CNN)") == "Convolutional Neural Network (CNN)"
    assert clean_name("[BERT]") == "BERT"


def test_the_same_concept_written_two_ways_becomes_one_entity():
    """Without this there is no bridge between papers and nothing is found."""
    graph = {
        "nodes": [
            node("e:cnn", "CNN", "Method", ["p1"]),
            node("e:cnn long", "Convolutional Neural Network (CNN)", "Method", ["p2"]),
            node("e:speech", "Speech Classification", "Task", ["p1"]),
            node("e:vision", "Image Segmentation", "Task", ["p2"]),
        ],
        "edges": [
            edge("e:cnn", "e:speech", "APPLIED_TO", ["p1"]),
            edge("e:cnn long", "e:vision", "APPLIED_TO", ["p2"]),
        ],
    }
    discovery = DiscoveryGraph(graph)

    assert len(discovery.nodes) == 3, "CNN and its expansion did not merge"

    # And the merge creates the cross-paper chain that makes discovery possible.
    candidates, _ = open_discovery(graph, "Speech Classification")
    assert "Image Segmentation" in [c.c_name for c in candidates]


def test_resolution_can_be_switched_off():
    graph = {
        "nodes": [
            node("e:cnn", "CNN", "Method", ["p1"]),
            node("e:cnn long", "Convolutional Neural Network (CNN)", "Method", ["p2"]),
        ],
        "edges": [],
    }
    assert len(DiscoveryGraph(graph, merge_aliases=False).nodes) == 2


# --- closed discovery --------------------------------------------------------


def test_closed_discovery_names_the_bridge(swanson_graph):
    chains, diagnostics = closed_discovery(swanson_graph, "Fish Oil", "Raynaud's Disease")

    assert "Blood Viscosity" in [chain.b_name for chain in chains]
    assert diagnostics["already_linked"] is False
    assert diagnostics["shared_papers"] == []


def test_closed_discovery_reports_an_unknown_term(swanson_graph):
    chains, diagnostics = closed_discovery(swanson_graph, "Fish Oil", "Nothing At All")
    assert chains == []
    assert "not in the knowledge graph" in diagnostics["reason"]


def test_rankable_terms_orders_by_connectivity(swanson_graph):
    terms = rankable_terms(swanson_graph)
    assert terms[0]["name"] == "Measurement"
    assert all(term["degree"] > 0 for term in terms)


# --- hypothesis agent --------------------------------------------------------


def test_prompt_contains_the_chain_and_invites_rejection(swanson_graph):
    candidates, _ = open_discovery(swanson_graph, "Fish Oil")
    raynaud = next(c for c in candidates if c.c_name == "Raynaud's Disease").to_dict()

    prompt = build_prompt(raynaud)
    assert "Blood Viscosity" in prompt
    assert "Willing to reject" in prompt or "willing to reject" in prompt.lower()


def test_verdicts_are_normalised():
    assert normalise({"verdict": "PROMISING", "hypothesis": "x"})["verdict"] == "promising"
    assert normalise({"verdict": "nonsense"})["verdict"] == "plausible"
    assert normalise({})["hypothesis"] is None


def test_promising_candidates_are_surfaced_first(swanson_graph):
    from app.services.llm import set_llm
    from tests.conftest import FakeGemini

    class Judge(FakeGemini):
        def __init__(self):
            super().__init__()
            self.calls = []
            self.index = 0

        def generate_json(self, prompt, system_instruction=None, schema=None, **kwargs):
            self.calls.append(prompt)
            self.index += 1
            verdict = "trivial" if self.index == 1 else "promising"
            return {"verdict": verdict, "hypothesis": "h", "reasoning": "r"}

    set_llm(Judge())
    candidates, _ = open_discovery(swanson_graph, "Fish Oil")
    assessed, delta = assess_many([c.to_dict() for c in candidates][:2])

    assert delta["llm_calls"] == len(assessed)
    verdicts = [item["assessment"]["verdict"] for item in assessed]
    assert VERDICT_ORDER[verdicts[0]] <= VERDICT_ORDER[verdicts[-1]]


def test_assessment_failure_is_reported_not_raised(swanson_graph):
    from app.services.llm import set_llm
    from tests.conftest import FakeGemini

    set_llm(FakeGemini(fail=True))
    candidates, _ = open_discovery(swanson_graph, "Fish Oil")
    assessed, delta = assess_many([c.to_dict() for c in candidates][:1])

    assert delta["errors"]
    assert assessed[0]["assessment"] is None


def test_no_candidates_makes_no_calls():
    assessed, delta = assess_many([])
    assert assessed == []
    assert delta["llm_calls"] == 0


# --- HTTP --------------------------------------------------------------------


def seed_graph(graph):
    """Persist the graph *and* the papers it refers to.

    The graph endpoints filter by the caller's papers, because one graph
    database holds every account's entities. A graph whose papers nobody owns is
    correctly invisible, so the fixture's p1..p4 have to exist as rows belonging
    to the signed-in account.
    """
    from app.database import session_scope
    from app.models import Paper, PaperStatus
    from app.services.graph_store import get_graph_store
    from tests.conftest import seed_user_id

    owner = seed_user_id()
    referenced = {
        paper_id
        for item in graph["nodes"] + graph["edges"]
        for paper_id in (item.get("papers") or [])
    }

    with session_scope() as session:
        for paper_id in sorted(referenced):
            if session.get(Paper, paper_id) is None:
                session.add(
                    Paper(
                        id=paper_id,
                        filename=f"{paper_id}.pdf",
                        file_path="",
                        content_hash=paper_id,
                        status=PaperStatus.INDEXED,
                        owner_id=owner,
                    )
                )
        session.commit()

    get_graph_store().persist(graph)


def test_discover_endpoint(client, swanson_graph):
    seed_graph(swanson_graph)
    payload = client.post("/api/lbd", json={"source": "Fish Oil", "mode": "strict"}).json()

    assert payload["source"] == "Fish Oil"
    assert "Raynaud's Disease" in [c["c_name"] for c in payload["candidates"]]
    assert payload["diagnostics"]["entities"] > 0


def test_discover_unknown_term_is_404(client, swanson_graph):
    seed_graph(swanson_graph)
    assert client.post("/api/lbd", json={"source": "Nothing"}).status_code == 404


def test_closed_endpoint(client, swanson_graph):
    seed_graph(swanson_graph)
    payload = client.post(
        "/api/lbd/closed", json={"source": "Fish Oil", "target": "Raynaud's Disease"}
    ).json()

    assert "Blood Viscosity" in [chain["b_name"] for chain in payload["chains"]]


def test_closed_endpoint_requires_a_target(client, swanson_graph):
    seed_graph(swanson_graph)
    assert client.post("/api/lbd/closed", json={"source": "Fish Oil"}).status_code == 400


def test_terms_endpoint(client, swanson_graph):
    seed_graph(swanson_graph)
    terms = client.get("/api/lbd/terms").json()
    assert any(term["name"] == "Measurement" for term in terms)


def test_assessment_saves_hypotheses(client, swanson_graph, fake_llm):
    from app.services.llm import set_llm
    from tests.conftest import FakeGemini

    seed_graph(swanson_graph)
    set_llm(
        FakeGemini(
            {
                "Hypothesis Agent": {
                    "verdict": "promising",
                    "hypothesis": "Dietary fish oil may reduce Raynaud's symptoms.",
                    "reasoning": "Both relate through blood viscosity.",
                    "proposed_test": "A randomised trial measuring digital blood flow.",
                    "novelty": "high",
                }
            }
        )
    )

    payload = client.post(
        "/api/lbd", json={"source": "Fish Oil", "assess": True, "assess_limit": 2}
    ).json()

    assert payload["llm_calls"] >= 1
    assert payload["saved_ids"]

    saved = client.get("/api/lbd/hypotheses").json()
    assert saved
    assert saved[0]["verdict"] == "promising"
    assert "fish oil" in saved[0]["statement"].lower()


def test_hypothesis_star_and_delete(client, swanson_graph, fake_llm):
    from app.services.llm import set_llm
    from tests.conftest import FakeGemini

    seed_graph(swanson_graph)
    set_llm(FakeGemini({"Hypothesis Agent": {"verdict": "plausible", "hypothesis": "h", "reasoning": "r"}}))
    client.post("/api/lbd", json={"source": "Fish Oil", "assess": True, "assess_limit": 1})

    hypothesis_id = client.get("/api/lbd/hypotheses").json()[0]["id"]

    starred = client.post(f"/api/lbd/hypotheses/{hypothesis_id}/star").json()
    assert starred["starred"] is True
    assert len(client.get("/api/lbd/hypotheses", params={"starred": True}).json()) == 1

    assert client.delete(f"/api/lbd/hypotheses/{hypothesis_id}").status_code == 204
    assert client.get("/api/lbd/hypotheses").json() == []
