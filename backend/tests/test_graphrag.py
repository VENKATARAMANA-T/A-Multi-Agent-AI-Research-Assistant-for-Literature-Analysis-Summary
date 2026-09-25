"""GraphRAG — answering from relationships rather than from passages.

The fixture graph encodes a question no single chunk can answer: two methods
evaluated on the same dataset, where no paper states that they share it. Vector
search cannot surface that; traversal can.
"""

from __future__ import annotations

import pytest

from app.services.graph_retrieval import (
    EMBEDDING_FLOOR,
    candidate_terms,
    expand,
    format_facts,
    link_entities,
    retrieve,
)


def node(node_id, name, kind, papers=None, **extra):
    return {"id": node_id, "name": name, "type": kind, "papers": papers or [], **extra}


def edge(source, target, relation, papers=None, evidence=""):
    return {
        "source": source,
        "target": target,
        "type": relation,
        "papers": papers or [],
        "evidence": evidence,
    }


@pytest.fixture
def speech_graph():
    """Two methods share a dataset, but no paper says so."""
    return {
        "nodes": [
            node("paper:p1", "Wav2vec Dysarthria Study", "Paper", paper_id="p1"),
            node("paper:p2", "BiLSTM Stuttering Study", "Paper", paper_id="p2"),
            node("e:wav2vec", "wav2vec", "Method", ["p1"]),
            node("e:bilstm", "BiLSTM", "Method", ["p2"]),
            node("e:uaspeech", "UA-Speech", "Dataset", ["p1", "p2"]),
            node("e:dysarthria", "Dysarthria", "Concept", ["p1"]),
            node("e:author", "Ada Lovelace", "Author", ["p1"]),
            # An unmerged alias of wav2vec, as real extraction produces.
            node("e:wav2vec long", "Wave To Vector (wav2vec)", "Method", ["p1"]),
        ],
        "edges": [
            edge("e:wav2vec", "e:uaspeech", "EVALUATED_ON", ["p1"], "trained on UA-Speech"),
            edge("e:bilstm", "e:uaspeech", "EVALUATED_ON", ["p2"]),
            edge("e:wav2vec", "e:dysarthria", "APPLIED_TO", ["p1"]),
            edge("e:wav2vec long", "e:uaspeech", "EVALUATED_ON", ["p1"]),
            edge("paper:p1", "e:wav2vec", "MENTIONS", ["p1"]),
            edge("paper:p1", "e:author", "AUTHORED_BY", ["p1"]),
        ],
    }


# --- term extraction ---------------------------------------------------------


def test_longer_phrases_come_first():
    terms = candidate_terms("Which methods use wav2vec on UA-Speech?")
    assert len(terms[0].split()) >= len(terms[-1].split())


def test_stopwords_alone_are_not_terms():
    assert "the" not in [t.lower() for t in candidate_terms("the model")]


def test_empty_question_yields_nothing():
    assert candidate_terms("") == []


# --- entity linking ----------------------------------------------------------


def test_exact_name_is_linked(speech_graph):
    entities = [n for n in speech_graph["nodes"] if n["type"] not in ("Paper", "Author")]
    matches = link_entities("How does wav2vec perform?", entities, use_embeddings=False)

    assert "wav2vec" in [m.name for m in matches]
    assert matches[0].method in ("exact", "alias")


def test_an_acronym_links_to_its_expansion(speech_graph):
    """A paper writes the term in full once and uses the acronym after."""
    entities = [n for n in speech_graph["nodes"] if n["type"] not in ("Paper", "Author")]
    matches = link_entities("Tell me about Wave To Vector", entities, use_embeddings=False)

    assert matches, "the expanded form did not link"


def test_a_question_about_nothing_links_nothing(speech_graph):
    entities = [n for n in speech_graph["nodes"] if n["type"] not in ("Paper", "Author")]
    matches = link_entities("What is the capital of France?", entities, use_embeddings=False)

    assert matches == []


def test_seed_count_is_capped(speech_graph):
    entities = [n for n in speech_graph["nodes"] if n["type"] not in ("Paper", "Author")]
    matches = link_entities(
        "wav2vec BiLSTM UA-Speech Dysarthria", entities, max_seeds=2, use_embeddings=False
    )
    assert len(matches) <= 2


def test_embedding_floor_is_conservative():
    """A wrongly linked entity produces confidently wrong context."""
    assert EMBEDDING_FLOOR >= 0.5


# --- subgraph expansion ------------------------------------------------------


def test_expansion_finds_the_shared_dataset(speech_graph):
    """The question no chunk answers: what else used this dataset?"""
    facts, _ = expand(speech_graph, ["e:wav2vec"], hops=2)
    sentences = [fact.sentence() for fact in facts]

    assert any("wav2vec" in s and "UA-Speech" in s for s in sentences)
    assert any("BiLSTM" in s and "UA-Speech" in s for s in sentences)


def test_structural_edges_are_excluded(speech_graph):
    """"Paper mentions entity" and authorship are bookkeeping, not findings."""
    facts, _ = expand(speech_graph, ["e:wav2vec"], hops=2)
    relations = {fact.relation for fact in facts}

    assert "MENTIONS" not in relations
    assert "AUTHORED_BY" not in relations
    assert "Ada Lovelace" not in " ".join(fact.sentence() for fact in facts)


def test_alias_duplicates_collapse_to_one_fact(speech_graph):
    """The same concept under two names emitted the identical fact twice."""
    facts, _ = expand(speech_graph, ["e:wav2vec"], hops=1)
    evaluated = [f for f in facts if f.relation == "EVALUATED_ON" and "UA-Speech" in f.target]

    assert len(evaluated) == 1, [f.sentence() for f in evaluated]


def test_facts_carry_their_papers(speech_graph):
    facts, _ = expand(speech_graph, ["e:wav2vec"], hops=1)
    fact = next(f for f in facts if f.relation == "EVALUATED_ON")

    assert fact.papers
    assert fact.paper_titles  # resolved to readable titles for citation


def test_hop_limit_is_respected(speech_graph):
    one_hop, _ = expand(speech_graph, ["e:dysarthria"], hops=1)
    two_hop, _ = expand(speech_graph, ["e:dysarthria"], hops=2)

    assert len(two_hop) > len(one_hop)


def test_fact_cap_is_respected(speech_graph):
    facts, _ = expand(speech_graph, ["e:wav2vec"], hops=3, max_facts=2)
    assert len(facts) <= 2


# --- formatting --------------------------------------------------------------


def test_facts_are_numbered_for_citation(speech_graph):
    facts, _ = expand(speech_graph, ["e:wav2vec"], hops=1)
    text = format_facts(facts)

    assert text.startswith("[G1]")
    assert "from:" in text


def test_formatting_respects_a_character_budget(speech_graph):
    facts, _ = expand(speech_graph, ["e:wav2vec"], hops=2)
    assert len(format_facts(facts, max_chars=80)) <= 200


# --- retrieve ----------------------------------------------------------------


def test_retrieve_returns_context_for_a_relational_question(speech_graph):
    context = retrieve("What was wav2vec evaluated on?", graph=speech_graph)

    assert context.found
    assert context.matches
    assert context.paper_ids


def test_retrieve_explains_an_empty_graph():
    context = retrieve("anything", graph={"nodes": [], "edges": []})

    assert context.found is False
    assert "empty" in context.reason


def test_retrieve_explains_an_unmatched_question(speech_graph):
    context = retrieve("What is the capital of France?", graph=speech_graph)

    assert context.found is False
    assert "did not match" in context.reason or "No entity" in context.reason


def test_retrieve_reports_an_isolated_entity():
    graph = {"nodes": [node("e:lonely", "Lonely Concept", "Concept", ["p1"])], "edges": []}
    context = retrieve("Tell me about Lonely Concept", graph=graph)

    assert context.found is False
    assert context.matches
    assert "no relationships" in context.reason


# --- the agent pipeline ------------------------------------------------------


def seed_graph(graph):
    from app.services.graph_store import get_graph_store

    get_graph_store().persist(graph)


def ingest(path) -> str:
    from app.database import session_scope
    from app.services.ingestion import process_paper, store_upload

    with session_scope() as session:
        paper, _ = store_upload(session, path.name, path.read_bytes())
        process_paper(session, paper.id)
        return paper.id


def test_hybrid_uses_both_passages_and_relationships(sample_pdf, speech_graph, fake_llm):
    from app.agents.workflow import run_workflow

    ingest(sample_pdf)
    seed_graph(speech_graph)

    result = run_workflow("qa", question="What was wav2vec evaluated on?", retrieval_mode="hybrid")

    assert result["retrieved"], "vector retrieval produced nothing"
    assert result["graph_facts"], "graph retrieval produced nothing"

    prompt = fake_llm.calls[-1]["prompt"]
    assert "Relationships from the knowledge graph" in prompt
    assert "[G1]" in prompt


def test_vector_mode_skips_the_graph(sample_pdf, speech_graph, fake_llm):
    from app.agents.workflow import run_workflow

    ingest(sample_pdf)
    seed_graph(speech_graph)

    result = run_workflow("qa", question="What was wav2vec evaluated on?", retrieval_mode="vector")

    assert result["retrieved"]
    assert result["graph_facts"] == []
    assert "Relationships from the knowledge graph" not in fake_llm.calls[-1]["prompt"]


def test_graph_mode_skips_vector_search(sample_pdf, speech_graph, fake_llm):
    from app.agents.workflow import run_workflow

    ingest(sample_pdf)
    seed_graph(speech_graph)

    result = run_workflow("qa", question="What was wav2vec evaluated on?", retrieval_mode="graph")

    assert result["retrieved"] == []
    assert result["graph_facts"]
    # The answer still has context, so the agent must run rather than skip.
    assert result["answer"] is not None


def test_an_unmatched_question_falls_back_to_text(sample_pdf, fake_llm):
    """No entity matched must degrade to vector search, not return nothing."""
    from app.agents.workflow import run_workflow

    ingest(sample_pdf)
    seed_graph({"nodes": [], "edges": []})

    result = run_workflow("qa", question="What does the paper say about sparsity?")

    assert result["graph_facts"] == []
    assert result["retrieved"], "fell back to nothing instead of vector search"
    assert result["status"] == "completed"

    trace = {event["node"]: event for event in result["trace"]}
    assert trace["graph_retrieval"]["status"] == "empty"


def test_graph_markers_resolve_to_facts():
    from app.agents.qa import attach_sources

    facts = [
        {
            "sentence": "wav2vec evaluated on UA-Speech",
            "relation": "EVALUATED_ON",
            "source": "wav2vec",
            "target": "UA-Speech",
            "papers": ["p1"],
            "paper_titles": ["A Paper"],
            "evidence": "trained on UA-Speech",
        }
    ]
    answer = attach_sources({"supporting_sources": ["G1"]}, retrieved=[], graph_facts=facts)

    assert answer["graph_sources"][0]["marker"] == "G1"
    assert answer["graph_sources"][0]["target"] == "UA-Speech"
    assert answer["sources"] == []


def test_an_unresolvable_marker_is_dropped():
    """A citation the reader cannot follow is worse than no citation."""
    from app.agents.qa import attach_sources

    answer = attach_sources({"supporting_sources": ["G9", "S9"]}, retrieved=[], graph_facts=[])

    assert answer["graph_sources"] == []
    assert answer["sources"] == []


# --- HTTP --------------------------------------------------------------------


def upload(client, path):
    return client.post(
        "/api/papers/upload?wait=true",
        files=[("files", (path.name, path.read_bytes(), "application/pdf"))],
    ).json()["results"][0]["paper_id"]


def relabel(graph: dict, paper_id: str) -> dict:
    """Point a fixture graph at a real paper.

    The ask endpoint scopes the graph to the selected papers, so a graph whose
    entities belong to papers outside the corpus is correctly filtered away.
    """
    return {
        "nodes": [{**n, "papers": [paper_id] if n.get("papers") else n.get("papers", [])}
                  for n in graph["nodes"]],
        "edges": [{**e, "papers": [paper_id]} for e in graph["edges"]],
    }


def test_ask_accepts_a_retrieval_mode(client, sample_pdf, speech_graph, fake_llm):
    paper_id = upload(client, sample_pdf)
    seed_graph(relabel(speech_graph, paper_id))

    payload = client.post(
        "/api/agents/ask",
        json={"question": "What was wav2vec evaluated on?", "mode": "hybrid"},
    ).json()

    assert payload["graph_facts"]
    assert payload["graph_matches"]


def test_graph_context_is_scoped_to_the_selected_papers(client, sample_pdf, second_pdf, speech_graph, fake_llm):
    """Narrowing the corpus must narrow the graph too, or scope means nothing."""
    first = upload(client, sample_pdf)
    second = upload(client, second_pdf)
    seed_graph(relabel(speech_graph, first))

    included = client.post(
        "/api/agents/ask",
        json={"question": "What was wav2vec evaluated on?", "paper_ids": [first]},
    ).json()
    excluded = client.post(
        "/api/agents/ask",
        json={"question": "What was wav2vec evaluated on?", "paper_ids": [second]},
    ).json()

    assert included["graph_facts"]
    assert excluded["graph_facts"] == []


def test_invalid_mode_is_rejected(client, sample_pdf, fake_llm):
    upload(client, sample_pdf)
    response = client.post(
        "/api/agents/ask", json={"question": "Anything?", "mode": "telepathy"}
    )
    assert response.status_code == 422
