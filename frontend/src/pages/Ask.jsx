import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../api/client';
import AgentTrace from '../components/AgentTrace';
import PaperPicker from '../components/PaperPicker';
import { Badge, Card, ErrorBanner, Spinner, WarningBanner, statusTone } from '../components/common';
import { useCorpus } from '../context/CorpusContext';

const SUGGESTIONS = [
  'What datasets are used across these papers?',
  'How do the proposed methods differ from prior work?',
  'What limitations do the authors acknowledge?',
  // Relational: answered by the links between papers, not by any one passage.
  'Which methods were evaluated on the same dataset?',
];

const MODES = [
  ['hybrid', 'Hybrid', 'Passages plus the relationships between them. Best for most questions.'],
  ['vector', 'Passages', 'Text excerpts only — the classic retrieval-augmented answer.'],
  [
    'graph',
    'Relationships',
    'Knowledge graph only. Answers questions about how things relate, which no single passage contains.',
  ],
];

export default function Ask() {
  const { effectiveIds, indexedPapers, llmReady } = useCorpus();
  const [question, setQuestion] = useState('');
  const [mode, setMode] = useState('hybrid');
  const [topK, setTopK] = useState(8);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history.length, busy]);

  const submit = async (event) => {
    event?.preventDefault();
    const text = question.trim();
    if (text.length < 3 || busy) return;

    setBusy(true);
    setError(null);
    try {
      const result = await api.ask({
        question: text,
        paper_ids: effectiveIds,
        top_k: topK,
        mode,
      });
      setHistory((current) => [...current, { question: text, result }]);
      setQuestion('');
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page page-ask">
      <header className="page-header">
        <div>
          <h1>Ask your corpus</h1>
          <p className="page-sub">
            Retrieval-augmented answers grounded in {effectiveIds.length} paper
            {effectiveIds.length === 1 ? '' : 's'}, with citations back to the source pages.
          </p>
        </div>
      </header>

      {!llmReady && (
        <WarningBanner>
          Gemini is not configured, so the QA agent cannot answer. Semantic search still works —
          the retrieved excerpts below each answer come from ChromaDB.
        </WarningBanner>
      )}

      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      <div className="ask-layout">
        <div className="ask-main">
          {history.length === 0 && (
            <Card title="Try a question">
              <div className="suggestion-grid">
                {SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    className="suggestion"
                    onClick={() => setQuestion(suggestion)}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </Card>
          )}

          {history.map((entry, index) => (
            <div key={index} className="qa-entry">
              <div className="qa-question">
                <span className="qa-avatar">You</span>
                <p>{entry.question}</p>
              </div>

              <Card
                title="Question Answering Agent"
                actions={
                  entry.result.answer?.confidence && (
                    <Badge tone={statusTone(entry.result.answer.confidence)}>
                      {entry.result.answer.confidence} confidence
                    </Badge>
                  )
                }
              >
                {entry.result.answer ? (
                  <>
                    <p className="answer-text">{entry.result.answer.answer}</p>

                    {entry.result.answer.caveats?.length > 0 && (
                      <div className="callout">
                        <strong>Caveats</strong>
                        <ul>
                          {entry.result.answer.caveats.map((caveat, i) => (
                            <li key={i}>{caveat}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {entry.result.answer.sources?.length > 0 && (
                      <div className="sources">
                        <h4>Cited sources</h4>
                        {entry.result.answer.sources.map((source) => (
                          <div key={source.marker + source.chunk_id} className="source">
                            <div className="source-head">
                              <Badge tone="info">{source.marker}</Badge>
                              <strong>{source.paper_title}</strong>
                              {source.kind === 'figure' && (
                                <Badge tone="success">{source.label || 'Figure'}</Badge>
                              )}
                              <span className="muted">
                                {source.page ? `p.${source.page}` : ''}
                                {source.section ? ` · ${source.section}` : ''} · score{' '}
                                {source.score}
                              </span>
                            </div>
                            {source.kind === 'figure' && source.figure_id && (
                              <img
                                className="source-figure"
                                src={api.figureImageUrl(source.figure_id)}
                                alt={source.label || 'Cited figure'}
                                loading="lazy"
                              />
                            )}
                            <p className="source-excerpt">{source.excerpt}…</p>
                            {source.paper_id && (
                              <Link
                                className="btn btn-ghost btn-sm"
                                to={
                                  `/reader/${source.paper_id}?` +
                                  (source.kind === 'figure' && source.figure_id
                                    ? `figure=${source.figure_id}`
                                    : `chunk=${source.chunk_id}`)
                                }
                              >
                                Open in paper →
                              </Link>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                    {entry.result.answer.follow_up_questions?.length > 0 && (
                      <div className="followups">
                        <h4>Follow-up questions</h4>
                        <div className="suggestion-grid">
                          {entry.result.answer.follow_up_questions.map((followUp) => (
                            <button
                              key={followUp}
                              type="button"
                              className="suggestion"
                              onClick={() => setQuestion(followUp)}
                            >
                              {followUp}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <p className="error-text">
                    The agent produced no answer. {entry.result.errors?.join(' ')}
                  </p>
                )}

                {entry.result.answer?.graph_sources?.length > 0 && (
                  <div className="sources">
                    <h4>Cited relationships</h4>
                    {entry.result.answer.graph_sources.map((fact) => (
                      <div key={fact.marker} className="source">
                        <div className="source-head">
                          <Badge tone="success">{fact.marker}</Badge>
                          <strong>{fact.sentence}</strong>
                        </div>
                        <p className="source-excerpt">
                          From {fact.paper_titles?.join(', ') || 'the knowledge graph'}
                          {fact.evidence ? ` — “${fact.evidence}”` : ''}
                        </p>
                      </div>
                    ))}
                  </div>
                )}

                {entry.result.graph_facts?.length > 0 && (
                  <details className="retrieved">
                    <summary>
                      {entry.result.graph_facts.length} relationships from the graph
                      {entry.result.graph_matches?.length > 0 &&
                        ` · matched ${entry.result.graph_matches.map((m) => m.name).join(', ')}`}
                    </summary>
                    <ul>
                      {entry.result.graph_facts.slice(0, 20).map((fact, i) => (
                        <li key={i}>
                          <span className="muted">[G{i + 1}]</span> {fact.sentence}
                          {fact.paper_titles?.length > 0 && (
                            <div className="cell-sub">{fact.paper_titles.join(', ')}</div>
                          )}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}

                {entry.result.retrieved?.length > 0 && (
                  <details className="retrieved">
                    <summary>{entry.result.retrieved.length} retrieved chunks</summary>
                    <ul>
                      {entry.result.retrieved.map((chunk, i) => (
                        <li key={chunk.chunk_id}>
                          <span className="muted">
                            [S{i + 1}] {chunk.paper_title} · p.{chunk.page_start ?? '?'} · score{' '}
                            {chunk.score}
                            {chunk.source === 'ocr' && (
                              <span className="ocr-flag" title="Recovered by OCR — may contain recognition errors">
                                ⌾ OCR
                              </span>
                            )}
                          </span>
                          <p>{chunk.text.slice(0, 300)}…</p>
                        </li>
                      ))}
                    </ul>
                  </details>
                )}

                <AgentTrace
                  trace={entry.result.trace}
                  errors={entry.result.errors}
                  llmCalls={entry.result.llm_calls}
                  durationMs={entry.result.duration_ms}
                />
              </Card>
            </div>
          ))}

          {busy && <Spinner label="Retrieving and reasoning…" />}
          <div ref={bottomRef} />

          <form className="ask-form" onSubmit={submit}>
            <textarea
              className="input textarea"
              rows={3}
              placeholder={
                indexedPapers.length
                  ? 'Ask anything about the selected papers…'
                  : 'Upload papers before asking questions.'
              }
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) submit(event);
              }}
              disabled={indexedPapers.length === 0}
            />
            <div className="ask-controls">
              <label className="field-inline">
                Using
                <select
                  className="input input-select"
                  value={mode}
                  onChange={(event) => setMode(event.target.value)}
                  title={MODES.find(([key]) => key === mode)?.[2]}
                >
                  {MODES.map(([key, label]) => (
                    <option key={key} value={key}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field-inline">
                Top-k
                <input
                  type="number"
                  className="input input-number"
                  min={1}
                  max={30}
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                />
              </label>
              <span className="muted">Ctrl/⌘ + Enter to send</span>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={busy || question.trim().length < 3}
              >
                Ask
              </button>
            </div>
          </form>
        </div>

        <aside className="ask-side">
          <Card title="Corpus scope">
            <PaperPicker compact />
          </Card>
        </aside>
      </div>
    </div>
  );
}
