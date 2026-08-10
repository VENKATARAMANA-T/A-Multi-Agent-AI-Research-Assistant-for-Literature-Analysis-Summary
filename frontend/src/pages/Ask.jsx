import { useEffect, useRef, useState } from 'react';
import api from '../api/client';
import AgentTrace from '../components/AgentTrace';
import PaperPicker from '../components/PaperPicker';
import { Badge, Card, ErrorBanner, Spinner, WarningBanner, statusTone } from '../components/common';
import { useCorpus } from '../context/CorpusContext';

const SUGGESTIONS = [
  'What datasets are used across these papers?',
  'How do the proposed methods differ from prior work?',
  'What limitations do the authors acknowledge?',
  'Which evaluation metrics are reported, and on what benchmarks?',
];

export default function Ask() {
  const { effectiveIds, indexedPapers, llmReady } = useCorpus();
  const [question, setQuestion] = useState('');
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
      const result = await api.ask({ question: text, paper_ids: effectiveIds, top_k: topK });
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
                              <span className="muted">
                                {source.page ? `p.${source.page}` : ''}
                                {source.section ? ` · ${source.section}` : ''} · score{' '}
                                {source.score}
                              </span>
                            </div>
                            <p className="source-excerpt">{source.excerpt}…</p>
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

                {entry.result.retrieved?.length > 0 && (
                  <details className="retrieved">
                    <summary>{entry.result.retrieved.length} retrieved chunks</summary>
                    <ul>
                      {entry.result.retrieved.map((chunk, i) => (
                        <li key={chunk.chunk_id}>
                          <span className="muted">
                            [S{i + 1}] {chunk.paper_title} · p.{chunk.page_start ?? '?'} · score{' '}
                            {chunk.score}
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
