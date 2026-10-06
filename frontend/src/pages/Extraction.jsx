import { useState } from 'react';
import api from '../api/client';
import AgentTrace from '../components/AgentTrace';
import HistoryPanel from '../components/HistoryPanel';
import PaperPicker from '../components/PaperPicker';
import { Card, ErrorBanner, Spinner, WarningBanner } from '../components/common';
import { useCorpus } from '../context/CorpusContext';

const TABS = [
  ['comparison', 'Comparison'],
  ['datasets', 'Datasets'],
  ['methods', 'Methods'],
  ['metrics', 'Metrics'],
  ['tasks', 'Tasks'],
];

export default function Extraction() {
  const { effectiveIds, indexedPapers, llmReady } = useCorpus();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  // Bumped after a run so the history list picks up the new entry.
  const [historyKey, setHistoryKey] = useState(0);
  const [tab, setTab] = useState('comparison');

  const run = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.extract({ paper_ids: effectiveIds }));
      setHistoryKey((n) => n + 1);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const extraction = result?.extraction;
  const aggregate = extraction?.aggregate;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Information extraction</h1>
          <p className="page-sub">
            Datasets, methods, metrics and tasks pulled out of each paper, then rolled up across the
            corpus.
          </p>
        </div>
        <button
          type="button"
          className="btn btn-primary"
          onClick={run}
          disabled={busy || indexedPapers.length === 0}
        >
          {busy ? 'Extracting…' : 'Run Extraction Agent'}
        </button>
      </header>

      {!llmReady && <WarningBanner>Gemini is not configured — this agent will return an error.</WarningBanner>}
      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      <div className="grid-side">
        <div>
          {busy && <Spinner label="Extracting entities, one call per paper…" />}

          {extraction && (
            <>
              <div className="tabs">
                {TABS.map(([key, label]) => (
                  <button
                    key={key}
                    type="button"
                    className={`tab ${tab === key ? 'is-active' : ''}`}
                    onClick={() => setTab(key)}
                  >
                    {label}
                    {key !== 'comparison' && aggregate?.[key] && (
                      <span className="tab-count">{aggregate[key].length}</span>
                    )}
                  </button>
                ))}
              </div>

              {tab === 'comparison' && (
                <Card title="Per-paper comparison">
                  <div className="table-scroll">
                    <table className="table">
                      <thead>
                        <tr>
                          <th>Paper</th>
                          <th>Tasks</th>
                          <th>Methods</th>
                          <th>Datasets</th>
                          <th>Metrics</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(extraction.papers || []).map((paper) => (
                          <tr key={paper.paper_id}>
                            <td>
                              <strong>{paper.paper_title}</strong>
                            </td>
                            <td>{paper.tasks?.join(', ') || '—'}</td>
                            <td>{paper.methods?.map((m) => m.name).join(', ') || '—'}</td>
                            <td>{paper.datasets?.map((d) => d.name).join(', ') || '—'}</td>
                            <td>
                              {paper.metrics?.length
                                ? paper.metrics
                                    .slice(0, 4)
                                    .map((m) => `${m.name}${m.value ? `: ${m.value}` : ''}`)
                                    .join(' · ')
                                : '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              )}

              {tab !== 'comparison' && (
                <Card title={TABS.find(([key]) => key === tab)?.[1]}>
                  {(aggregate?.[tab] || []).length === 0 ? (
                    <p className="muted">Nothing extracted in this category.</p>
                  ) : (
                    <div className="table-scroll">
                      <table className="table">
                        <thead>
                          <tr>
                            <th>Name</th>
                            <th>Papers</th>
                            <th>Appears in</th>
                            {tab !== 'tasks' && <th>Details</th>}
                          </tr>
                        </thead>
                        <tbody>
                          {aggregate[tab].map((entry) => (
                            <tr key={entry.name}>
                              <td>
                                <strong>{entry.name}</strong>
                              </td>
                              <td>{entry.papers?.length ?? 0}</td>
                              <td className="cell-sub">{entry.papers?.join(', ')}</td>
                              {tab !== 'tasks' && (
                                <td className="cell-sub">
                                  {(entry.details || [])
                                    .slice(0, 3)
                                    .map((detail, index) => (
                                      <div key={index}>
                                        {Object.entries(detail)
                                          .filter(([key]) => key !== 'paper')
                                          .map(([key, value]) => `${key}: ${value}`)
                                          .join(' · ')}
                                      </div>
                                    ))}
                                </td>
                              )}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </Card>
              )}

              {(extraction.papers || []).map((paper) => (
                <Card key={paper.paper_id} title={paper.paper_title} subtitle="Extracted detail">
                  <div className="grid-2">
                    <div>
                      <h4>Research questions</h4>
                      <ul className="bullets">
                        {(paper.research_questions || []).map((question, index) => (
                          <li key={index}>{question}</li>
                        ))}
                        {(paper.research_questions || []).length === 0 && (
                          <li className="muted">None extracted.</li>
                        )}
                      </ul>
                    </div>
                    <div>
                      <h4>Limitations</h4>
                      <ul className="bullets">
                        {(paper.limitations || []).map((limitation, index) => (
                          <li key={index}>{limitation}</li>
                        ))}
                        {(paper.limitations || []).length === 0 && (
                          <li className="muted">None extracted.</li>
                        )}
                      </ul>
                    </div>
                  </div>
                  {paper.tools?.length > 0 && (
                    <div className="chip-row">
                      {paper.tools.map((tool) => (
                        <span key={tool} className="chip">
                          {tool}
                        </span>
                      ))}
                    </div>
                  )}
                </Card>
              ))}

              <AgentTrace
                trace={result.trace}
                errors={result.errors}
                llmCalls={result.llm_calls}
                durationMs={result.duration_ms}
              />
            </>
          )}
        </div>

        <aside>
          <Card title="Corpus scope">
            <PaperPicker compact />
          </Card>

          <HistoryPanel
            intents={['extract']}
            refreshKey={historyKey}
            onOpen={(saved) => setResult(saved)}
          />
        </aside>
      </div>
    </div>
  );
}
