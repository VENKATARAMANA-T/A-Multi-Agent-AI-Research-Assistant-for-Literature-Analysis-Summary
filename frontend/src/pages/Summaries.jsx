import { useState } from 'react';
import api from '../api/client';
import AgentTrace from '../components/AgentTrace';
import HistoryPanel from '../components/HistoryPanel';
import PaperPicker from '../components/PaperPicker';
import VerificationBadge from '../components/VerificationBadge';
import { BulletList, Card, ErrorBanner, Spinner, WarningBanner } from '../components/common';
import { useCorpus } from '../context/CorpusContext';

export default function Summaries() {
  const { effectiveIds, selectedIds, indexedPapers, llmReady } = useCorpus();
  const [scope, setScope] = useState('multi');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  // Bumped after a run so the history list picks up the new entry.
  const [historyKey, setHistoryKey] = useState(0);

  const run = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const paperIds = scope === 'single' ? selectedIds.slice(0, 1) : effectiveIds;
      if (scope === 'single' && paperIds.length !== 1) {
        throw new Error('Select exactly one paper for a single-paper summary.');
      }
      setResult(await api.summarize({ paper_ids: paperIds, scope }));
      setHistoryKey((n) => n + 1);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const summary = result?.summary;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Summaries</h1>
          <p className="page-sub">
            Structured single-paper summaries, or a comparative synthesis across the corpus.
          </p>
        </div>
        <button
          type="button"
          className="btn btn-primary"
          onClick={run}
          disabled={busy || indexedPapers.length === 0}
        >
          {busy ? 'Summarising…' : 'Run Summarization Agent'}
        </button>
      </header>

      {!llmReady && <WarningBanner>Gemini is not configured — this agent will return an error.</WarningBanner>}
      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      <div className="grid-side">
        <div>
          <Card title="Mode">
            <div className="radio-row">
              <label className={`radio-card ${scope === 'multi' ? 'is-active' : ''}`}>
                <input
                  type="radio"
                  name="scope"
                  checked={scope === 'multi'}
                  onChange={() => setScope('multi')}
                />
                <span>
                  <strong>Comparative synthesis</strong>
                  <small>Themes, agreements, contradictions and trends across all selected papers.</small>
                </span>
              </label>
              <label className={`radio-card ${scope === 'single' ? 'is-active' : ''}`}>
                <input
                  type="radio"
                  name="scope"
                  checked={scope === 'single'}
                  onChange={() => setScope('single')}
                />
                <span>
                  <strong>Single paper</strong>
                  <small>Problem, approach, findings, contributions, limitations, future work.</small>
                </span>
              </label>
            </div>
          </Card>

          {busy && <Spinner label="Reading the papers…" />}

          {summary && summary.scope === 'single' && (
            <Card title={summary.paper_title} subtitle="Single-paper summary">
              <p className="lead">{summary.tldr}</p>
              <div className="grid-2">
                <div>
                  <h4>Problem</h4>
                  <p>{summary.problem}</p>
                  <h4>Approach</h4>
                  <p>{summary.approach}</p>
                </div>
                <div>
                  <h4>Key findings</h4>
                  <BulletList items={summary.key_findings} />
                  <h4>Contributions</h4>
                  <BulletList items={summary.contributions} />
                </div>
              </div>
              <div className="grid-2">
                <div>
                  <h4>Limitations</h4>
                  <BulletList items={summary.limitations} />
                </div>
                <div>
                  <h4>Future work</h4>
                  <BulletList items={summary.future_work} />
                </div>
              </div>
              {summary.keywords?.length > 0 && (
                <div className="chip-row">
                  {summary.keywords.map((keyword) => (
                    <span key={keyword} className="chip">
                      {keyword}
                    </span>
                  ))}
                </div>
              )}
            </Card>
          )}

          {summary && summary.scope === 'multi' && (
            <>
              <Card title="Corpus overview" subtitle={`${summary.paper_ids?.length ?? 0} papers synthesised`}>
                <p className="lead">{summary.overview}</p>
              </Card>

              <Card title="Themes">
                {(summary.themes || []).map((theme, index) => (
                  <div key={index} className="theme">
                    <h4>{theme.name}</h4>
                    <p>{theme.description}</p>
                    {theme.papers?.length > 0 && (
                      <div className="chip-row">
                        {theme.papers.map((paper) => (
                          <span key={paper} className="chip chip-muted">
                            {paper}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </Card>

              <div className="grid-2">
                <Card title="Agreements">
                  <BulletList items={summary.agreements} />
                </Card>
                <Card title="Disagreements">
                  <BulletList items={summary.disagreements} />
                </Card>
              </div>

              <div className="grid-2">
                <Card title="Methodological trends">
                  <BulletList items={summary.methodological_trends} />
                </Card>
                <Card title="Shared datasets">
                  <BulletList items={summary.shared_datasets} />
                </Card>
              </div>
            </>
          )}

          {summary && (
            <Card title="Fact check" subtitle="Each claim re-checked against evidence retrieved for it">
              <VerificationBadge
                runId={result.run_id}
                paperIds={result.paper_ids}
                source={scope === 'single' ? 'summarize' : 'multi_summarize'}
                subject={summary.paper_title || 'Corpus synthesis'}
                label="Verify this summary"
              />
            </Card>
          )}

          {result && (
            <AgentTrace
              trace={result.trace}
              errors={result.errors}
              llmCalls={result.llm_calls}
              durationMs={result.duration_ms}
            />
          )}
        </div>

        <aside>
          <Card title="Corpus scope">
            <PaperPicker mode={scope === 'single' ? 'single' : 'multi'} compact />
          </Card>

          <HistoryPanel
            intents={['summarize', 'multi_summarize']}
            refreshKey={historyKey}
            onOpen={(saved) => setResult(saved)}
          />
        </aside>
      </div>
    </div>
  );
}
