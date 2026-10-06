import { useState } from 'react';
import api from '../api/client';
import AgentTrace from '../components/AgentTrace';
import HistoryPanel from '../components/HistoryPanel';
import PaperPicker from '../components/PaperPicker';
import VerificationBadge from '../components/VerificationBadge';
import { Badge, BulletList, Card, ErrorBanner, Spinner, WarningBanner, statusTone } from '../components/common';
import { useCorpus } from '../context/CorpusContext';

export default function Gaps() {
  const { effectiveIds, indexedPapers, llmReady } = useCorpus();
  const [focus, setFocus] = useState('');
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
      setResult(await api.gaps({ paper_ids: effectiveIds, focus: focus.trim() || null }));
      setHistoryKey((n) => n + 1);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const gaps = result?.gaps;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Research gaps</h1>
          <p className="page-sub">
            What this literature has <em>not</em> established — and concrete studies that would
            close each gap.
          </p>
        </div>
        <button
          type="button"
          className="btn btn-primary"
          onClick={run}
          disabled={busy || indexedPapers.length === 0}
        >
          {busy ? 'Analysing…' : 'Run Research Gap Agent'}
        </button>
      </header>

      {!llmReady && <WarningBanner>Gemini is not configured — this agent will return an error.</WarningBanner>}
      {indexedPapers.length === 1 && (
        <WarningBanner>
          Gap analysis is far more reliable across multiple papers. Upload at least three related
          papers for useful results.
        </WarningBanner>
      )}
      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      <div className="grid-side">
        <div>
          <Card title="Optional focus">
            <input
              className="input"
              placeholder="e.g. low-resource languages, clinical deployment, evaluation methodology"
              value={focus}
              onChange={(event) => setFocus(event.target.value)}
            />
          </Card>

          {busy && <Spinner label="Reading across the corpus for what is missing…" />}

          {gaps && (
            <>
              <Card title="Landscape" subtitle={`${gaps.analysed_papers} papers analysed`}>
                <p className="lead">{gaps.landscape_summary}</p>
              </Card>

              {(gaps.gaps || []).map((gap, index) => (
                <Card
                  key={index}
                  title={`Gap ${index + 1}: ${gap.title}`}
                  actions={
                    <>
                      <Badge tone={statusTone(gap.severity)}>{gap.severity} severity</Badge>
                      <Badge tone="info">{String(gap.category).replace(/_/g, ' ')}</Badge>
                    </>
                  }
                >
                  <p>{gap.description}</p>

                  {gap.evidence?.length > 0 && (
                    <div className="callout">
                      <strong>Evidence</strong>
                      <BulletList items={gap.evidence} />
                    </div>
                  )}

                  <div className="direction">
                    <strong>Proposed direction</strong>
                    <p>{gap.proposed_direction}</p>
                  </div>

                  {gap.related_papers?.length > 0 && (
                    <div className="chip-row">
                      {gap.related_papers.map((paper) => (
                        <span key={paper} className="chip chip-muted">
                          {paper}
                        </span>
                      ))}
                    </div>
                  )}
                </Card>
              ))}

              <div className="grid-2">
                <Card title="Under-explored intersections">
                  <BulletList items={gaps.underexplored_intersections} />
                </Card>
                <Card title="Open questions">
                  <BulletList items={gaps.open_questions} />
                </Card>
              </div>

              <Card
                title="Fact check"
                subtitle="A gap only matters if the literature really is silent — this checks that"
              >
                <VerificationBadge
                  runId={result.run_id}
                  paperIds={result.paper_ids}
                  source="gap"
                  subject={focus.trim() || 'Research gaps'}
                  label="Verify these gaps"
                />
              </Card>

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
            intents={['gap']}
            refreshKey={historyKey}
            onOpen={(saved) => setResult(saved)}
          />
        </aside>
      </div>
    </div>
  );
}
