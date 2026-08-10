import { useCallback, useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import api from '../api/client';
import PaperPicker from '../components/PaperPicker';
import { Card, EmptyState, ErrorBanner, Spinner, WarningBanner } from '../components/common';
import { useCorpus } from '../context/CorpusContext';

export default function Reports() {
  const { effectiveIds, indexedPapers, llmReady } = useCorpus();
  const [reports, setReports] = useState([]);
  const [active, setActive] = useState(null);
  const [title, setTitle] = useState('');
  const [focus, setFocus] = useState('');
  const [includeNarrative, setIncludeNarrative] = useState(true);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setReports(await api.listReports());
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const generate = async () => {
    setBusy(true);
    setError(null);
    try {
      const report = await api.createReport({
        paper_ids: effectiveIds,
        title: title.trim() || null,
        focus: focus.trim() || null,
        include_narrative: includeNarrative,
        include_graph: true,
      });
      setActive(report);
      await load();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const open = async (id) => {
    try {
      setActive(await api.getReport(id));
    } catch (err) {
      setError(err.message);
    }
  };

  const remove = async (id) => {
    if (!window.confirm('Delete this report?')) return;
    try {
      await api.deleteReport(id);
      if (active?.id === id) setActive(null);
      await load();
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Reports</h1>
          <p className="page-sub">
            A full literature review: corpus table, synthesis, comparison tables, research gaps and
            references — exportable as Markdown or PDF.
          </p>
        </div>
        <button
          type="button"
          className="btn btn-primary"
          onClick={generate}
          disabled={busy || indexedPapers.length === 0}
        >
          {busy ? 'Generating…' : 'Generate report'}
        </button>
      </header>

      {!llmReady && (
        <WarningBanner>
          Gemini is not configured. A report will still be produced, but the synthesis, comparison
          and gap sections will be empty.
        </WarningBanner>
      )}
      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      <div className="grid-side">
        <div>
          {busy && (
            <Spinner label="Running every agent, then rendering Markdown and PDF — this takes a while…" />
          )}

          {active?.agent_errors?.length > 0 && (
            <WarningBanner>
              <strong>This report is incomplete.</strong> Some agents did not finish, so the
              sections they populate are empty:
              <ul>
                {active.agent_errors.map((message, index) => (
                  <li key={index}>{message}</li>
                ))}
              </ul>
            </WarningBanner>
          )}

          {active ? (
            <Card
              title={active.title}
              subtitle={`${active.paper_ids.length} papers · created ${new Date(
                active.created_at,
              ).toLocaleString()}`}
              actions={
                <div className="row-actions">
                  <a className="btn btn-ghost btn-sm" href={api.reportMarkdownUrl(active.id)}>
                    Markdown
                  </a>
                  <a
                    className="btn btn-primary btn-sm"
                    href={api.reportPdfUrl(active.id)}
                    target="_blank"
                    rel="noreferrer"
                  >
                    PDF
                  </a>
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm"
                    onClick={() => setActive(null)}
                  >
                    Close
                  </button>
                </div>
              }
            >
              <div className="markdown">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{active.markdown}</ReactMarkdown>
              </div>
            </Card>
          ) : (
            <Card title="Report options">
              <label className="field">
                Title
                <input
                  className="input"
                  placeholder="Literature Review: Transformer Efficiency"
                  value={title}
                  onChange={(event) => setTitle(event.target.value)}
                />
              </label>
              <label className="field">
                Focus (optional)
                <input
                  className="input"
                  placeholder="e.g. evaluation methodology, deployment constraints"
                  value={focus}
                  onChange={(event) => setFocus(event.target.value)}
                />
              </label>
              <label className="field-check">
                <input
                  type="checkbox"
                  checked={includeNarrative}
                  onChange={(event) => setIncludeNarrative(event.target.checked)}
                />
                Generate the prose synthesis with Gemini (slower, much richer)
              </label>
            </Card>
          )}
        </div>

        <aside>
          <Card title="Corpus scope">
            <PaperPicker compact />
          </Card>

          <Card title="Saved reports">
            {loading ? (
              <Spinner label="Loading…" />
            ) : reports.length === 0 ? (
              <EmptyState title="No reports yet" description="Generate one from your corpus." />
            ) : (
              <ul className="report-list">
                {reports.map((report) => (
                  <li key={report.id}>
                    <button type="button" className="report-item" onClick={() => open(report.id)}>
                      <strong>{report.title}</strong>
                      <span className="muted">
                        {report.paper_ids.length} papers ·{' '}
                        {new Date(report.created_at).toLocaleDateString()}
                        {report.has_pdf ? ' · PDF' : ''}
                      </span>
                    </button>
                    <button
                      type="button"
                      className="btn btn-ghost btn-sm"
                      onClick={() => remove(report.id)}
                    >
                      ✕
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </aside>
      </div>
    </div>
  );
}
