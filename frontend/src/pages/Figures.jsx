import { useCallback, useEffect, useMemo, useState } from 'react';
import api from '../api/client';
import PaperPicker from '../components/PaperPicker';
import { Badge, Card, EmptyState, ErrorBanner, Spinner, WarningBanner } from '../components/common';
import { useCorpus } from '../context/CorpusContext';

const KIND_TONES = { figure: 'info', table: 'success', chart: 'info', algorithm: 'neutral' };
const STATUS_TONES = { analysed: 'success', pending: 'warning', failed: 'danger', skipped: 'neutral' };

const FILTERS = [
  ['all', 'All'],
  ['figure', 'Figures'],
  ['table', 'Tables'],
  ['analysed', 'Read by AI'],
  ['pending', 'Not yet read'],
];

export default function Figures() {
  // Deliberately `selectedIds`, not `effectiveIds`. Everywhere else an empty
  // selection means "the whole corpus", but a wall of every figure from every
  // paper is not a useful default here — and "Read 40 figures" costs 40
  // requests. Nothing selected shows nothing, so the scope is always explicit.
  const { selectedIds, indexedPapers, llmReady } = useCorpus();
  const [figures, setFigures] = useState([]);
  const [estimate, setEstimate] = useState(null);
  const [filter, setFilter] = useState('all');
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [expanded, setExpanded] = useState(null);

  const load = useCallback(async () => {
    if (selectedIds.length === 0) {
      setFigures([]);
      setEstimate(null);
      setLoading(false);
      return;
    }

    setLoading(true);
    try {
      const [items, cost] = await Promise.all([
        api.listFigures({ paper_ids: selectedIds }),
        api.figureEstimate({ paper_ids: selectedIds }).catch(() => null),
      ]);
      setFigures(items);
      setEstimate(cost);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [selectedIds]);

  useEffect(() => {
    load();
  }, [load]);

  const analyse = async () => {
    setBusy(true);
    setError(null);
    try {
      const payload = await api.analyseFigures({ paper_ids: selectedIds });
      setResult(payload);
      await load();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const visible = useMemo(() => {
    if (filter === 'all') return figures;
    if (filter === 'analysed' || filter === 'pending') {
      return figures.filter((item) => item.status === filter);
    }
    return figures.filter((item) => item.kind === filter);
  }, [figures, filter]);

  const pending = estimate?.pending ?? 0;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Figures &amp; tables</h1>
          <p className="page-sub">
            {selectedIds.length === 0
              ? 'Select a paper on the right to see the charts, diagrams and tables lifted out of it.'
              : `Charts, diagrams and tables from ${selectedIds.length} selected paper${
                  selectedIds.length === 1 ? '' : 's'
                } — and what the vision model reads in them.`}
          </p>
        </div>
        <button
          type="button"
          className="btn btn-primary"
          onClick={analyse}
          disabled={busy || pending === 0 || selectedIds.length === 0}
        >
          {busy ? 'Reading figures…' : `Read ${pending} figure${pending === 1 ? '' : 's'}`}
        </button>
      </header>

      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      {!llmReady && (
        <WarningBanner>
          Gemini is not configured, so figures cannot be read. Extraction still works — the
          images and any tables parsed directly from the PDF are below.
        </WarningBanner>
      )}

      {pending > 0 && (
        <WarningBanner>
          <strong>
            Reading these costs {estimate.requests_required} API request
            {estimate.requests_required === 1 ? '' : 's'}
          </strong>{' '}
          — one per figure. On the Gemini free tier the daily allowance is about 20 requests in
          total, so this is worth spending deliberately.
          {estimate.already_readable > 0 && (
            <>
              {' '}
              {estimate.already_readable} table
              {estimate.already_readable === 1 ? ' was' : 's were'} parsed straight from the PDF
              and cost nothing.
            </>
          )}
        </WarningBanner>
      )}

      {result && (
        <Card title="Last run" subtitle={`${result.analysed} read · ${result.failed} failed · ${result.llm_calls} requests used`}>
          {result.errors?.length > 0 && (
            <ul className="bullets">
              {result.errors.map((message, index) => (
                <li key={index} className="error-text">
                  {message}
                </li>
              ))}
            </ul>
          )}
          {result.detail && <p className="muted">{result.detail}</p>}
        </Card>
      )}

      <div className="grid-side">
        <div>
          <div className="tabs">
            {FILTERS.map(([key, label]) => (
              <button
                key={key}
                type="button"
                className={`tab ${filter === key ? 'is-active' : ''}`}
                onClick={() => setFilter(key)}
              >
                {label}
                <span className="tab-count">
                  {key === 'all'
                    ? figures.length
                    : figures.filter((f) => f.kind === key || f.status === key).length}
                </span>
              </button>
            ))}
          </div>

          {loading ? (
            <Spinner label="Loading figures…" />
          ) : selectedIds.length === 0 ? (
            <EmptyState
              title="No paper selected"
              description={
                indexedPapers.length === 0
                  ? 'Upload a paper containing charts or tables — they are extracted automatically during indexing.'
                  : 'Choose one or more papers under “Corpus scope” to see their figures and tables.'
              }
            />
          ) : visible.length === 0 ? (
            <EmptyState
              title={figures.length === 0 ? 'No figures in these papers' : 'Nothing matches this filter'}
              description={
                figures.length === 0
                  ? 'This paper has no extractable charts or tables. Figures are found by their captions, so a paper without numbered captions yields none.'
                  : 'Try another filter.'
              }
            />
          ) : (
            <div className="figure-grid">
              {visible.map((figure) => (
                <Card
                  key={figure.id}
                  title={
                    <span className="figure-title">
                      {figure.label}
                      <Badge tone={KIND_TONES[figure.kind] || 'neutral'}>{figure.kind}</Badge>
                      <Badge tone={STATUS_TONES[figure.status] || 'neutral'}>{figure.status}</Badge>
                    </span>
                  }
                  subtitle={`${figure.paper_title || figure.paper_id} · page ${figure.page}`}
                  actions={
                    <button
                      type="button"
                      className="btn btn-ghost btn-sm"
                      onClick={() => setExpanded(expanded === figure.id ? null : figure.id)}
                    >
                      {expanded === figure.id ? 'Less' : 'More'}
                    </button>
                  }
                >
                  {figure.has_image && (
                    <img
                      className="figure-image"
                      src={api.figureImageUrl(figure.id)}
                      alt={figure.caption || figure.label}
                      loading="lazy"
                    />
                  )}

                  <p className="figure-caption">{figure.caption}</p>

                  {figure.takeaway && (
                    <div className="direction">
                      <strong>What it shows</strong>
                      <p>{figure.takeaway}</p>
                    </div>
                  )}

                  {expanded === figure.id && (
                    <div className="figure-detail">
                      {figure.description && (
                        <>
                          <h4>Description</h4>
                          <p>{figure.description}</p>
                        </>
                      )}

                      {figure.chart_type && (
                        <p className="muted">Chart type: {figure.chart_type}</p>
                      )}

                      {Object.keys(figure.axes || {}).length > 0 && (
                        <>
                          <h4>Axes</h4>
                          <ul className="bullets">
                            {Object.entries(figure.axes).map(([key, value]) => (
                              <li key={key}>
                                <strong>{key.replace(/_/g, ' ')}:</strong> {value}
                              </li>
                            ))}
                          </ul>
                        </>
                      )}

                      {figure.series?.length > 0 && (
                        <>
                          <h4>Series</h4>
                          <ul className="bullets">
                            {figure.series.map((item, index) => (
                              <li key={index}>
                                <strong>{item.name}</strong>
                                {item.trend ? ` — ${item.trend}` : ''}
                                {item.notable_values?.length > 0 && (
                                  <div className="cell-sub">{item.notable_values.join(' · ')}</div>
                                )}
                              </li>
                            ))}
                          </ul>
                        </>
                      )}

                      {figure.findings?.length > 0 && (
                        <>
                          <h4>Findings</h4>
                          <ul className="bullets">
                            {figure.findings.map((item, index) => (
                              <li key={index}>{item}</li>
                            ))}
                          </ul>
                        </>
                      )}

                      {figure.table_markdown && (
                        <>
                          <h4>Parsed table</h4>
                          <pre className="figure-table">{figure.table_markdown}</pre>
                        </>
                      )}

                      {figure.analysis_error && (
                        <p className="error-text">{figure.analysis_error}</p>
                      )}
                    </div>
                  )}
                </Card>
              ))}
            </div>
          )}
        </div>

        <aside>
          <Card title="Corpus scope">
            <PaperPicker compact />
          </Card>
          <Card title="How this works">
            <p className="muted">
              Extraction happens automatically when a paper is indexed and costs nothing.
              Reading a figure needs the vision model, which is one API request per figure —
              so it stays a deliberate action rather than something that happens on upload.
            </p>
            <p className="muted">
              Tables that could be parsed straight from the PDF are marked <em>skipped</em>:
              their contents are already available, so a picture of them would add nothing.
            </p>
          </Card>
        </aside>
      </div>
    </div>
  );
}
