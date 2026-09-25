import { useCallback, useEffect, useState } from 'react';
import api from '../api/client';
import PaperPicker from '../components/PaperPicker';
import { Badge, Card, EmptyState, ErrorBanner, Spinner, WarningBanner } from '../components/common';
import { useCorpus } from '../context/CorpusContext';

const TYPES = [
  ['text', 'Text'],
  ['number', 'Number'],
  ['boolean', 'Yes / No'],
  ['list', 'List'],
];

const PRESETS = [
  { name: 'Sample size', description: 'Number of participants or examples.', type: 'number' },
  { name: 'Dataset', description: 'Corpora or benchmarks used.', type: 'list' },
  { name: 'Code released?', description: 'Did the authors publish their code?', type: 'boolean' },
  { name: 'Hardware', description: 'GPUs or compute used for training.', type: 'text' },
  { name: 'Limitations', description: 'Weaknesses the authors acknowledge.', type: 'list' },
];

const blank = () => ({ name: '', description: '', type: 'text' });

export default function Matrix() {
  const { effectiveIds, indexedPapers, llmReady } = useCorpus();
  const [columns, setColumns] = useState([blank()]);
  const [name, setName] = useState('');
  const [result, setResult] = useState(null);
  const [saved, setSaved] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [evidenceFor, setEvidenceFor] = useState(null);

  const loadSaved = useCallback(async () => {
    try {
      setSaved(await api.listMatrixRuns());
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    loadSaved();
  }, [loadSaved]);

  const update = (index, patch) =>
    setColumns((current) => current.map((col, i) => (i === index ? { ...col, ...patch } : col)));

  const run = async () => {
    const usable = columns.filter((column) => column.name.trim());
    if (usable.length === 0) {
      setError('Add at least one column with a name.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const payload = await api.runMatrix({
        columns: usable,
        paper_ids: effectiveIds,
        name: name.trim() || null,
      });
      setResult(payload);
      await loadSaved();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const open = async (id) => {
    try {
      setResult(await api.getMatrixRun(id));
    } catch (err) {
      setError(err.message);
    }
  };

  const remove = async (id) => {
    if (!window.confirm('Delete this comparison?')) return;
    try {
      await api.deleteMatrixRun(id);
      if (result?.id === id) setResult(null);
      await loadSaved();
    } catch (err) {
      setError(err.message);
    }
  };

  const cellText = (cell) => {
    if (!cell) return '—';
    const value = cell.value;
    return Array.isArray(value) ? value.join(', ') : value ?? '—';
  };

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Comparison matrix</h1>
          <p className="page-sub">
            Define your own columns and have them filled across every paper — the comparison
            table a literature review actually needs.
          </p>
        </div>
        <button
          type="button"
          className="btn btn-primary"
          onClick={run}
          disabled={busy || indexedPapers.length === 0}
        >
          {busy ? 'Filling…' : `Fill for ${effectiveIds.length} paper${effectiveIds.length === 1 ? '' : 's'}`}
        </button>
      </header>

      {!llmReady && <WarningBanner>Gemini is not configured — this will return an error.</WarningBanner>}
      {effectiveIds.length > 0 && (
        <WarningBanner>
          This costs <strong>{effectiveIds.length} API request{effectiveIds.length === 1 ? '' : 's'}</strong> —
          one per paper, regardless of how many columns you define.
        </WarningBanner>
      )}
      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      <div className="grid-side">
        <div>
          <Card
            title="Columns"
            subtitle="Ask for anything the papers might state"
            actions={
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={() => setColumns((c) => [...c, blank()])}
              >
                + Add column
              </button>
            }
          >
            <label className="field">
              Name this comparison (optional)
              <input
                className="input"
                placeholder="Reproducibility across speech papers"
                value={name}
                onChange={(event) => setName(event.target.value)}
              />
            </label>

            {columns.map((column, index) => (
              <div key={index} className="matrix-column-row">
                <input
                  className="input"
                  placeholder="Column name, e.g. Sample size"
                  value={column.name}
                  onChange={(event) => update(index, { name: event.target.value })}
                />
                <input
                  className="input"
                  placeholder="What exactly should it contain?"
                  value={column.description}
                  onChange={(event) => update(index, { description: event.target.value })}
                />
                <select
                  className="input input-select"
                  value={column.type}
                  onChange={(event) => update(index, { type: event.target.value })}
                >
                  {TYPES.map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  onClick={() => setColumns((c) => c.filter((_, i) => i !== index))}
                  disabled={columns.length === 1}
                  aria-label="Remove column"
                >
                  ✕
                </button>
              </div>
            ))}

            <div className="chip-row">
              {PRESETS.map((preset) => (
                <button
                  key={preset.name}
                  type="button"
                  className="chip chip-button"
                  onClick={() =>
                    setColumns((current) => {
                      const usable = current.filter((c) => c.name.trim());
                      return [...usable, preset];
                    })
                  }
                >
                  + {preset.name}
                </button>
              ))}
            </div>
          </Card>

          {busy && <Spinner label="Reading each paper…" />}

          {result && (
            <Card
              title={result.name || 'Comparison'}
              subtitle={`${result.rows.length} papers · ${result.columns.length} columns · ${result.llm_calls} requests`}
              actions={
                result.id && (
                  <a className="btn btn-ghost btn-sm" href={api.matrixCsvUrl(result.id)}>
                    Download CSV
                  </a>
                )
              }
            >
              {result.errors?.length > 0 && (
                <ul className="bullets">
                  {result.errors.map((message, index) => (
                    <li key={index} className="error-text">
                      {message}
                    </li>
                  ))}
                </ul>
              )}

              {result.rows.length === 0 ? (
                <EmptyState title="No rows" description="Every paper failed — see the errors above." />
              ) : (
                <div className="table-scroll">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Paper</th>
                        {result.columns.map((column) => (
                          <th key={column.key}>{column.name}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.rows.map((row) => (
                        <tr key={row.paper_id}>
                          <td>
                            <strong>{row.paper_title}</strong>
                          </td>
                          {result.columns.map((column) => {
                            const cell = row.cells?.[column.key];
                            const id = `${row.paper_id}:${column.key}`;
                            return (
                              <td key={column.key}>
                                <button
                                  type="button"
                                  className={`matrix-cell ${cell?.reported ? '' : 'is-missing'}`}
                                  onClick={() => setEvidenceFor(evidenceFor === id ? null : id)}
                                  title={cell?.evidence || 'No supporting text'}
                                >
                                  {cellText(cell)}
                                </button>
                                {evidenceFor === id && cell?.evidence && (
                                  <div className="matrix-evidence">“{cell.evidence}”</div>
                                )}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <p className="muted">
                Click any cell to see the sentence it came from. Greyed cells mean the paper does
                not report that item — they are deliberately blank rather than guessed.
              </p>
            </Card>
          )}
        </div>

        <aside>
          <Card title="Corpus scope">
            <PaperPicker compact />
          </Card>
          <Card title="Saved comparisons">
            {saved.length === 0 ? (
              <p className="muted">None yet.</p>
            ) : (
              <ul className="report-list">
                {saved.map((run) => (
                  <li key={run.id}>
                    <button type="button" className="report-item" onClick={() => open(run.id)}>
                      <strong>{run.name}</strong>
                      <span className="muted">
                        {run.paper_count} papers · {run.column_count} columns
                      </span>
                    </button>
                    <button type="button" className="btn btn-ghost btn-sm" onClick={() => remove(run.id)}>
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
