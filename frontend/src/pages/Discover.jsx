import { useState } from 'react';
import api from '../api/client';
import PaperPicker from '../components/PaperPicker';
import { Badge, Card, EmptyState, ErrorBanner, Spinner } from '../components/common';
import { useCorpus } from '../context/CorpusContext';

const MODES = [
  ['gaps', 'Missing from your corpus'],
  ['related', 'Related to one paper'],
  ['search', 'Search the literature'],
];

export default function Discover() {
  const { indexedPapers, selectedIds, effectiveIds } = useCorpus();
  const [mode, setMode] = useState('gaps');
  const [paperId, setPaperId] = useState('');
  const [query, setQuery] = useState('');
  const [yearFrom, setYearFrom] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      if (mode === 'gaps') {
        if (effectiveIds.length === 0) throw new Error('Upload a paper first.');
        setResult(await api.discoverGaps({ paper_ids: effectiveIds }));
      } else if (mode === 'related') {
        const target = paperId || indexedPapers[0]?.id;
        if (!target) throw new Error('Select a paper first.');
        setResult(await api.discoverRelated({ paper_id: target }));
      } else {
        if (query.trim().length < 3) throw new Error('Enter at least three characters.');
        setResult(
          await api.discoverSearch({
            q: query.trim(),
            year_from: yearFrom ? Number(yearFrom) : undefined,
          }),
        );
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Discover</h1>
          <p className="page-sub">
            Find related papers in OpenAlex, an open catalogue of around 250 million works.
            Anything already in your library is filtered out, and nothing is added to it
            automatically — download what looks useful and upload it yourself.
          </p>
        </div>
        <button type="button" className="btn btn-primary" onClick={run} disabled={busy}>
          {busy ? 'Searching…' : 'Search'}
        </button>
      </header>

      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      <Card title="What to look for">
        <div className="tabs">
          {MODES.map(([key, label]) => (
            <button
              key={key}
              type="button"
              className={`tab ${mode === key ? 'is-active' : ''}`}
              onClick={() => setMode(key)}
            >
              {label}
            </button>
          ))}
        </div>

        {mode === 'related' && (
          <label className="field">
            Paper
            <select
              className="input input-select"
              value={paperId}
              onChange={(event) => setPaperId(event.target.value)}
            >
              <option value="">— choose a paper —</option>
              {indexedPapers.map((paper) => (
                <option key={paper.id} value={paper.id}>
                  {paper.title || paper.filename}
                </option>
              ))}
            </select>
          </label>
        )}

        {mode === 'search' && (
          <>
            <label className="field">
              Query
              <input
                className="input"
                placeholder="dysarthria severity classification wav2vec"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => event.key === 'Enter' && run()}
              />
            </label>
            <label className="field">
              Published from (optional)
              <input
                className="input input-number"
                type="number"
                placeholder="2020"
                value={yearFrom}
                onChange={(event) => setYearFrom(event.target.value)}
              />
            </label>
          </>
        )}

        {mode === 'gaps' && (
          <>
            <p className="muted">
              Looks at what the papers below cite and relate to, then ranks the works you do not
              have by how often they recur. A paper several of yours point at is the most likely
              thing missing from the review.
            </p>
            <label className="field">
              Base the search on
              <PaperPicker compact />
            </label>
            <p className="muted">
              {selectedIds.length === 0
                ? `Searching all ${indexedPapers.length} paper${
                    indexedPapers.length === 1 ? '' : 's'
                  } — tick specific ones above to narrow it.`
                : `Searching ${selectedIds.length} selected paper${
                    selectedIds.length === 1 ? '' : 's'
                  }.`}{' '}
              One OpenAlex lookup per paper, capped at 12. No API key and no Gemini quota is used.
            </p>
          </>
        )}
      </Card>

      {busy && <Spinner label="Querying OpenAlex…" />}

      {result && (
        <Card
          title={`${result.candidates.length} candidate${result.candidates.length === 1 ? '' : 's'}`}
          subtitle={`for “${result.query}”${result.excluded_known ? ' · papers you already have are excluded' : ''}`}
        >
          {result.searched_papers?.length > 0 && (
            <details className="retrieved">
              <summary>Based on {result.searched_papers.length} of your papers</summary>
              <ul className="bullets">
                {result.searched_papers.map((title) => (
                  <li key={title}>{title}</li>
                ))}
              </ul>
            </details>
          )}
          {result.candidates.length === 0 ? (
            <EmptyState
              title="Nothing new found"
              description="Either the catalogue has no related work, or you already have all of it."
            />
          ) : (
            <div className="candidate-list">
              {result.candidates.map((item) => (
                <div key={item.external_id} className="candidate">
                  <div className="candidate-head">
                    <strong>{item.title}</strong>
                    {item.referenced_by_corpus > 1 && (
                      <Badge tone="warning">
                        linked from {item.referenced_by_corpus} of your papers
                      </Badge>
                    )}
                    <Badge tone="neutral">{item.relation}</Badge>
                  </div>
                  <p className="muted">
                    {(item.authors || []).slice(0, 4).join(', ')}
                    {item.authors?.length > 4 ? ' et al.' : ''}
                    {item.year ? ` · ${item.year}` : ''}
                    {item.venue ? ` · ${item.venue}` : ''}
                    {` · ${item.citations} citations`}
                  </p>
                  {item.abstract && <p className="candidate-abstract">{item.abstract.slice(0, 340)}…</p>}
                  <div className="row-actions">
                    {item.open_access_url && (
                      <a
                        className="btn btn-primary btn-sm"
                        href={item.open_access_url}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Open access PDF
                      </a>
                    )}
                    {item.doi && (
                      <a
                        className="btn btn-ghost btn-sm"
                        href={`https://doi.org/${item.doi}`}
                        target="_blank"
                        rel="noreferrer"
                      >
                        doi:{item.doi}
                      </a>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
          <p className="muted">
            Download anything useful and upload it on the Upload page — candidates are suggestions
            to review, not papers added to your corpus.
          </p>
        </Card>
      )}
    </div>
  );
}
