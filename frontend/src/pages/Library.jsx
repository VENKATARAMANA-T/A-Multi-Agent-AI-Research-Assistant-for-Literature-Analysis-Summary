import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../api/client';
import { Badge, Card, EmptyState, ErrorBanner, Spinner, statusTone } from '../components/common';
import { useCorpus } from '../context/CorpusContext';

export default function Library() {
  const { papers, loading, refresh, selectedIds, toggleSelected } = useCorpus();
  const [query, setQuery] = useState('');
  const [error, setError] = useState(null);
  const [busyId, setBusyId] = useState(null);
  const [expanded, setExpanded] = useState(null);
  const [chunks, setChunks] = useState({});

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return papers;
    return papers.filter(
      (paper) =>
        (paper.title || '').toLowerCase().includes(needle) ||
        (paper.filename || '').toLowerCase().includes(needle) ||
        (paper.authors || []).some((author) => author.toLowerCase().includes(needle)) ||
        (paper.keywords || []).some((keyword) => keyword.toLowerCase().includes(needle)),
    );
  }, [papers, query]);

  const handleDelete = async (paper) => {
    if (!window.confirm(`Delete "${paper.title || paper.filename}" and all derived data?`)) return;
    setBusyId(paper.id);
    try {
      await api.deletePaper(paper.id);
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusyId(null);
    }
  };

  const handleReindex = async (paper) => {
    setBusyId(paper.id);
    try {
      await api.reindexPaper(paper.id);
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusyId(null);
    }
  };

  const handleExpand = async (paper) => {
    if (expanded === paper.id) {
      setExpanded(null);
      return;
    }
    setExpanded(paper.id);
    if (!chunks[paper.id]) {
      try {
        const payload = await api.getPaperChunks(paper.id, { limit: 5 });
        setChunks((current) => ({ ...current, [paper.id]: payload.chunks }));
      } catch (err) {
        setError(err.message);
      }
    }
  };

  if (loading) return <Spinner label="Loading library…" />;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Library</h1>
          <p className="page-sub">{papers.length} papers · click a row to select it for analysis</p>
        </div>
        <div className="row-actions">
          <input
            className="input input-search"
            placeholder="Filter by title, author or keyword…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <a
            className="btn btn-ghost btn-sm"
            href={api.bibliographyUrl({ style: 'bibtex' })}
            title="Export every paper as a .bib file"
          >
            Export BibTeX
          </a>
        </div>
      </header>

      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      {filtered.length === 0 ? (
        <EmptyState
          title={papers.length === 0 ? 'No papers yet' : 'No matches'}
          description={
            papers.length === 0
              ? 'Upload PDFs to begin building your corpus.'
              : 'Try a different search term.'
          }
        />
      ) : (
        <div className="paper-list">
          {filtered.map((paper) => (
            <Card
              key={paper.id}
              className={selectedIds.includes(paper.id) ? 'is-selected' : ''}
              title={
                <span className="paper-title-row">
                  <input
                    type="checkbox"
                    checked={selectedIds.includes(paper.id)}
                    onChange={() => toggleSelected(paper.id)}
                    disabled={paper.status !== 'indexed'}
                    aria-label={`Select ${paper.title || paper.filename}`}
                  />
                  {paper.title || paper.filename}
                </span>
              }
              subtitle={
                <>
                  {(paper.authors || []).slice(0, 6).join(', ') || 'Unknown authors'}
                  {paper.year ? ` · ${paper.year}` : ''}
                  {paper.venue ? ` · ${paper.venue}` : ''}
                </>
              }
              actions={
                <div className="row-actions">
                  <Badge tone={statusTone(paper.status)}>{paper.status}</Badge>
                  <Link className="btn btn-ghost btn-sm" to={`/app/reader/${paper.id}`}>
                    Read
                  </Link>
                  <a
                    className="btn btn-ghost btn-sm"
                    href={api.paperFileUrl(paper.id)}
                    target="_blank"
                    rel="noreferrer"
                  >
                    PDF
                  </a>
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm"
                    onClick={() => handleExpand(paper)}
                  >
                    {expanded === paper.id ? 'Hide' : 'Details'}
                  </button>
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm"
                    onClick={() => handleReindex(paper)}
                    disabled={busyId === paper.id}
                  >
                    Reindex
                  </button>
                  <button
                    type="button"
                    className="btn btn-danger btn-sm"
                    onClick={() => handleDelete(paper)}
                    disabled={busyId === paper.id}
                  >
                    Delete
                  </button>
                </div>
              }
            >
              <div className="paper-meta">
                <span>{paper.page_count} pages</span>
                <span>{paper.chunk_count} chunks</span>
                <span>{(paper.char_count / 1000).toFixed(1)}k characters</span>
                <span>{(paper.size_bytes / 1048576).toFixed(2)} MB</span>
                {paper.doi && (
                  <a href={`https://doi.org/${paper.doi}`} target="_blank" rel="noreferrer">
                    doi:{paper.doi}
                  </a>
                )}
              </div>

              {paper.error && <p className="error-text">{paper.error}</p>}

              {paper.abstract && <p className="paper-abstract">{paper.abstract}</p>}

              {paper.keywords?.length > 0 && (
                <div className="chip-row">
                  {paper.keywords.map((keyword) => (
                    <span key={keyword} className="chip">
                      {keyword}
                    </span>
                  ))}
                </div>
              )}

              {expanded === paper.id && (
                <div className="paper-details">
                  <h4>Detected sections</h4>
                  {paper.sections?.length ? (
                    <div className="chip-row">
                      {paper.sections.map((section) => (
                        <span key={section} className="chip chip-muted">
                          {section.replace(/_/g, ' ')}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="muted">No canonical section headings detected.</p>
                  )}

                  <h4>First chunks</h4>
                  {chunks[paper.id] ? (
                    <ul className="chunk-list">
                      {chunks[paper.id].map((chunk) => (
                        <li key={chunk.id}>
                          <span className="chunk-meta">
                            #{chunk.index} · p.{chunk.page_start ?? '?'} ·{' '}
                            {chunk.section || 'unlabelled'} · ~{chunk.token_estimate} tokens
                          </span>
                          <p>{chunk.text.slice(0, 400)}…</p>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <Spinner label="Loading chunks…" />
                  )}
                </div>
              )}
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
