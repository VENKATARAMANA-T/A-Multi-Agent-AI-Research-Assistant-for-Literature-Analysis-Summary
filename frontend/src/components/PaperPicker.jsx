import { useCorpus } from '../context/CorpusContext';

/** Corpus scope selector — an empty selection means "all indexed papers". */
export default function PaperPicker({ mode = 'multi', compact = false }) {
  const { indexedPapers, selectedIds, setSelectedIds, toggleSelected, selectAll, clearSelection } =
    useCorpus();

  if (indexedPapers.length === 0) {
    return <p className="muted">No indexed papers yet — upload some PDFs first.</p>;
  }

  const handleClick = (id) => {
    if (mode === 'single') setSelectedIds(selectedIds[0] === id ? [] : [id]);
    else toggleSelected(id);
  };

  return (
    <div className={`picker ${compact ? 'picker-compact' : ''}`}>
      <div className="picker-toolbar">
        <span className="muted">
          {selectedIds.length === 0
            ? `Scope: all ${indexedPapers.length} papers`
            : `Scope: ${selectedIds.length} of ${indexedPapers.length} selected`}
        </span>
        {mode === 'multi' && (
          <div className="picker-buttons">
            <button type="button" className="btn btn-ghost btn-sm" onClick={selectAll}>
              Select all
            </button>
            <button type="button" className="btn btn-ghost btn-sm" onClick={clearSelection}>
              Clear
            </button>
          </div>
        )}
      </div>

      <ul className="picker-list">
        {indexedPapers.map((paper) => {
          const active = selectedIds.includes(paper.id);
          return (
            <li key={paper.id}>
              <button
                type="button"
                className={`picker-item ${active ? 'is-active' : ''}`}
                onClick={() => handleClick(paper.id)}
                aria-pressed={active}
              >
                <span className="picker-check" aria-hidden="true">
                  {active ? '✓' : ''}
                </span>
                <span className="picker-text">
                  <span className="picker-title">{paper.title || paper.filename}</span>
                  <span className="picker-meta">
                    {(paper.authors || []).slice(0, 2).join(', ') || 'Unknown authors'}
                    {paper.year ? ` · ${paper.year}` : ''} · {paper.chunk_count} chunks
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
