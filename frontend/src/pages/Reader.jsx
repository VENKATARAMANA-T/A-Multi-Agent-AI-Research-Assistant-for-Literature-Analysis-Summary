import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useParams, useSearchParams } from 'react-router-dom';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';
import api, { getAccessToken } from '../api/client';
import { Badge, Card, ErrorBanner, Spinner } from '../components/common';

// Vite resolves the worker from the installed pdfjs-dist, so the viewer works
// offline instead of reaching for a CDN.
const workerUrl = new URL('pdfjs-dist/build/pdf.worker.min.mjs', import.meta.url);

// The `v` is a cache-buster, and it has to be here rather than in the filename.
// This worker was once served with the wrong MIME type (nginx has no entry for
// .mjs) under `Cache-Control: immutable`, which tells a browser not to
// revalidate for a year — so every browser that saw the broken response keeps
// replaying it and the reader stays broken after the server is fixed. The
// filename's content hash cannot rescue us: the file itself never changed, so
// the hash is identical. A new query string is the only thing that makes the
// browser ask again. Bump it if a cached asset ever needs forcing out.
workerUrl.searchParams.set('v', '2');
pdfjs.GlobalWorkerOptions.workerSrc = workerUrl.toString();

const LEVELS = [
  ['simple', 'Plain English'],
  ['standard', 'Standard'],
  ['technical', 'Technical'],
];

export default function Reader() {
  const { paperId } = useParams();
  const [params, setParams] = useSearchParams();

  const [paper, setPaper] = useState(null);
  const [numPages, setNumPages] = useState(0);
  const [page, setPage] = useState(Number(params.get('page')) || 1);
  const [scale, setScale] = useState(1.2);
  const [pageSize, setPageSize] = useState({ width: 0, height: 0 });
  const [highlight, setHighlight] = useState(null);
  const [error, setError] = useState(null);

  const [selection, setSelection] = useState('');
  const [level, setLevel] = useState('standard');
  const [explaining, setExplaining] = useState(false);
  const [explanation, setExplanation] = useState(null);
  const [citation, setCitation] = useState(null);

  const pageRef = useRef(null);
  const containerRef = useRef(null);
  const chunkId = params.get('chunk');

  /**
   * PDF.js fetches the file itself, outside our API client, so the bearer token
   * has to be handed to it explicitly — otherwise every paper fails with a 401
   * now that the endpoint belongs to an account.
   *
   * Memoised because react-pdf treats a new `file` object as a new document: an
   * inline literal would be a fresh reference on every render and the viewer
   * would reload the PDF endlessly.
   */
  const pdfSource = useMemo(
    () => ({
      url: api.paperFileUrl(paperId),
      httpHeaders: { Authorization: `Bearer ${getAccessToken()}` },
    }),
    [paperId],
  );

  /** Opens the original in a new tab, which also cannot send our header. */
  const openOriginal = useCallback(async () => {
    try {
      const response = await fetch(api.paperFileUrl(paperId), {
        headers: { Authorization: `Bearer ${getAccessToken()}` },
      });
      if (!response.ok) throw new Error(`The server refused the file (${response.status}).`);

      // A blob URL carries no headers, so the new tab needs no credentials.
      const url = URL.createObjectURL(await response.blob());
      window.open(url, '_blank', 'noopener');
      // Revoked late: revoking immediately can race the tab's own load.
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } catch (err) {
      setError(err.message);
    }
  }, [paperId]);
  const figureId = params.get('figure');

  /* Measure the rendered canvas rather than trusting the render callback's
     shape: the overlay maths needs the page's CSS width, and reading it from
     the DOM keeps this working across react-pdf versions. */
  const measurePage = useCallback((rendered) => {
    // react-pdf reports the rendered size; fall back to the DOM if a future
    // version stops passing it, so the overlay maths never silently breaks.
    let width = rendered?.width;
    let height = rendered?.height;
    if (!width) {
      const canvas = pageRef.current?.querySelector('canvas');
      width = canvas?.clientWidth || canvas?.width;
      height = canvas?.clientHeight || canvas?.height;
    }
    if (width) setPageSize((current) => (current.width === width ? current : { width, height }));
  }, []);

  /* The canvas is mounted by <Document> only after the PDF loads, so an effect
     keyed on page/scale runs too early to find it, and react-pdf's render
     callback does not reliably carry the size. Watching the container catches
     the canvas whenever it appears or changes size. */
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;

    measurePage();
    const resize = new ResizeObserver(() => measurePage());
    const mutation = new MutationObserver(() => measurePage());
    resize.observe(container);
    mutation.observe(container, { childList: true, subtree: true });

    return () => {
      resize.disconnect();
      mutation.disconnect();
    };
  }, [measurePage]);

  useEffect(() => {
    api.getPaper(paperId).then(setPaper).catch((err) => setError(err.message));
  }, [paperId]);

  // Resolve a citation to its exact rectangles on the page.
  useEffect(() => {
    if (!chunkId && !figureId) {
      setHighlight(null);
      return;
    }
    api
      .highlight({ paper_id: paperId, chunk_id: chunkId, figure_id: figureId })
      .then((result) => {
        setHighlight(result);
        if (result.page) setPage(result.page);
      })
      .catch((err) => setError(err.message));
  }, [paperId, chunkId, figureId]);

  const onSelect = useCallback(() => {
    const text = window.getSelection()?.toString().trim() ?? '';
    if (text.length >= 10) setSelection(text);
  }, []);

  const explain = async () => {
    setExplaining(true);
    setExplanation(null);
    try {
      setExplanation(await api.explain({ text: selection, level, paper_id: paperId }));
    } catch (err) {
      setError(err.message);
    } finally {
      setExplaining(false);
    }
  };

  const copyCitation = async (style) => {
    try {
      const result = await api.citation(paperId, style);
      setCitation(result.text);
      await navigator.clipboard?.writeText(result.text).catch(() => {});
    } catch (err) {
      setError(err.message);
    }
  };

  const goTo = (next) => {
    const clamped = Math.min(Math.max(1, next), numPages || 1);
    setPage(clamped);
    const updated = new URLSearchParams(params);
    updated.set('page', String(clamped));
    setParams(updated, { replace: true });
  };

  // PDF coordinates have their origin at the bottom-left; CSS starts top-left.
  const overlays = useMemo(() => {
    if (!highlight?.rects?.length || highlight.page !== page) return [];
    if (!pageSize.width || !highlight.page_width) return [];
    const ratio = pageSize.width / highlight.page_width;
    return highlight.rects.map(([x0, y0, x1, y1], index) => ({
      key: index,
      left: x0 * ratio,
      top: y0 * ratio,
      width: (x1 - x0) * ratio,
      height: (y1 - y0) * ratio,
    }));
  }, [highlight, page, pageSize]);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>{paper?.title || 'Reader'}</h1>
          <p className="page-sub">
            {(paper?.authors || []).slice(0, 4).join(', ')}
            {paper?.year ? ` · ${paper.year}` : ''}
            {paper?.page_count ? ` · ${paper.page_count} pages` : ''}
          </p>
        </div>
        <div className="row-actions">
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => copyCitation('bibtex')}>
            Copy BibTeX
          </button>
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => copyCitation('apa')}>
            Copy APA
          </button>
          <button type="button" className="btn btn-ghost btn-sm" onClick={openOriginal}>
            Original
          </button>
        </div>
      </header>

      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      {citation && (
        <Card title="Citation" actions={
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => setCitation(null)}>
            Close
          </button>
        }>
          <pre className="figure-table">{citation}</pre>
          <p className="muted">Copied to the clipboard.</p>
        </Card>
      )}

      {highlight && !highlight.found && (
        <Card title="Passage not located">
          <p className="muted">
            The cited text could not be matched in the PDF — extraction normalises ligatures and
            hyphenation, so a few passages will not match exactly. Page {highlight.page} is shown.
          </p>
        </Card>
      )}

      <div className="reader-layout">
        <div className="reader-main">
          <div className="reader-toolbar">
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => goTo(page - 1)} disabled={page <= 1}>
              ‹ Prev
            </button>
            <span className="muted">
              Page {page} of {numPages || '?'}
            </span>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => goTo(page + 1)}
              disabled={numPages > 0 && page >= numPages}
            >
              Next ›
            </button>
            <span className="reader-spacer" />
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setScale((s) => Math.max(0.6, s - 0.2))}>
              −
            </button>
            <span className="muted">{Math.round(scale * 100)}%</span>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setScale((s) => Math.min(2.6, s + 0.2))}>
              +
            </button>
          </div>

          <div className="reader-canvas" onMouseUp={onSelect} ref={containerRef}>
            <Document
              file={pdfSource}
              onLoadSuccess={({ numPages: total }) => setNumPages(total)}
              onLoadError={(err) => setError(`Could not open the PDF: ${err.message}`)}
              loading={<Spinner label="Loading PDF…" />}
            >
              <div className="reader-page" ref={pageRef}>
                <Page
                  pageNumber={page}
                  scale={scale}
                  renderAnnotationLayer={false}
                  onRenderSuccess={measurePage}
                />
                {overlays.map((rect) => (
                  <span
                    key={rect.key}
                    className="reader-highlight"
                    style={{
                      left: `${rect.left}px`,
                      top: `${rect.top}px`,
                      width: `${rect.width}px`,
                      height: `${rect.height}px`,
                    }}
                  />
                ))}
              </div>
            </Document>
          </div>
        </div>

        <aside className="reader-side">
          <Card title="Explain a passage" className="explain-card">
            {selection ? (
              <>
                <blockquote className="reader-selection">{selection.slice(0, 400)}</blockquote>
                <div className="tabs">
                  {LEVELS.map(([key, label]) => (
                    <button
                      key={key}
                      type="button"
                      className={`tab ${level === key ? 'is-active' : ''}`}
                      onClick={() => setLevel(key)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <button type="button" className="btn btn-primary btn-sm" onClick={explain} disabled={explaining}>
                  {explaining ? 'Explaining…' : 'Explain this'}
                </button>
              </>
            ) : (
              <p className="muted">
                Select any text in the page to have it explained. Selections shorter than ten
                characters are ignored.
              </p>
            )}

            {explanation && (
              <div className="figure-detail explain-result">
                {explanation.explanation ? (
                  <>
                    <p>{explanation.explanation}</p>

                    {explanation.terms?.length > 0 && (
                      <>
                        <h4>Terms</h4>
                        <ul className="bullets">
                          {explanation.terms.map((item, index) => (
                            <li key={index}>
                              <strong>{item.term}</strong> — {item.meaning}
                            </li>
                          ))}
                        </ul>
                      </>
                    )}

                    {explanation.background && (
                      <div className="callout">
                        <strong>Background (not from this passage)</strong>
                        <p>{explanation.background}</p>
                      </div>
                    )}

                    {explanation.why_it_matters && (
                      <div className="direction">
                        <strong>Why it matters</strong>
                        <p>{explanation.why_it_matters}</p>
                      </div>
                    )}

                    {explanation.caveats?.length > 0 && (
                      <>
                        <h4>Caveats</h4>
                        <ul className="bullets">
                          {explanation.caveats.map((item, index) => (
                            <li key={index}>{item}</li>
                          ))}
                        </ul>
                      </>
                    )}
                  </>
                ) : (
                  <p className="error-text">{explanation.errors?.join(' ') || 'No explanation returned.'}</p>
                )}
              </div>
            )}
          </Card>

          {highlight?.found && (
            <Card title="Cited passage">
              <Badge tone="success">Highlighted on page {highlight.page}</Badge>
              <p className="muted">Matched on: “{highlight.matched_phrase}”</p>
            </Card>
          )}
        </aside>
      </div>
    </div>
  );
}
