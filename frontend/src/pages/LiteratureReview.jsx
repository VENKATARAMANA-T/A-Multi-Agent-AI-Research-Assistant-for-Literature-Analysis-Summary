import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import api from '../api/client';
import PaperPicker from '../components/PaperPicker';
import { Badge, Card, EmptyState, ErrorBanner, Spinner, WarningBanner } from '../components/common';
import { useCorpus } from '../context/CorpusContext';
import { useFlash } from '../context/FlashContext';

const CITATION_RE = /\[\s*S(\d{1,3})((?:\s*(?:,|;|&|and)\s*S?\d{1,3})*)\s*\]/gi;

const SUGGESTIONS = [
  'Speech-based detection of neurodevelopmental and speech disorders',
  'Self-supervised audio representations for clinical speech analysis',
  'Privacy-preserving machine learning for speech data',
];

/**
 * Rewrites every [S#] marker as a Markdown link before rendering.
 *
 * Walking react-markdown's children to find citations means missing every one
 * that falls inside bold text, a list item or a table cell. Turning the marker
 * into a link in the source instead lets the Markdown renderer place it, and a
 * custom `a` component turns that link into a router link — so a citation works
 * wherever the model happened to put it.
 *
 * Grouped citations are split here too, in case an older saved review still
 * holds "[S2, S3]" from before the backend started splitting them.
 */
function linkifyCitations(markdown) {
  return markdown.replace(CITATION_RE, (whole, first, rest) => {
    const numbers = [first, ...(rest || '').match(/\d{1,3}/g) || []];
    const seen = [];
    numbers.forEach((n) => {
      const marker = `S${Number(n)}`;
      if (!seen.includes(marker)) seen.push(marker);
    });
    return seen.map((marker) => `[${marker}](rc-cite:${marker})`).join('');
  });
}

/** One section's prose, rendered as Markdown with live citations. */
function Prose({ text, byMarker, lead = false }) {
  const components = {
    a({ href, children, ...rest }) {
      if (!href?.startsWith('rc-cite:')) {
        return (
          <a href={href} target="_blank" rel="noreferrer" {...rest}>
            {children}
          </a>
        );
      }

      const marker = href.slice('rc-cite:'.length);
      const citation = byMarker[marker];
      if (!citation) return <span className="cite-dead">[{marker}]</span>;

      return (
        <Link
          className="cite-link"
          to={`/app/reader/${citation.paper_id}`}
          title={`${citation.title}${citation.year ? ` (${citation.year})` : ''}`}
        >
          {marker}
        </Link>
      );
    },
  };

  return (
    <div className={`review-prose ${lead ? 'is-lead' : ''}`}>
      {/* react-markdown strips URLs whose scheme it does not recognise, which
          emptied the href on every citation and left fifty dead anchors in the
          page. The identity transform keeps `rc-cite:` intact; it is safe here
          because these links are generated from our own markers, never from
          anything a model or a document supplied. */}
      <ReactMarkdown components={components} urlTransform={(url) => url}>
        {linkifyCitations(text)}
      </ReactMarkdown>
    </div>
  );
}

const WORDS_PER_MINUTE = 220;

function readingStats(sections) {
  const words = sections.reduce(
    (total, section) => total + section.text.trim().split(/\s+/).filter(Boolean).length,
    0,
  );
  return { words, minutes: Math.max(1, Math.round(words / WORDS_PER_MINUTE)) };
}

function countCitations(sections) {
  return sections.reduce((total, section) => {
    const matches = section.text.match(/\[S\d{1,3}\]/g);
    return total + (matches ? matches.length : 0);
  }, 0);
}


export default function LiteratureReview() {
  const { effectiveIds, selectedIds, indexedPapers, llmReady } = useCorpus();
  const flash = useFlash();

  const [topic, setTopic] = useState('');
  const [focus, setFocus] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [review, setReview] = useState(null);
  const [saved, setSaved] = useState([]);

  const loadSaved = useCallback(async () => {
    try {
      setSaved(await api.listReviews());
    } catch {
      // A failed list must not hide the generator itself.
    }
  }, []);

  useEffect(() => {
    loadSaved();
  }, [loadSaved]);

  const generate = async () => {
    if (topic.trim().length < 3) {
      flash.error('Give the review a topic first.');
      return;
    }

    setBusy(true);
    setError(null);
    try {
      const payload = await api.generateReview({
        topic: topic.trim(),
        paper_ids: effectiveIds,
        focus: focus.trim() || null,
      });
      setReview(payload);
      flash.success(`Review written from ${payload.paper_ids.length} papers.`);
      loadSaved();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const open = async (id) => {
    setBusy(true);
    try {
      setReview(await api.getReview(id));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (id) => {
    try {
      await api.deleteReview(id);
      if (review?.id === id) setReview(null);
      flash.info('Review deleted.');
      loadSaved();
    } catch (err) {
      flash.error(err.message);
    }
  };

  const byMarker = Object.fromEntries((review?.citations || []).map((c) => [c.marker, c]));
  const stats = readingStats(review?.sections || []);
  const papersInScope = selectedIds.length || indexedPapers.length;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Literature review</h1>
          <p className="page-sub">
            A written review of your corpus in ten sections, with every claim cited back to the
            paper it came from.
          </p>
        </div>
        <button
          type="button"
          className="btn btn-primary"
          onClick={generate}
          disabled={busy || indexedPapers.length === 0}
        >
          {busy ? 'Writing…' : 'Generate review'}
        </button>
      </header>

      {!llmReady && (
        <WarningBanner>Gemini is not configured — this agent will return an error.</WarningBanner>
      )}
      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      <div className="grid-side">
        <div>
          <Card title="What should the review be about?">
            <label className="field">
              Topic
              <input
                className="input"
                placeholder="e.g. machine learning for speech disorder detection"
                value={topic}
                onChange={(event) => setTopic(event.target.value)}
                onKeyDown={(event) => event.key === 'Enter' && generate()}
              />
            </label>

            {!review && (
              <div className="suggestion-grid">
                {SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    className="suggestion"
                    onClick={() => setTopic(suggestion)}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            )}

            <label className="field">
              Particular focus (optional)
              <input
                className="input"
                placeholder="e.g. emphasise evaluation methodology and dataset bias"
                value={focus}
                onChange={(event) => setFocus(event.target.value)}
              />
            </label>

            <p className="muted">
              Writing {papersInScope} paper{papersInScope === 1 ? '' : 's'} into ten sections.
              Costs <strong>3 API requests</strong> whatever the corpus size — the sections are
              written in three passes and the reference list is built from stored metadata for
              nothing.
            </p>
          </Card>

          {busy && !review && <Spinner label="Reading the corpus and writing…" />}

          {review && (
            <>
              <article className="review-doc">
                <header className="review-head">
                  <span className="review-kicker">Literature review</span>
                  <h2 className="review-title">{review.topic}</h2>

                  <div className="review-meta">
                    <span>
                      <strong>{review.citations.length}</strong> papers
                    </span>
                    <span>
                      <strong>{countCitations(review.sections)}</strong> citations
                    </span>
                    <span>
                      <strong>{stats.words.toLocaleString()}</strong> words
                    </span>
                    <span>~{stats.minutes} min read</span>
                    <Badge tone={review.status === 'completed' ? 'success' : 'warning'}>
                      {review.status}
                    </Badge>
                  </div>

                  <div className="review-actions">
                    {review.id && (
                      <>
                        <a className="btn btn-ghost btn-sm" href={api.reviewMarkdownUrl(review.id)}>
                          ↓ Markdown
                        </a>
                        <a className="btn btn-ghost btn-sm" href={api.reviewPdfUrl(review.id)}>
                          ↓ PDF
                        </a>
                      </>
                    )}
                    <span className="review-hint">
                      Every <span className="cite-link">S1</span> opens that paper in the reader
                    </span>
                  </div>
                </header>

                {review.errors?.length > 0 && (
                  <div className="callout callout-warning review-note">
                    <strong>Some sections could not be written.</strong>
                    <ul className="bullets">
                      {review.errors.map((message) => (
                        <li key={message}>{message}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {review.sections.map((section, index) => (
                  <section key={section.key} id={`sec-${section.key}`} className="review-section">
                    <h3 className="review-section-head">
                      <span className="review-section-number">{index + 1}</span>
                      {/* The number is already in the badge, so it is stripped
                          from the title rather than printed twice. */}
                      {section.title.replace(/^\d+\.\s*/, '')}
                    </h3>
                    <Prose text={section.text} byMarker={byMarker} lead={index === 0} />
                  </section>
                ))}

                <section id="sec-references" className="review-section">
                  <h3 className="review-section-head">
                    <span className="review-section-number">10</span>
                    References
                  </h3>
                  <ol className="review-references">
                    {review.citations.map((citation) => (
                      <li key={citation.marker} id={`cite-${citation.marker}`}>
                        <Link className="cite-link" to={`/app/reader/${citation.paper_id}`}>
                          {citation.marker}
                        </Link>
                        <span className="reference-body">
                          {(citation.authors || []).slice(0, 3).join(', ')}
                          {citation.authors?.length > 3 ? ' et al.' : ''}
                          {citation.year ? ` (${citation.year}). ` : ' '}
                          <em>{citation.title}</em>
                          {citation.venue ? `. ${citation.venue}` : ''}
                          {citation.doi ? ` · doi:${citation.doi}` : ''}
                        </span>
                      </li>
                    ))}
                  </ol>
                </section>
              </article>
            </>
          )}
        </div>

        <aside>
          <Card title="Corpus scope">
            <PaperPicker compact />
          </Card>

          <Card title="Saved reviews">
            {saved.length === 0 ? (
              <p className="muted">None yet.</p>
            ) : (
              <ul className="saved-list">
                {saved.map((item) => (
                  <li key={item.id}>
                    <button type="button" className="saved-open" onClick={() => open(item.id)}>
                      <span className="saved-topic">{item.topic}</span>
                      <span className="cell-sub">
                        {item.paper_ids.length} papers · {item.section_count} sections ·{' '}
                        {new Date(item.created_at).toLocaleDateString()}
                      </span>
                    </button>
                    <button
                      type="button"
                      className="btn btn-ghost btn-sm"
                      onClick={() => remove(item.id)}
                    >
                      Delete
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card title={review ? 'Contents' : 'What it writes'}>
            {review ? (
              <ol className="section-list is-nav">
                {review.sections.map((section) => (
                  <li key={section.key}>
                    <a href={`#sec-${section.key}`}>{section.title.replace(/^\d+\.\s*/, '')}</a>
                  </li>
                ))}
                <li>
                  <a href="#sec-references">References</a>
                </li>
              </ol>
            ) : (
              <ol className="section-list">
                <li>Introduction</li>
                <li>Research Evolution</li>
                <li>Existing Approaches</li>
                <li>Dataset Landscape</li>
                <li>Method Comparison</li>
                <li>Conflicting Findings</li>
                <li>Research Gaps</li>
                <li>Open Problems</li>
                <li>Proposed Research Directions</li>
                <li>References</li>
              </ol>
            )}
          </Card>
        </aside>
      </div>

      {!review && !busy && indexedPapers.length === 0 && (
        <EmptyState
          title="No papers yet"
          description="Upload a few papers and the review has something to be written from."
        />
      )}
    </div>
  );
}
