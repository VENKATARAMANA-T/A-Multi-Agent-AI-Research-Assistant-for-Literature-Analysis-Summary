import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../api/client';
import PaperPicker from '../components/PaperPicker';
import { Badge, Card, EmptyState, ErrorBanner, Spinner, WarningBanner } from '../components/common';
import { useCorpus } from '../context/CorpusContext';
import { useFlash } from '../context/FlashContext';

const CITATION_RE = /\[(S\d{1,3})\]/g;

const SUGGESTIONS = [
  'Speech-based detection of neurodevelopmental and speech disorders',
  'Self-supervised audio representations for clinical speech analysis',
  'Privacy-preserving machine learning for speech data',
];

/**
 * Renders a paragraph, turning every [S#] marker into a link to that paper.
 *
 * The citation is the point of the whole feature — a review you cannot follow
 * back to its sources is just prose — so the markers have to be clickable
 * rather than literal text.
 */
function Cited({ text, byMarker }) {
  const parts = [];
  let last = 0;
  let match;

  CITATION_RE.lastIndex = 0;
  while ((match = CITATION_RE.exec(text)) !== null) {
    if (match.index > last) parts.push(text.slice(last, match.index));

    const citation = byMarker[match[1]];
    parts.push(
      citation ? (
        <Link
          key={`${match[1]}-${match.index}`}
          className="cite-link"
          to={`/app/reader/${citation.paper_id}`}
          title={citation.title}
        >
          [{match[1]}]
        </Link>
      ) : (
        // Should not happen — the backend drops unresolvable markers — but a
        // stray one renders as plain text rather than a dead link.
        <span key={`${match[1]}-${match.index}`}>[{match[1]}]</span>
      ),
    );
    last = match.index + match[0].length;
  }

  if (last < text.length) parts.push(text.slice(last));
  return <>{parts}</>;
}

/** Section prose arrives as Markdown paragraphs; split them and keep the citations live. */
function Prose({ text, byMarker }) {
  return (
    <>
      {text
        .split(/\n{2,}/)
        .map((paragraph) => paragraph.trim())
        .filter(Boolean)
        .map((paragraph, index) => (
          <p key={index} className="review-para">
            <Cited text={paragraph} byMarker={byMarker} />
          </p>
        ))}
    </>
  );
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
              <Card
                title={`Literature Review: ${review.topic}`}
                subtitle={`${review.sections.length} sections · ${review.citations.length} papers cited · ${review.llm_calls} API requests`}
                actions={
                  <>
                    <Badge tone={review.status === 'completed' ? 'success' : 'warning'}>
                      {review.status}
                    </Badge>
                    {review.id && (
                      <>
                        <a className="btn btn-ghost btn-sm" href={api.reviewMarkdownUrl(review.id)}>
                          Markdown
                        </a>
                        <a className="btn btn-ghost btn-sm" href={api.reviewPdfUrl(review.id)}>
                          PDF
                        </a>
                      </>
                    )}
                  </>
                }
              >
                {review.errors?.length > 0 && (
                  <div className="callout callout-warning">
                    <strong>Some sections could not be written.</strong>
                    <ul className="bullets">
                      {review.errors.map((message) => (
                        <li key={message}>{message}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <p className="muted">
                  Every <span className="cite-link">[S#]</span> opens that paper in the reader.
                </p>
              </Card>

              <article className="review-doc">
                {review.sections.map((section) => (
                  <section key={section.key} className="review-section">
                    <h2>{section.title}</h2>
                    <Prose text={section.text} byMarker={byMarker} />
                  </section>
                ))}

                <section className="review-section">
                  <h2>10. References</h2>
                  <ol className="review-references">
                    {review.citations.map((citation) => (
                      <li key={citation.marker}>
                        <Link className="cite-link" to={`/app/reader/${citation.paper_id}`}>
                          [{citation.marker}]
                        </Link>{' '}
                        {(citation.authors || []).slice(0, 3).join(', ')}
                        {citation.authors?.length > 3 ? ' et al.' : ''}
                        {citation.year ? ` (${citation.year}). ` : '. '}
                        <em>{citation.title}</em>
                        {citation.venue ? `. ${citation.venue}` : ''}
                        {citation.doi ? ` · doi:${citation.doi}` : ''}
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

          <Card title="What it writes">
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
