import { useCallback, useEffect, useState } from 'react';
import api from '../api/client';
import { Badge, Card, EmptyState, ErrorBanner, Spinner, WarningBanner } from '../components/common';
import { useCorpus } from '../context/CorpusContext';

const VERDICT_TONES = {
  promising: 'success',
  plausible: 'info',
  trivial: 'neutral',
  implausible: 'danger',
};

const MODES = [
  [
    'strict',
    'Separate literatures',
    'Swanson’s criterion: the two concepts share no paper at all. This is a genuine discovery, and needs a large corpus to find anything.',
  ],
  [
    'unstated',
    'Unstated links',
    'Only requires that no paper states the connection directly. Far more productive on a small corpus, but some results will be obvious.',
  ],
];

export default function Hypotheses() {
  const { health } = useCorpus();
  const [terms, setTerms] = useState([]);
  const [source, setSource] = useState('');
  const [mode, setMode] = useState('unstated');
  const [assess, setAssess] = useState(false);
  const [result, setResult] = useState(null);
  const [saved, setSaved] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(null);

  const load = useCallback(async () => {
    try {
      const [available, stored] = await Promise.all([
        api.lbdTerms({ limit: 40 }),
        api.listHypotheses(),
      ]);
      setTerms(available);
      setSaved(stored);
      if (!source && available.length) setSource(available[0].name);
    } catch (err) {
      setError(err.message);
    }
    // `source` intentionally omitted: this only seeds the initial choice.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const run = async () => {
    if (!source.trim()) {
      setError('Choose a concept to start from.');
      return;
    }
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const payload = await api.runLbd({
        source: source.trim(),
        mode,
        assess,
        assess_limit: 5,
        limit: 15,
      });
      setResult(payload);
      if (assess) await load();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const star = async (id, next) => {
    try {
      await api.starHypothesis(id, next);
      await load();
    } catch (err) {
      setError(err.message);
    }
  };

  const remove = async (id) => {
    if (!window.confirm('Delete this hypothesis?')) return;
    try {
      await api.deleteHypothesis(id);
      await load();
    } catch (err) {
      setError(err.message);
    }
  };

  const diagnostics = result?.diagnostics;
  const graphEmpty = terms.length === 0;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Hypotheses</h1>
          <p className="page-sub">
            Connections the literature implies but never states. If one paper links A to B and
            another links B to C, while nothing links A to C, then A–C is worth investigating.
          </p>
        </div>
        <button type="button" className="btn btn-primary" onClick={run} disabled={busy || graphEmpty}>
          {busy ? 'Searching…' : 'Find connections'}
        </button>
      </header>

      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      {graphEmpty && (
        <WarningBanner>
          <strong>The knowledge graph is empty.</strong> This method reads the graph, not the text,
          so build it first on the Knowledge Graph page. Results improve sharply with corpus size —
          the technique was designed for literatures of thousands of papers.
        </WarningBanner>
      )}

      <div className="grid-side">
        <div>
          <Card title="Search">
            <label className="field">
              Starting concept
              <input
                className="input"
                list="lbd-terms"
                placeholder="e.g. Dysarthria"
                value={source}
                onChange={(event) => setSource(event.target.value)}
                onKeyDown={(event) => event.key === 'Enter' && run()}
              />
              <datalist id="lbd-terms">
                {terms.map((term) => (
                  <option key={term.id} value={term.name}>
                    {term.type} · {term.degree} links
                  </option>
                ))}
              </datalist>
            </label>

            <div className="radio-row">
              {MODES.map(([key, label, description]) => (
                <label key={key} className={`radio-card ${mode === key ? 'is-active' : ''}`}>
                  <input
                    type="radio"
                    name="lbd-mode"
                    checked={mode === key}
                    onChange={() => setMode(key)}
                  />
                  <span>
                    <strong>{label}</strong>
                    <small>{description}</small>
                  </span>
                </label>
              ))}
            </div>

            <label className="field-check">
              <input
                type="checkbox"
                checked={assess}
                onChange={(event) => setAssess(event.target.checked)}
                disabled={!health?.llm?.configured}
              />
              Have the top 5 assessed and turned into testable hypotheses — costs 5 API requests
            </label>
          </Card>

          {busy && <Spinner label="Walking the graph…" />}

          {result && (
            <Card
              title={`${result.candidates.length} implied connection${result.candidates.length === 1 ? '' : 's'}`}
              subtitle={`from ${result.source} · ${result.mode === 'strict' ? 'separate literatures' : 'unstated links'}`}
            >
              {diagnostics && (
                <p className="muted">
                  Walked {diagnostics.b_terms ?? 0} neighbouring terms and examined{' '}
                  {diagnostics.c_examined ?? 0} candidates.{' '}
                  {diagnostics.rejected_already_linked > 0 &&
                    `${diagnostics.rejected_already_linked} were already stated directly. `}
                  {diagnostics.rejected_shared_paper > 0 &&
                    `${diagnostics.rejected_shared_paper} appeared in the same paper, so they are not separate literatures.`}
                </p>
              )}

              {result.errors?.length > 0 && (
                <ul className="bullets">
                  {result.errors.map((message, index) => (
                    <li key={index} className="error-text">
                      {message}
                    </li>
                  ))}
                </ul>
              )}

              {result.candidates.length === 0 ? (
                <EmptyState
                  title="Nothing found"
                  description={
                    result.mode === 'strict'
                      ? 'No pair of concepts is linked indirectly while appearing in entirely separate papers. Try “Unstated links”, or add more papers.'
                      : 'No indirect connections from this concept. Try a more central term.'
                  }
                />
              ) : (
                <div className="candidate-list">
                  {result.candidates.map((candidate) => {
                    const judgement = candidate.assessment;
                    const key = `${candidate.a_id}:${candidate.c_id}`;
                    return (
                      <div key={key} className="candidate">
                        <div className="candidate-head">
                          <strong>
                            {candidate.a_name} <span className="muted">→</span> {candidate.c_name}
                          </strong>
                          <Badge tone="neutral">{candidate.c_type}</Badge>
                          {judgement && (
                            <Badge tone={VERDICT_TONES[judgement.verdict] || 'neutral'}>
                              {judgement.verdict}
                            </Badge>
                          )}
                          <span className="muted">
                            {candidate.support} bridge{candidate.support === 1 ? '' : 's'} · score{' '}
                            {candidate.score}
                          </span>
                        </div>

                        <ul className="bullets chain-paths">
                          {candidate.chains.slice(0, 3).map((chain, index) => (
                            <li key={index}>
                              <code>{candidate.a_name}</code>
                              <span className="muted"> —{chain.a_to_b}→ </span>
                              <strong>{chain.b_name}</strong>
                              <span className="muted"> —{chain.b_to_c}→ </span>
                              <code>{candidate.c_name}</code>
                            </li>
                          ))}
                        </ul>

                        {judgement?.hypothesis && (
                          <div className="direction">
                            <strong>Hypothesis</strong>
                            <p>{judgement.hypothesis}</p>
                          </div>
                        )}

                        {judgement?.why_not && (
                          <p className="muted">
                            <strong>Rejected:</strong> {judgement.why_not}
                          </p>
                        )}

                        {judgement && (
                          <button
                            type="button"
                            className="btn btn-ghost btn-sm"
                            onClick={() => setExpanded(expanded === key ? null : key)}
                          >
                            {expanded === key ? 'Less' : 'More'}
                          </button>
                        )}

                        {expanded === key && judgement && (
                          <div className="figure-detail">
                            {judgement.reasoning && (
                              <>
                                <h4>Reasoning</h4>
                                <p>{judgement.reasoning}</p>
                              </>
                            )}
                            {judgement.mechanism && (
                              <>
                                <h4>Possible mechanism</h4>
                                <p>{judgement.mechanism}</p>
                              </>
                            )}
                            {judgement.proposed_test && (
                              <>
                                <h4>How to test it</h4>
                                <p>{judgement.proposed_test}</p>
                              </>
                            )}
                            <p className="muted">
                              Novelty: {judgement.novelty} · Confidence: {judgement.confidence}
                            </p>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </Card>
          )}
        </div>

        <aside>
          <Card title="Saved hypotheses">
            {saved.length === 0 ? (
              <p className="muted">None yet. Assess some candidates to keep them.</p>
            ) : (
              <ul className="report-list">
                {saved.map((item) => (
                  <li key={item.id}>
                    <button
                      type="button"
                      className="report-item"
                      onClick={() => star(item.id, !item.starred)}
                      title={item.starred ? 'Unstar' : 'Star'}
                    >
                      <strong>
                        {item.starred ? '★ ' : ''}
                        {item.source_term} → {item.target_term}
                      </strong>
                      <span className="muted">
                        {item.verdict}
                        {item.novelty ? ` · ${item.novelty} novelty` : ''}
                      </span>
                    </button>
                    <button type="button" className="btn btn-ghost btn-sm" onClick={() => remove(item.id)}>
                      ✕
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card title="How this works">
            <p className="muted">
              Swanson found in 1986 that one body of medical work linked dietary fish oil to blood
              viscosity, while a separate body linked blood viscosity to Raynaud&rsquo;s disease —
              and no paper mentioned both ends. The connection was implied by the literature
              without anyone having stated it, and later trials supported it.
            </p>
            <p className="muted">
              Intermediate terms are weighted by how many things they connect to. A term linked to
              everything — &ldquo;model&rdquo;, &ldquo;accuracy&rdquo; — links everything to
              everything and is discounted heavily, which is what keeps the ranking meaningful.
            </p>
          </Card>
        </aside>
      </div>
    </div>
  );
}
