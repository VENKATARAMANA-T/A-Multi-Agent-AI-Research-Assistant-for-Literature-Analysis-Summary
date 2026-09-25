import { useState } from 'react';
import api from '../api/client';
import { Badge, ErrorBanner, Spinner } from './common';

const VERDICTS = {
  supported: ['success', 'Supported', 'The evidence states this.'],
  partially_supported: ['warning', 'Partly supported', 'Some of it holds, or it claims more than the evidence shows.'],
  unsupported: ['danger', 'Unsupported', 'Nothing in the corpus addresses this.'],
  contradicted: ['danger', 'Contradicted', 'The evidence says something different.'],
  unverifiable: ['neutral', 'Not checkable', 'An opinion or hedge that no evidence could settle.'],
};

const scoreTone = (score) => (score >= 0.85 ? 'success' : score >= 0.6 ? 'warning' : 'danger');

const scoreLabel = (score) =>
  score >= 0.85 ? 'Well grounded' : score >= 0.6 ? 'Mostly grounded' : 'Weakly grounded';

/**
 * "Verify this" — runs the Verification Agent over a piece of generated text.
 *
 * The agent retrieves its own evidence for each claim rather than reusing what
 * produced the text, so a green badge here means the corpus independently backs
 * the statement, not that the model agreed with itself.
 */
export default function VerificationBadge({
  text = null,
  paperIds = [],
  runId = null,
  source = 'text',
  subject = null,
  label = 'Verify this',
}) {
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [open, setOpen] = useState(false);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const payload = await api.verify({
        text,
        agent_run_id: runId,
        paper_ids: paperIds,
        source,
        subject,
      });
      setResult(payload);
      setOpen(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  // With a run id the backend pulls the checkable prose out of the stored
  // result itself, so the caller need not flatten a structured summary by hand.
  if (!runId && (!text || text.trim().length < 20)) return null;

  return (
    <div className="verification">
      {!result && !busy && (
        <button type="button" className="btn btn-ghost btn-sm" onClick={run} title="Two model calls">
          ✓ {label}
          <span className="muted"> · 2 calls</span>
        </button>
      )}

      {busy && <Spinner label="Re-retrieving evidence and checking each claim…" />}

      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      {result && (
        <>
          <div className="verification-head">
            <Badge tone={scoreTone(result.score)}>
              {scoreLabel(result.score)} · {Math.round(result.score * 100)}%
            </Badge>
            <span className="muted">
              {result.checked} claim{result.checked === 1 ? '' : 's'} checked
              {result.problems?.length > 0 && ` · ${result.problems.length} need attention`}
            </span>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setOpen(!open)}>
              {open ? 'Hide' : 'Show'} claims
            </button>
          </div>

          {result.errors?.length > 0 && (
            <p className="error-text">{result.errors.join(' ')}</p>
          )}

          {open && (
            <ul className="claim-list">
              {result.claims.map((claim, index) => {
                const [tone, title, hint] = VERDICTS[claim.verdict] || VERDICTS.unverifiable;
                return (
                  <li key={index} className={`claim claim-${tone}`}>
                    <div className="claim-head">
                      <Badge tone={tone}>{title}</Badge>
                      {claim.confidence && <span className="muted">{claim.confidence} confidence</span>}
                    </div>
                    <p className="claim-text">{claim.text}</p>
                    {claim.explanation && <p className="claim-why">{claim.explanation}</p>}
                    {claim.evidence_quote && (
                      <blockquote className="claim-quote">“{claim.evidence_quote}”</blockquote>
                    )}
                    {claim.evidence?.length > 0 && (
                      <details className="claim-evidence">
                        <summary>
                          {claim.evidence.length} passage
                          {claim.evidence.length === 1 ? '' : 's'} retrieved for this claim
                        </summary>
                        <ul>
                          {claim.evidence.map((item) => (
                            <li key={item.chunk_id}>
                              <span className="muted">
                                {item.paper_title} · p.{item.page ?? '?'} · score {item.score}
                              </span>
                              <p>{item.text.slice(0, 300)}…</p>
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                    {!claim.explanation && <p className="claim-why muted">{hint}</p>}
                  </li>
                );
              })}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
