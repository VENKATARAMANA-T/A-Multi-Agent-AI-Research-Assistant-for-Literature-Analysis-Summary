import { useState } from 'react';
import { Badge } from './common';

const TONES = {
  ok: 'success',
  error: 'danger',
  unavailable: 'danger',
  skipped: 'neutral',
  warning: 'warning',
  empty: 'neutral',
};

/** Renders the LangGraph execution trace so agent behaviour is inspectable. */
export default function AgentTrace({ trace, errors, llmCalls, durationMs }) {
  const [open, setOpen] = useState(false);
  if (!trace || trace.length === 0) return null;

  return (
    <div className="trace">
      <button type="button" className="trace-toggle" onClick={() => setOpen((v) => !v)}>
        {open ? '▾' : '▸'} Agent trace · {trace.length} nodes · {llmCalls ?? 0} LLM calls ·{' '}
        {durationMs ?? 0} ms
      </button>

      {open && (
        <div className="trace-body">
          <ol className="trace-list">
            {trace.map((event, index) => (
              <li key={index} className="trace-item">
                <Badge tone={TONES[event.status] || 'info'}>{event.status}</Badge>
                <code className="trace-node">{event.node}</code>
                <span className="trace-duration">{event.duration_ms} ms</span>
                <div className="trace-details">
                  {Object.entries(event)
                    .filter(([key]) => !['node', 'status', 'duration_ms'].includes(key))
                    .map(([key, value]) => (
                      <span key={key} className="trace-detail">
                        {key}={typeof value === 'object' ? JSON.stringify(value) : String(value)}
                      </span>
                    ))}
                </div>
              </li>
            ))}
          </ol>

          {errors && errors.length > 0 && (
            <div className="trace-errors">
              <strong>Errors</strong>
              <ul>
                {errors.map((error, index) => (
                  <li key={index}>{error}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
