export function Spinner({ label = 'Working…' }) {
  return (
    <div className="spinner" role="status" aria-live="polite">
      <span className="spinner-dot" />
      <span>{label}</span>
    </div>
  );
}

export function ErrorBanner({ error, onDismiss }) {
  if (!error) return null;
  return (
    <div className="banner banner-error" role="alert">
      <div>
        <strong>Something went wrong.</strong>
        <div className="banner-detail">{String(error)}</div>
      </div>
      {onDismiss && (
        <button type="button" className="btn btn-ghost btn-sm" onClick={onDismiss}>
          Dismiss
        </button>
      )}
    </div>
  );
}

export function WarningBanner({ children }) {
  return (
    <div className="banner banner-warning" role="status">
      <div>{children}</div>
    </div>
  );
}

export function EmptyState({ title, description, action }) {
  return (
    <div className="empty-state">
      <h3>{title}</h3>
      {description && <p>{description}</p>}
      {action}
    </div>
  );
}

export function Card({ title, subtitle, actions, children, className = '' }) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="card-header">
          <div>
            {title && <h2 className="card-title">{title}</h2>}
            {subtitle && <p className="card-subtitle">{subtitle}</p>}
          </div>
          {actions && <div className="card-actions">{actions}</div>}
        </header>
      )}
      <div className="card-body">{children}</div>
    </section>
  );
}

export function Badge({ children, tone = 'neutral' }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

export function Stat({ label, value, hint }) {
  return (
    <div className="stat">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}

export function BulletList({ items, empty = 'None reported.' }) {
  if (!items || items.length === 0) return <p className="muted">{empty}</p>;
  return (
    <ul className="bullets">
      {items.map((item, index) => (
        <li key={index}>{typeof item === 'string' ? item : JSON.stringify(item)}</li>
      ))}
    </ul>
  );
}

export function statusTone(status) {
  switch (status) {
    case 'indexed':
    case 'completed':
    case 'high':
      return 'success';
    case 'failed':
      return 'danger';
    case 'partial':
    case 'medium':
      return 'warning';
    case 'duplicate':
    case 'low':
      return 'neutral';
    default:
      return 'info';
  }
}
