import { NavLink, Outlet } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useCorpus } from '../context/CorpusContext';

const NAV = [
  { to: '/app', label: 'Dashboard', icon: '◎', end: true },
  { to: '/app/upload', label: 'Upload', icon: '↑' },
  { to: '/app/library', label: 'Library', icon: '▤' },
  { to: '/app/ask', label: 'Ask (RAG)', icon: '?' },
  { to: '/app/summaries', label: 'Summaries', icon: '≡' },
  { to: '/app/extraction', label: 'Extraction', icon: '⌗' },
  { to: '/app/figures', label: 'Figures', icon: '▩' },
  { to: '/app/matrix', label: 'Comparison', icon: '⊞' },
  { to: '/app/discover', label: 'Discover', icon: '✦' },
  { to: '/app/gaps', label: 'Research Gaps', icon: '◇' },
  { to: '/app/graph', label: 'Knowledge Graph', icon: '⁂' },
  { to: '/app/reports', label: 'Reports', icon: '▦' },
];

function initials(user) {
  const letters = `${user.first_name?.[0] || ''}${user.last_name?.[0] || ''}`.trim();
  return (letters || user.username.slice(0, 2)).toUpperCase();
}

export default function Layout() {
  const { health, indexedPapers, llmReady } = useCorpus();
  const { user, signOut } = useAuth();

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">RC</span>
          <div>
            <div className="brand-name">ResearchCompass</div>
            <div className="brand-tag">Multi-agent literature analysis</div>
          </div>
        </div>

        <nav className="nav">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `nav-item ${isActive ? 'is-active' : ''}`}
            >
              <span className="nav-icon" aria-hidden="true">
                {item.icon}
              </span>
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className={`status-dot ${llmReady ? 'is-ok' : 'is-warn'}`} />
          <div>
            <div className="status-line">
              Gemini {llmReady ? 'connected' : 'not configured'}
            </div>
            <div className="status-sub">
              {indexedPapers.length} indexed · graph: {health?.graph?.backend || 'n/a'}
            </div>
            <div className="status-sub">
              OCR: {health?.ocr?.engine || (health?.ocr?.enabled ? 'unavailable' : 'off')}
            </div>
            {health?.embeddings?.degraded && (
              <div className="status-sub status-warn">embeddings: fallback mode</div>
            )}
          </div>
        </div>

        {user && (
          <div className="sidebar-profile">
            <NavLink
              to="/app/profile"
              className={({ isActive }) => `profile-link ${isActive ? 'is-active' : ''}`}
            >
              <span className="avatar" aria-hidden="true">
                {initials(user)}
              </span>
              <span className="profile-text">
                <span className="profile-name">{user.full_name}</span>
                <span className="profile-sub">{user.email}</span>
              </span>
            </NavLink>
            <button
              type="button"
              className="profile-signout"
              onClick={signOut}
              title="Sign out"
              aria-label="Sign out"
            >
              ⏻
            </button>
          </div>
        )}
      </aside>

      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
