import { NavLink, Outlet } from 'react-router-dom';
import { useCorpus } from '../context/CorpusContext';

const NAV = [
  { to: '/', label: 'Dashboard', icon: '◎', end: true },
  { to: '/upload', label: 'Upload', icon: '↑' },
  { to: '/library', label: 'Library', icon: '▤' },
  { to: '/ask', label: 'Ask (RAG)', icon: '?' },
  { to: '/summaries', label: 'Summaries', icon: '≡' },
  { to: '/extraction', label: 'Extraction', icon: '⌗' },
  { to: '/figures', label: 'Figures', icon: '▩' },
  { to: '/matrix', label: 'Comparison', icon: '⊞' },
  { to: '/discover', label: 'Discover', icon: '✦' },
  { to: '/gaps', label: 'Research Gaps', icon: '◇' },
  { to: '/graph', label: 'Knowledge Graph', icon: '⁂' },
  { to: '/reports', label: 'Reports', icon: '▦' },
];

export default function Layout() {
  const { health, indexedPapers, llmReady } = useCorpus();

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
      </aside>

      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
