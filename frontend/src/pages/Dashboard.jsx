import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../api/client';
import { Card, ErrorBanner, Spinner, Stat, WarningBanner } from '../components/common';
import { useCorpus } from '../context/CorpusContext';

const PIPELINE = [
  ['1', 'Upload', 'PDFs arrive through the React client'],
  ['2', 'Extract', 'PyMuPDF pulls text, sections and metadata'],
  ['3', 'Chunk', 'RecursiveCharacterTextSplitter, page-aware'],
  ['4', 'Embed', 'sentence-transformers MiniLM vectors'],
  ['5', 'Index', 'ChromaDB persistent collection'],
  ['6', 'Agents', 'LangGraph orchestrates six Gemini agents'],
  ['7', 'Graph', 'Entities and relations land in Neo4j'],
  ['8', 'Report', 'Markdown + PDF literature review'],
];

export default function Dashboard() {
  const { health, indexedPapers, papers, llmReady } = useCorpus();
  const [stats, setStats] = useState(null);
  const [runs, setRuns] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [statsPayload, runsPayload] = await Promise.all([
          api.stats(),
          api.listRuns({ limit: 8 }),
        ]);
        if (cancelled) return;
        setStats(statsPayload);
        setRuns(runsPayload);
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [papers.length]);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Dashboard</h1>
          <p className="page-sub">
            Corpus overview, pipeline status, and recent multi-agent activity.
          </p>
        </div>
        <Link className="btn btn-primary" to="/upload">
          Upload papers
        </Link>
      </header>

      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      {!llmReady && (
        <WarningBanner>
          <strong>Gemini is not configured.</strong> Upload, chunking, embedding and semantic
          search all work, but the agents need an API key. Add <code>GOOGLE_API_KEY</code> to{' '}
          <code>backend/.env</code> and restart the API.
        </WarningBanner>
      )}

      {health?.embeddings?.degraded && (
        <WarningBanner>
          <strong>Fallback embeddings in use.</strong> The sentence-transformers model could not be
          loaded, so a deterministic hashing embedder is active. Retrieval still works but is less
          semantically accurate.
        </WarningBanner>
      )}

      {loading ? (
        <Spinner label="Loading corpus statistics…" />
      ) : (
        <>
          <div className="stat-grid">
            <Stat label="Papers indexed" value={stats?.indexed ?? 0} hint={`${stats?.papers ?? 0} uploaded`} />
            <Stat label="Text chunks" value={stats?.chunks ?? 0} />
            <Stat label="Vectors" value={stats?.vectors ?? 0} hint={health?.embeddings?.backend} />
            <Stat
              label="Graph nodes"
              value={stats?.graph?.node_count ?? 0}
              hint={`${stats?.graph?.edge_count ?? 0} relations`}
            />
            <Stat label="Agent runs" value={stats?.agent_runs ?? 0} />
            <Stat label="Reports" value={stats?.reports ?? 0} />
          </div>

          <div className="grid-2">
            <Card title="Processing pipeline" subtitle="Every uploaded paper travels this path">
              <ol className="pipeline">
                {PIPELINE.map(([step, name, detail]) => (
                  <li key={step}>
                    <span className="pipeline-step">{step}</span>
                    <span className="pipeline-name">{name}</span>
                    <span className="pipeline-detail">{detail}</span>
                  </li>
                ))}
              </ol>
            </Card>

            <Card title="Recent agent runs" subtitle="LangGraph execution history">
              {runs.length === 0 ? (
                <p className="muted">No agent runs yet.</p>
              ) : (
                <table className="table">
                  <thead>
                    <tr>
                      <th>Intent</th>
                      <th>Status</th>
                      <th>Papers</th>
                      <th>Duration</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map((run) => (
                      <tr key={run.id}>
                        <td>
                          <code>{run.intent}</code>
                          {run.question && <div className="cell-sub">{run.question}</div>}
                        </td>
                        <td>{run.status}</td>
                        <td>{run.paper_ids?.length ?? 0}</td>
                        <td>{run.duration_ms} ms</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Card>
          </div>

          {stats?.papers_by_year?.length > 0 && (
            <Card title="Publication years" subtitle="Distribution across the corpus">
              <div className="year-chart">
                {stats.papers_by_year.map(({ year, count }) => {
                  const max = Math.max(...stats.papers_by_year.map((y) => y.count));
                  return (
                    <div key={year} className="year-bar">
                      <div
                        className="year-bar-fill"
                        style={{ height: `${(count / max) * 100}%` }}
                        title={`${count} paper(s)`}
                      />
                      <span className="year-label">{year}</span>
                    </div>
                  );
                })}
              </div>
            </Card>
          )}

          {indexedPapers.length === 0 && (
            <Card title="Get started">
              <p>
                Upload three to five related papers, then run the{' '}
                <Link to="/gaps">Research Gap agent</Link> to see where the literature stops.
              </p>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
